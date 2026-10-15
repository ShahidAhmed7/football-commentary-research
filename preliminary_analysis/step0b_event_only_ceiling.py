import json, os, re, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
SN = os.environ.get("SN_LABELS_DIR", "data/soccernet_labels")
LFCBI = os.environ.get("LFCBI_DIR", "data/lfcbi/data")
OUT = os.environ.get("OUT_DIR", "preliminary_analysis/outputs")
CLASSES = ["Penalty", "Kick-off", "Goal", "Substitution", "Offside", "Shots on target", "Shots off target",
           "Clearance", "Ball out of play", "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick",
           "Corner", "Yellow card", "Red card", "Yellow->red card"]
OFFPLAY_A = ["Foul", "Goal", "Penalty", "Red card", "Yellow card", "Yellow->red card", "Substitution", "Offside", "Ball out of play"]
CAP = 300.0
rng = np.random.default_rng(0)

utt = pd.concat([pd.read_csv(f"{LFCBI}/spotting/{s}.csv",
                             usecols=["game", "half", "start", "end", "text", "language", "target_label", "split"])
                 for s in ["train", "valid", "test"]])
if LANG != "all":
    utt = utt[utt.language == LANG]
utt = utt.dropna(subset=["half", "start", "target_label"]).copy()
utt["half"] = utt.half.astype(int)
utt["y"] = (utt.target_label == 2.0).astype(int)

events = {}
for g in utt.game.unique():
    p = f"{SN}/{g}/Labels-v2.json"
    if not os.path.exists(p):
        continue
    ann = json.load(open(p))["annotations"]
    ev = pd.DataFrame([{"half": int(a["gameTime"].split(" - ")[0]), "t": int(a["position"]) / 1000.0,
                        "label": a["label"], "shown": a.get("visibility") == "visible"} for a in ann])
    events[g] = ev.sort_values("t")
n_before = utt.game.nunique()
utt = utt[utt.game.isin(events)].copy()
print(f"[{LANG}] games with labels: {utt.game.nunique()}/{n_before}; utterances: {len(utt)}; bg rate {utt.y.mean():.4f}")

# timebase sanity: goal event -> first utterance with goal vocabulary
goal_re = re.compile(r"\b(goal|scores?|scored|gol|tor|but)\b", re.I)
lags = []
for (g, h), u in utt.groupby(["game", "half"]):
    ev = events[g]; ev = ev[(ev.half == h) & (ev.label == "Goal")]
    gu = u[u.text.fillna("").str.contains(goal_re.pattern, flags=re.I)]
    for t in ev.t:
        c = gu[(gu.start >= t - 30) & (gu.start <= t + 30)]
        if len(c):
            lags.append(float(c.start.iloc[(c.start - t).abs().argmin()] - t))
lags = np.array(lags)
print(f"goal timebase lag (utt_start - event): n={len(lags)} median={np.median(lags):.2f}s "
      f"IQR=[{np.percentile(lags,25):.2f},{np.percentile(lags,75):.2f}] |lag|<=5s: {(np.abs(lags)<=5).mean():.2f}")

rows = []
for (g, h), u in utt.sort_values("start").groupby(["game", "half"]):
    ev = events[g]; ev = ev[ev.half == h]
    et, el, es = ev.t.to_numpy(), ev.label.to_numpy(), ev.shown.to_numpy()
    prev_end, prev_dur, prev_y = np.nan, np.nan, np.nan
    for r in u.itertuples():
        t = r.start
        m = et <= t
        f = {"half": h, "clock": t / 60.0}
        for c in CLASSES:
            tt = et[m & (el == c)]
            f["since_" + c] = min(t - tt[-1], CAP) if len(tt) else CAP
        f["n_ev_60"] = int(((et > t - 60) & m).sum())
        f["n_ev_15"] = int(((et > t - 15) & m).sum())
        f["last_shown"] = float(es[m][-1]) if m.any() else -1.0
        f["gap"] = t - prev_end if prev_end == prev_end else -1.0
        f["prev_dur"] = prev_dur if prev_dur == prev_dur else -1.0
        f["prev_y"] = prev_y if prev_y == prev_y else -1.0
        last15 = [(t - a, b) for a, b in zip(et[m], el[m]) if t - a <= 15]
        if any(b == "Kick-off" for _, b in last15):
            f["mori_p"] = 1.0
        elif any(b in OFFPLAY_A for _, b in last15):
            f["mori_p"] = 0.5
        else:
            f["mori_p"] = 0.185
        f.update(game=g, split=r.split, y=r.y, text=r.text if isinstance(r.text, str) else "")
        rows.append(f)
        prev_end, prev_dur, prev_y = r.end, r.end - r.start, r.y
df = pd.DataFrame(rows)
tr, va, te = (df[df.split == s] for s in ["train", "valid", "test"])
print(f"split utterances train/valid/test: {len(tr)}/{len(va)}/{len(te)}; test games {te.game.nunique()}")

EV = ["half", "clock", "n_ev_60", "n_ev_15", "last_shown"] + ["since_" + c for c in CLASSES]
SETS = {"E_events_clock": EV, "E+T_timing": EV + ["gap", "prev_dur"], "E+T+prevlabel_ORACLE": EV + ["gap", "prev_dur", "prev_y"]}

def best_thr(y, p):
    ts = np.quantile(p, np.linspace(0.5, 0.99, 50))
    return max(ts, key=lambda t: f1_score(y, p >= t))

def boot(y, p, yhat, games, B=1000):
    ug = games.unique(); idx = {g: np.where(games.to_numpy() == g)[0] for g in ug}
    f1s, aucs = [], []
    for _ in range(B):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        f1s.append(f1_score(y[s], yhat[s])); aucs.append(roc_auc_score(y[s], p[s]))
    return np.percentile(f1s, [2.5, 97.5]), np.percentile(aucs, [2.5, 97.5])

res = []
yte = te.y.to_numpy()
prior = tr.y.mean()
res.append(dict(model="prior_constant(all bg)", f1=f1_score(yte, np.ones_like(yte)), auc=0.5, ap=yte.mean()))
mori = te.mori_p.to_numpy()
f1m = np.mean([f1_score(yte, rng.random(len(yte)) < mori) for _ in range(20)])
res.append(dict(model="Mori_rule (sampled, 20 seeds)", f1=f1m, auc=roc_auc_score(yte, mori), ap=average_precision_score(yte, mori)))
for name, cols in SETS.items():
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[cols], tr.y)
    pv, pt = clf.predict_proba(va[cols])[:, 1], clf.predict_proba(te[cols])[:, 1]
    thr = best_thr(va.y.to_numpy(), pv); yhat = (pt >= thr).astype(int)
    (f1lo, f1hi), (alo, ahi) = boot(yte, pt, yhat, te.game)
    res.append(dict(model=name, f1=f1_score(yte, yhat), f1_ci=f"[{f1lo:.3f},{f1hi:.3f}]", auc=roc_auc_score(yte, pt), auc_ci=f"[{alo:.3f},{ahi:.3f}]", ap=average_precision_score(yte, pt)))
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
Xtr = vec.fit_transform(tr.text); lr = LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, tr.y)
pv, pt = lr.predict_proba(vec.transform(va.text))[:, 1], lr.predict_proba(vec.transform(te.text))[:, 1]
thr = best_thr(va.y.to_numpy(), pv); yhat = (pt >= thr).astype(int)
(f1lo, f1hi), (alo, ahi) = boot(yte, pt, yhat, te.game)
res.append(dict(model="TEXT_tfidf_LR (upper bound, sees the utterance)", f1=f1_score(yte, yhat), f1_ci=f"[{f1lo:.3f},{f1hi:.3f}]", auc=roc_auc_score(yte, pt), auc_ci=f"[{alo:.3f},{ahi:.3f}]", ap=average_precision_score(yte, pt)))
out = pd.DataFrame(res).round(4)
print(out.to_string(index=False))
out.to_csv(f"{OUT}/table_event_only_ceiling_{LANG}.csv", index=False)

# conditional bg rates in post-event windows (human/LLM labels), EN test+train
for c in ["Goal", "Yellow card", "Foul", "Corner", "Substitution"]:
    w = df[(df["since_" + c] <= 15)]
    print(f"P(bg | {c} in last 15s) = {w.y.mean():.3f} (n={len(w)})")
