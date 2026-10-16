import json, os, re, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
SN = os.environ.get("SN_LABELS_DIR", "data/soccernet_labels")
LFCBI = os.environ.get("LFCBI_DIR", "data/lfcbi/data")
# Separate from the notebook's outputs so script runs never overwrite the notebook's tables
OUT = os.environ.get("OUT_DIR", "preliminary_analysis/outputs/script_runs")
os.makedirs(OUT, exist_ok=True)

CLASSES = [
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside", "Shots on target", "Shots off target",
    "Clearance", "Ball out of play", "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick",
    "Corner", "Yellow card", "Red card", "Yellow->red card"
]
OFFPLAY_A = ["Foul", "Goal", "Penalty", "Red card", "Yellow card", "Yellow->red card", "Substitution", "Offside", "Ball out of play"]
CAP = 300.0
rng = np.random.default_rng(0)

utt = pd.concat([
    pd.read_csv(
        f"{LFCBI}/spotting/{s}.csv",
        usecols=["game", "half", "start", "end", "text", "language", "target_label", "split"]
    )
    for s in ["train", "valid", "test"]
], ignore_index=True)
if LANG != "all":
    utt = utt[utt.language == LANG]
utt = utt.dropna(subset=["half", "start", "end", "target_label", "text"]).copy()
utt["half"] = utt.half.astype(int)
utt["y"] = (utt.target_label == 2.0).astype(int)

events = {}
for g in utt.game.unique():
    p = f"{SN}/{g}/Labels-v2.json"
    if os.path.exists(p):
        ann = json.load(open(p))["annotations"]
        ev = pd.DataFrame([{
            "half": int(a["gameTime"].split(" - ")[0]),
            "t": int(a["position"]) / 1000.0,
            "label": a["label"],
            "shown": a.get("visibility") == "visible"
        } for a in ann])
        events[g] = ev.sort_values("t")

rows = []
for (g, h), u in utt.sort_values("start").groupby(["game", "half"]):
    if g not in events:
        continue
    ev = events[g]
    ev = ev[ev.half == h]
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

        # Mori et al. (ACL 2025) Sec 3.1 / spotting_module.py causal rule ([0, +15s] after event, p=0.18)
        past15 = [b for a, b in zip(et[m], el[m]) if t - a <= 15.0]
        if "Kick-off" in past15:
            f["mori_p_causal"] = 1.0
        elif any(b in OFFPLAY_A for b in past15):
            f["mori_p_causal"] = 0.5
        else:
            f["mori_p_causal"] = 0.18

        # Mori et al. Footnote 7 two-sided rule ([-15s, +15s] around event, p=0.18, non-causal)
        win15 = [b for a, b in zip(et, el) if abs(t - a) <= 15.0]
        if "Kick-off" in win15:
            f["mori_p_twosided"] = 1.0
        elif any(b in OFFPLAY_A for b in win15):
            f["mori_p_twosided"] = 0.5
        else:
            f["mori_p_twosided"] = 0.18

        f.update(game=g, split=r.split, y=r.y, text=r.text if isinstance(r.text, str) else "")
        rows.append(f)
        prev_end, prev_dur, prev_y = r.end, r.end - r.start, r.y

df = pd.DataFrame(rows)
tr, va, te = (df[df.split == s].copy() for s in ["train", "valid", "test"])
print(f"lang={LANG} games={df.game.nunique()} (tr={tr.game.nunique()}, va={va.game.nunique()}, te={te.game.nunique()}) n_te={len(te)} pos={te.y.mean():.4f}")

EV = ["half", "clock", "n_ev_60", "n_ev_15", "last_shown"] + ["since_" + c for c in CLASSES]
TIM = ["gap", "prev_dur"]
ORAC = ["prev_y"]

def best_thr(y, p):
    thrs = np.quantile(p, np.linspace(0.05, 0.95, 91))
    return max(thrs, key=lambda th: f1_score(y, p >= th))

def boot_ci(df_eval, p_col, thr=None, is_bernoulli=False, n_boot=200, seed=42):
    b_rng = np.random.default_rng(seed)
    games = df_eval.game.unique()
    grouped = {g: df_eval[df_eval.game == g] for g in games}
    f1s, aucs, aps = [], [], []
    for _ in range(n_boot):
        sg = b_rng.choice(games, len(games), replace=True)
        sub = pd.concat([grouped[g] for g in sg], ignore_index=True)
        yb = sub.y.to_numpy()
        pb = sub[p_col].to_numpy()
        if is_bernoulli:
            yhat = b_rng.random(len(pb)) < pb
            f1s.append(f1_score(yb, yhat))
        else:
            f1s.append(f1_score(yb, pb >= thr))
        aucs.append(roc_auc_score(yb, pb))
        aps.append(average_precision_score(yb, pb))
    return (
        f"[{np.percentile(f1s, 2.5):.3f},{np.percentile(f1s, 97.5):.3f}]",
        f"[{np.percentile(aucs, 2.5):.3f},{np.percentile(aucs, 97.5):.3f}]",
        f"[{np.percentile(aps, 2.5):.3f},{np.percentile(aps, 97.5):.3f}]",
    )

res = []
te["p_const1"] = 1.0
f1_ci, auc_ci, ap_ci = boot_ci(te, "p_const1", thr=0.5)
res.append({
    "model": "1a. Constant Always-Background (all 1)",
    "f1": round(f1_score(te.y, np.ones(len(te))), 4),
    "auc": 0.5000,
    "ap": round(float(te.y.mean()), 4),
    "f1_ci": f1_ci,
    "auc_ci": "[0.500,0.500]",
    "ap_ci": ap_ci,
})

te["p_persist"] = np.where(te.prev_y >= 0, te.prev_y, tr.y.mean())
f1_ci, auc_ci, ap_ci = boot_ci(te, "p_persist", thr=0.5)
res.append({
    "model": "1b. Persistence Baseline (same as prev_y, ORACLE)",
    "f1": round(f1_score(te.y, te.p_persist >= 0.5), 4),
    "auc": round(roc_auc_score(te.y, te.p_persist), 4),
    "ap": round(average_precision_score(te.y, te.p_persist), 4),
    "f1_ci": f1_ci,
    "auc_ci": auc_ci,
    "ap_ci": ap_ci,
})

for col_m, label_m in [
    ("mori_p_causal", "2a. Mori Rule Causal [0,+15s] (p=0.18, 20 seeds)"),
    ("mori_p_twosided", "2b. Mori Rule Two-Sided [-15s,+15s] (p=0.18, non-causal)"),
]:
    s_rng = np.random.default_rng(0)
    mf1 = np.mean([f1_score(te.y, s_rng.random(len(te)) < te[col_m].to_numpy()) for _ in range(20)])
    f1_ci, auc_ci, ap_ci = boot_ci(te, col_m, is_bernoulli=True)
    res.append({
        "model": label_m,
        "f1": round(float(mf1), 4),
        "auc": round(roc_auc_score(te.y, te[col_m]), 4),
        "ap": round(average_precision_score(te.y, te[col_m]), 4),
        "f1_ci": f1_ci,
        "auc_ci": auc_ci,
        "ap_ci": ap_ci,
    })

for name, cols in [
    ("3. HGB: Events + Clock (22 features)", EV),
    ("4. HGB: Events + Clock + Utterance Timing (24 features)", EV + TIM),
    ("5. HGB: Events + Timing + Oracle prev_y (25 features)", EV + TIM + ORAC),
]:
    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, random_state=0)
    clf.fit(tr[cols], tr.y)
    pv = clf.predict_proba(va[cols])[:, 1]
    th = best_thr(va.y.to_numpy(), pv)
    pcol = f"p_{name[:2]}"
    te[pcol] = clf.predict_proba(te[cols])[:, 1]
    f1_ci, auc_ci, ap_ci = boot_ci(te, pcol, thr=th)
    res.append({
        "model": name,
        "f1": round(f1_score(te.y, te[pcol] >= th), 4),
        "auc": round(roc_auc_score(te.y, te[pcol]), 4),
        "ap": round(average_precision_score(te.y, te[pcol]), 4),
        "f1_ci": f1_ci,
        "auc_ci": auc_ci,
        "ap_ci": ap_ci,
    })

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
Xtr = vec.fit_transform(tr.text)
Xva = vec.transform(va.text)
Xte = vec.transform(te.text)
lr = LogisticRegression(max_iter=500, C=2.0).fit(Xtr, tr.y)
pv = lr.predict_proba(Xva)[:, 1]
th = best_thr(va.y.to_numpy(), pv)
te["p_txt"] = lr.predict_proba(Xte)[:, 1]
f1_ci, auc_ci, ap_ci = boot_ci(te, "p_txt", thr=th)
res.append({
    "model": "6. TF-IDF + LR Text Reference (sees spoken words)",
    "f1": round(f1_score(te.y, te.p_txt >= th), 4),
    "auc": round(roc_auc_score(te.y, te.p_txt), 4),
    "ap": round(average_precision_score(te.y, te.p_txt), 4),
    "f1_ci": f1_ci,
    "auc_ci": auc_ci,
    "ap_ci": ap_ci,
})

out = pd.DataFrame(res)
print(out.to_string(index=False))
out.to_csv(f"{OUT}/table_binary_bg_benchmark_{LANG}.csv", index=False)
