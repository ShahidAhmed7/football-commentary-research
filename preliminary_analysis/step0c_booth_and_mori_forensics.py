import glob, json, os, re, urllib.parse
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

SN = os.environ.get("SN_LABELS_DIR", "data/soccernet_labels")
LFCBI = os.environ.get("LFCBI_DIR", "data/lfcbi/data")
OUT = os.environ.get("OUT_DIR", "preliminary_analysis/outputs")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

CLASSES = ["Penalty", "Kick-off", "Goal", "Substitution", "Offside", "Shots on target", "Shots off target",
           "Clearance", "Ball out of play", "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick",
           "Corner", "Yellow card", "Red card", "Yellow->red card"]
CAP = 300.0

# 1. Load LFCBI splits + events
df = pd.concat([pd.read_csv(f"{LFCBI}/spotting/{s}.csv",
                            usecols=["id", "game", "half", "start", "end", "text", "language", "speaker", "付加的情報か", "split"])
                for s in ["train", "valid", "test"]], ignore_index=True)
en = df[df.language == "en"].dropna(subset=["half", "start", "end", "speaker", "付加的情報か", "text"]).copy()
en["half"] = en.half.astype(int)
en["n_words"] = en.text.str.split().str.len()
en["dur"] = en["end"] - en["start"]
en["y_bg"] = en["付加的情報か"].astype(int)

events = {}
for g in en.game.unique():
    p = f"{SN}/{g}/Labels-v2.json"
    if os.path.exists(p):
        ann = json.load(open(p))["annotations"]
        events[g] = pd.DataFrame([{"half": int(a["gameTime"].split(" - ")[0]),
                                   "t": int(a["position"]) / 1000.0,
                                   "label": a["label"],
                                   "shown": a.get("visibility") == "visible"} for a in ann]).sort_values("t")

# 2. Strictly out-of-goal speaker role assignment (Lead_PBP = "Marcus", Colour_Analyst = "Gary")
non_goal = []
for (g, h), u in en.groupby(["game", "half"]):
    if g not in events:
        continue
    gt = events[g].query('half == @h and label == "Goal"').t.to_numpy()
    mask = np.ones(len(u), dtype=bool)
    for t0 in gt:
        mask &= ~((u.start.to_numpy() >= t0 - 60) & (u.start.to_numpy() <= t0 + 90))
    non_goal.append(u[mask])
ng = pd.concat(non_goal, ignore_index=True)

spk = ng.groupby(["game", "half", "speaker"]).agg(
    n_utt=("id", "size"),
    n_short=("n_words", lambda s: (s <= 4).sum())
).reset_index()
spk["rk_utt"] = spk.groupby(["game", "half"])["n_utt"].rank(ascending=False, method="first").astype(int)
top2 = spk[spk.rk_utt.isin([1, 2])].copy()
valid_halves = set(top2[top2.rk_utt == 2].query("n_utt >= 20").set_index(["game", "half"]).index)
top2 = top2[top2.set_index(["game", "half"]).index.isin(valid_halves)].copy()
top2["role"] = top2.groupby(["game", "half"])["n_short"].rank(ascending=False, method="first").map(
    {1.0: "Lead_PBP", 2.0: "Colour_Analyst"}
)
en2 = en.merge(top2[["game", "half", "speaker", "role"]], on=["game", "half", "speaker"])
en2["is_colour"] = (en2.role == "Colour_Analyst").astype(int)

# 3. Three-phase event-handoff curves with game-clustered bootstrap CIs
bins = [-60, -30, -15, -5, 0, 5, 15, 30, 45, 60]
handoff_rows = []
for ev_type in ["Goal", "Foul", "Yellow card", "Shots on target", "Corner", "Substitution", "Random_Control"]:
    w_list = []
    for (g, h), u in en2.groupby(["game", "half"]):
        if g not in events:
            continue
        if ev_type == "Random_Control":
            ts = rng.uniform(120, 2500, size=2)
        else:
            ts = events[g].query("half == @h and label == @ev_type").t.to_numpy()
        for t0 in ts:
            sub = u[(u.start >= t0 - 60) & (u.start <= t0 + 60)].copy()
            sub["dt"] = sub.start - t0
            sub["win"] = pd.cut(sub.dt, bins=bins)
            w_list.append(sub[["game", "win", "is_colour", "y_bg", "n_words", "dur"]])
    w = pd.concat(w_list, ignore_index=True).dropna(subset=["win"])
    for win, grp in w.groupby("win", observed=True):
        ug = grp.game.unique()
        idx = {g: grp.index[grp.game == g].to_numpy() for g in ug}
        col_vals = grp.is_colour.to_numpy()
        bg_vals = grp.y_bg.to_numpy()
        base = grp.index.to_numpy()
        pos = {k: np.searchsorted(base, v) for k, v in idx.items()}
        b_col, b_bg = [], []
        for _ in range(500):
            s = np.concatenate([pos[g] for g in rng.choice(ug, len(ug))])
            b_col.append(col_vals[s].mean())
            b_bg.append(bg_vals[s].mean())
        c_lo, c_hi = np.percentile(b_col, [2.5, 97.5])
        nb_lo, nb_hi = np.percentile(b_bg, [2.5, 97.5])
        handoff_rows.append({
            "event": ev_type, "window_s": str(win), "n_utt": len(grp),
            "colour_share": round(grp.is_colour.mean(), 3),
            "colour_ci": f"[{c_lo:.3f},{c_hi:.3f}]",
            "notebook_bg_rate": round(grp.y_bg.mean(), 3),
            "notebook_ci": f"[{nb_lo:.3f},{nb_hi:.3f}]",
            "median_words": float(grp.n_words.median()),
            "median_dur_s": round(float(grp.dur.median()), 2),
        })
handoff_df = pd.DataFrame(handoff_rows)
handoff_df.to_csv(f"{OUT}/table_two_voice_handoff_en.csv", index=False)

# 4. Speaker routing & handoff predictability on the held-out test split
rows = []
for (g, h), u in en2.sort_values("start").groupby(["game", "half"]):
    ev = events[g]; ev = ev[ev.half == h]
    et, el = ev.t.to_numpy(), ev.label.to_numpy()
    prev_end, prev_dur, prev_col, run_sec = np.nan, np.nan, np.nan, 0.0
    for r in u.itertuples():
        t = r.start
        m = et <= t
        f = {"half": h, "clock": t / 60.0}
        for c in CLASSES:
            tt = et[m & (el == c)]
            f["since_" + c] = min(t - tt[-1], CAP) if len(tt) else CAP
        f["min_since_any"] = min((t - et[m][-1]), CAP) if m.any() else CAP
        f["n_ev_60"] = int(((et > t - 60) & m).sum())
        f["n_ev_15"] = int(((et > t - 15) & m).sum())
        f["gap"] = t - prev_end if prev_end == prev_end else -1.0
        f["prev_dur"] = prev_dur if prev_dur == prev_dur else -1.0
        f["prev_colour"] = prev_col if prev_col == prev_col else -1.0
        f["run_sec"] = run_sec
        f.update(game=g, split=r.split, is_colour=r.is_colour, y_bg=r.y_bg)
        rows.append(f)
        if prev_col == r.is_colour and (t - prev_end) < 10.0:
            run_sec += r.dur
        else:
            run_sec = r.dur
        prev_end, prev_dur, prev_col = r.end, r.dur, r.is_colour

feat_df = pd.DataFrame(rows)
tr, va, te = (feat_df[feat_df.split == s] for s in ["train", "valid", "test"])
EV = ["half", "clock", "min_since_any", "n_ev_60", "n_ev_15"] + ["since_" + c for c in CLASSES]
TURN = EV + ["gap", "prev_dur", "prev_colour", "run_sec"]

clf_ev = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[EV], tr.is_colour)
pt_ev = clf_ev.predict_proba(te[EV])[:, 1]
clf_tu = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[TURN], tr.is_colour)
pt_tu = clf_tu.predict_proba(te[TURN])[:, 1]

tr_v = tr[tr.prev_colour >= 0].copy(); te_v = te[te.prev_colour >= 0].copy()
tr_v["switched"] = (tr_v.is_colour != tr_v.prev_colour).astype(int)
te_v["switched"] = (te_v.is_colour != te_v.prev_colour).astype(int)
clf_sw = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr_v[TURN], tr_v.switched)
psw = clf_sw.predict_proba(te_v[TURN])[:, 1]

spk_res = [
    {"task": "Who speaks next (Events only, all test)", "n": len(te), "pos_rate": round(te.is_colour.mean(), 3),
     "auroc": round(roc_auc_score(te.is_colour, pt_ev), 4), "ap": round(average_precision_score(te.is_colour, pt_ev), 4)},
    {"task": "Who speaks next (Events only, <=45s after Goal)", "n": int((te["since_Goal"] <= 45).sum()),
     "pos_rate": round(te.is_colour[te["since_Goal"] <= 45].mean(), 3),
     "auroc": round(roc_auc_score(te.is_colour[te["since_Goal"] <= 45], pt_ev[te["since_Goal"] <= 45]), 4),
     "ap": round(average_precision_score(te.is_colour[te["since_Goal"] <= 45], pt_ev[te["since_Goal"] <= 45]), 4)},
    {"task": "Who speaks next (Events only, <=45s after Shot on target)", "n": int((te["since_Shots on target"] <= 45).sum()),
     "pos_rate": round(te.is_colour[te["since_Shots on target"] <= 45].mean(), 3),
     "auroc": round(roc_auc_score(te.is_colour[te["since_Shots on target"] <= 45], pt_ev[te["since_Shots on target"] <= 45]), 4),
     "ap": round(average_precision_score(te.is_colour[te["since_Shots on target"] <= 45], pt_ev[te["since_Shots on target"] <= 45]), 4)},
    {"task": "Who speaks next (Events + Turn state, all test)", "n": len(te), "pos_rate": round(te.is_colour.mean(), 3),
     "auroc": round(roc_auc_score(te.is_colour, pt_tu), 4), "ap": round(average_precision_score(te.is_colour, pt_tu), 4)},
    {"task": "Speaker handoff on next clause (Events + Turn state)", "n": len(te_v), "pos_rate": round(te_v.switched.mean(), 3),
     "auroc": round(roc_auc_score(te_v.switched, psw), 4), "ap": round(average_precision_score(te_v.switched, psw), 4)},
]
pd.DataFrame(spk_res).to_csv(f"{OUT}/table_speaker_routing_en.csv", index=False)

# 5. Event collision / coalescing analysis across all 500 SoccerNet games
files = glob.glob(f"{SN}/**/Labels-v2.json", recursive=True)
evs = []
for f in files:
    g = f[len(SN) + 1:-len("/Labels-v2.json")]
    for a in json.load(open(f))["annotations"]:
        evs.append({"game": g, "half": int(a["gameTime"].split(" - ")[0]),
                    "t": int(a["position"]) / 1000.0, "label": a["label"]})
edf = pd.DataFrame(evs).sort_values(["game", "half", "t"]).reset_index(drop=True)
edf["prev_t"] = edf.groupby(["game", "half"])["t"].shift(1)
edf["prev_label"] = edf.groupby(["game", "half"])["label"].shift(1)
edf["dt_prev"] = edf["t"] - edf["prev_t"]

col_rows = []
for target in ["Goal", "Yellow card", "Red card", "Corner", "Penalty"]:
    sub = edf[edf.label == target]
    top_prev = sub[sub.dt_prev <= 15].prev_label.value_counts()
    top_name = top_prev.index[0] if len(top_prev) else "None"
    med_dt = sub[sub.prev_label == top_name].dt_prev.median() if len(top_prev) else np.nan
    col_rows.append({
        "target_event": target, "n_events": len(sub),
        "frac_prev_le_1s": round(float((sub.dt_prev <= 1.0).mean()), 3),
        "frac_prev_le_2s": round(float((sub.dt_prev <= 2.0).mean()), 3),
        "frac_prev_le_5s": round(float((sub.dt_prev <= 5.0).mean()), 3),
        "frac_prev_le_10s": round(float((sub.dt_prev <= 10.0).mean()), 3),
        "dominant_preceding_event": top_name,
        "median_lag_from_dominant_s": round(float(med_dt), 2),
    })
pd.DataFrame(col_rows).to_csv(f"{OUT}/table_event_collisions_500games.csv", index=False)

# 6. Forensic audit of Mori et al.'s 3,249 Wikipedia files
wiki_dir = f"{LFCBI}/addinfo_retrieval"
if os.path.exists(wiki_dir):
    wfiles = [f for f in sorted(os.listdir(wiki_dir)) if f.endswith(".txt")]
    names = [urllib.parse.unquote(f[:-4]) for f in wfiles]
    wrong_sport = [n for n in names if any(k in n.lower() for k in ["gridiron", "american_football", "baseball", "basketball", "rugby"])]
    wrong_era = [n for n in names if (m := re.search(r"born_(\d{4})", n)) and (int(m.group(1)) < 1970 or int(m.group(1)) > 2000)]
    post_2017 = sum(bool(re.search(r"\b20(18|19|20|21|22|23|24)\b", open(f"{wiki_dir}/{f}", errors="ignore").read())) for f in wfiles)
    pd.DataFrame([
        {"metric": "Total Wikipedia files in Mori release", "count": len(wfiles), "pct": 100.0, "examples": ""},
        {"metric": "Files containing post-2017 future data (temporal leakage for 2014-2017 matches)", "count": post_2017,
         "pct": round(100.0 * post_2017 / len(wfiles), 1), "examples": "2018-2024 career transfers/stats & match results"},
        {"metric": "Wrong sport disambiguation (gridiron / American football)", "count": len(wrong_sport),
         "pct": round(100.0 * len(wrong_sport) / len(wfiles), 2), "examples": "; ".join(wrong_sport[:6])},
        {"metric": "Wrong birth era (<1970 or >2000 for 2014-2017 players)", "count": len(wrong_era),
         "pct": round(100.0 * len(wrong_era) / len(wfiles), 2), "examples": "; ".join(wrong_era[:6])},
    ]).to_csv(f"{OUT}/table_mori_corpus_audit.csv", index=False)

print("Saved all 4 forensic & booth tables to", OUT)
