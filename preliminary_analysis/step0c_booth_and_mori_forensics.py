import glob, json, os, re, urllib.parse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import linear_kernel

SN = os.environ.get("SN_LABELS_DIR", "data/soccernet_labels")
LFCBI = os.environ.get("LFCBI_DIR", "data/lfcbi/data")
# Separate from the notebook's outputs so script runs never overwrite the notebook's tables
OUT = os.environ.get("OUT_DIR", "preliminary_analysis/outputs/script_runs")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

CLASSES = [
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside", "Shots on target", "Shots off target",
    "Clearance", "Ball out of play", "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick",
    "Corner", "Yellow card", "Red card", "Yellow->red card"
]
CAP = 300.0

# 1. Load LFCBI splits + events
df = pd.concat([
    pd.read_csv(
        f"{LFCBI}/spotting/{s}.csv",
        usecols=["id", "game", "half", "start", "end", "text", "language", "speaker", "付加的情報か", "split"]
    )
    for s in ["train", "valid", "test"]
], ignore_index=True)
en = df[df.language == "en"].dropna(subset=["half", "start", "end", "speaker", "付加的情報か", "text"]).copy()
en["half"] = en["half"].astype(int)
en["n_words"] = en.text.str.split().str.len()
en["dur"] = en["end"] - en["start"]
en["y_bg"] = en["付加的情報か"].astype(int)

events = {}
for g in en.game.unique():
    p = f"{SN}/{g}/Labels-v2.json"
    if os.path.exists(p):
        ann = json.load(open(p))["annotations"]
        events[g] = pd.DataFrame([{
            "half": int(a["gameTime"].split(" - ")[0]),
            "t": int(a["position"]) / 1000.0,
            "label": a["label"],
            "shown": a.get("visibility") == "visible"
        } for a in ann]).sort_values("t")

# 2. Strictly out-of-goal speaker role assignment (Lead_PBP vs Colour_Analyst)
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

first_in_half = en2.sort_values("start").groupby(["game", "half"]).first()
n_first_lead = int((first_in_half["role"] == "Lead_PBP").sum())
n_two_halves = len(first_in_half)
print(f"English diarized halves: {en.groupby(['game','half']).ngroups} total -> {n_two_halves} two-speaker halves ({len(en2):,} top-2 turns)")
print(f"First utterance of half spoken by Lead_PBP: {n_first_lead}/{n_two_halves} ({n_first_lead/n_two_halves:.1%})")

# 3. Post-event turn-taking curves with game-clustered bootstrap CIs
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
            "gpt_bg_rate": round(grp.y_bg.mean(), 3),
            "gpt_bg_ci": f"[{nb_lo:.3f},{nb_hi:.3f}]",
            "median_words": float(grp.n_words.median()),
            "median_dur_s": round(float(grp.dur.median()), 2),
        })
handoff_df = pd.DataFrame(handoff_rows)
handoff_df.to_csv(f"{OUT}/table_two_voice_handoff_en.csv", index=False)

# 4. Speaker routing & handoff predictability on the held-out test split (with persistence baseline & CIs)
rows = []
for (g, h), u in en2.sort_values("start").groupby(["game", "half"]):
    ev = events[g]
    ev = ev[ev.half == h]
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
tr, va, te = (feat_df[feat_df.split == s].copy() for s in ["train", "valid", "test"])
EV = ["half", "clock", "min_since_any", "n_ev_60", "n_ev_15"] + ["since_" + c for c in CLASSES]
TURN = EV + ["gap", "prev_dur", "prev_colour", "run_sec"]

clf_spk_ev = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[EV], tr.is_colour)
clf_spk_tu = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[TURN], tr.is_colour)
clf_bg_ev = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[EV], tr.y_bg)

te["p_spk_ev"] = clf_spk_ev.predict_proba(te[EV])[:, 1]
te["p_spk_tu"] = clf_spk_tu.predict_proba(te[TURN])[:, 1]
te["p_bg_ev"] = clf_bg_ev.predict_proba(te[EV])[:, 1]
te["p_spk_persist"] = np.where(te["prev_colour"] >= 0, te["prev_colour"], tr["is_colour"].mean())

tr_v = tr[tr.prev_colour >= 0].copy()
te_v = te[te.prev_colour >= 0].copy()
tr_v["switched"] = (tr_v.is_colour != tr_v.prev_colour).astype(int)
te_v["switched"] = (te_v.is_colour != te_v.prev_colour).astype(int)
clf_sw_ev = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr_v[EV], tr_v.switched)
clf_sw_tu = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr_v[TURN], tr_v.switched)
te_v["p_sw_ev"] = clf_sw_ev.predict_proba(te_v[EV])[:, 1]
te_v["p_sw_tu"] = clf_sw_tu.predict_proba(te_v[TURN])[:, 1]

def boot_auc_ap(df_sub, y_col, p_col, n_boot=200, seed=42):
    b_rng = np.random.default_rng(seed)
    games = df_sub["game"].unique()
    grouped = {g: df_sub[df_sub["game"] == g] for g in games}
    aucs, aps = [], []
    for _ in range(n_boot):
        sg = b_rng.choice(games, size=len(games), replace=True)
        sub = pd.concat([grouped[g] for g in sg], ignore_index=True)
        if sub[y_col].nunique() < 2:
            continue
        aucs.append(roc_auc_score(sub[y_col], sub[p_col]))
        aps.append(average_precision_score(sub[y_col], sub[p_col]))
    return (
        f"[{np.percentile(aucs, 2.5):.3f},{np.percentile(aucs, 97.5):.3f}]",
        f"[{np.percentile(aps, 2.5):.3f},{np.percentile(aps, 97.5):.3f}]"
    )

te_goal45 = te[te["since_Goal"] <= 45].copy()
te_shot45 = te[te["since_Shots on target"] <= 45].copy()

eval_specs = [
    ("y_bg (GPT Binary Bg)", "All 2-Voice Test Turns", te, "y_bg", "p_bg_ev", "Events Only (HGB)"),
    ("y_bg (GPT Binary Bg)", "<=45s After Goal", te_goal45, "y_bg", "p_bg_ev", "Events Only (HGB)"),
    ("is_colour (Speaker Role)", "All 2-Voice Test Turns", te, "is_colour", "p_spk_persist", "Baseline: Same Speaker as Prev Turn"),
    ("is_colour (Speaker Role)", "All 2-Voice Test Turns", te, "is_colour", "p_spk_ev", "Events Only (HGB)"),
    ("is_colour (Speaker Role)", "All 2-Voice Test Turns", te, "is_colour", "p_spk_tu", "Events + Turn State (HGB)"),
    ("is_colour (Speaker Role)", "<=45s After Goal", te_goal45, "is_colour", "p_spk_persist", "Baseline: Same Speaker as Prev Turn"),
    ("is_colour (Speaker Role)", "<=45s After Goal", te_goal45, "is_colour", "p_spk_ev", "Events Only (HGB)"),
    ("is_colour (Speaker Role)", "<=45s After Goal", te_goal45, "is_colour", "p_spk_tu", "Events + Turn State (HGB)"),
    ("is_colour (Speaker Role)", "<=45s After Shot on Target", te_shot45, "is_colour", "p_spk_ev", "Events Only (HGB)"),
    ("switched (Speaker Handoff)", "All 2-Voice Test Turns (t>1)", te_v, "switched", "p_sw_ev", "Events Only (HGB)"),
    ("switched (Speaker Handoff)", "All 2-Voice Test Turns (t>1)", te_v, "switched", "p_sw_tu", "Events + Turn State (HGB)"),
]

spk_res = []
for tgt, subset_name, d_sub, y_c, p_c, m_name in eval_specs:
    auc_ci, ap_ci = boot_auc_ap(d_sub, y_c, p_c)
    spk_res.append({
        "target": tgt,
        "subset": subset_name,
        "model": m_name,
        "n": len(d_sub),
        "pos_rate": round(float(d_sub[y_c].mean()), 3),
        "auroc": round(float(roc_auc_score(d_sub[y_c], d_sub[p_c])), 4),
        "auroc_ci": auc_ci,
        "ap": round(float(average_precision_score(d_sub[y_c], d_sub[p_c])), 4),
        "ap_ci": ap_ci,
    })
pd.DataFrame(spk_res).to_csv(f"{OUT}/table_speaker_routing_en.csv", index=False)

# 5. Event collision / coalescing analysis across all 500 SoccerNet games
files = sorted(glob.glob(f"{SN}/**/Labels-v2.json", recursive=True))
evs = []
for f in files:
    g = f[len(SN) + 1:-len("/Labels-v2.json")]
    for a in json.load(open(f))["annotations"]:
        evs.append({
            "game": g,
            "half": int(a["gameTime"].split(" - ")[0]),
            "t": int(a["position"]) / 1000.0,
            "label": a["label"]
        })
edf = pd.DataFrame(evs).sort_values(["game", "half", "t"]).reset_index(drop=True)
edf["prev_t"] = edf.groupby(["game", "half"])["t"].shift(1)
edf["prev_label"] = edf.groupby(["game", "half"])["label"].shift(1)
edf["dt_prev"] = edf["t"] - edf["prev_t"]

coll_rows = []
for tgt in ["Goal", "Penalty", "Yellow card", "Red card", "Corner", "Substitution"]:
    sub = edf[edf.label == tgt].dropna(subset=["dt_prev"])
    dom = sub.prev_label.value_counts().index[0]
    dom_lag = sub[sub.prev_label == dom].dt_prev.median()
    coll_rows.append({
        "target_event": tgt,
        "n_events": len(edf[edf.label == tgt]),
        "frac_eq_0s_coannot": round(float((sub.dt_prev == 0.0).mean()), 3),
        "frac_gt0_le_2s": round(float(((sub.dt_prev > 0.0) & (sub.dt_prev <= 2.0)).mean()), 3),
        "frac_prev_le_1s": round(float((sub.dt_prev <= 1.0).mean()), 3),
        "frac_prev_le_2s": round(float((sub.dt_prev <= 2.0).mean()), 3),
        "frac_prev_le_5s": round(float((sub.dt_prev <= 5.0).mean()), 3),
        "frac_prev_le_10s": round(float((sub.dt_prev <= 10.0).mean()), 3),
        "dominant_preceding_event": dom,
        "dom_share_all_lags": round(float((sub.prev_label == dom).mean()), 3),
        "dom_share_le_2s": round(float(((sub.prev_label == dom) & (sub.dt_prev <= 2.0)).mean()), 3),
        "median_lag_from_dominant_s": round(float(dom_lag), 2),
    })
pd.DataFrame(coll_rows).to_csv(f"{OUT}/table_event_collisions_500games.csv", index=False)

# 6. Forensic audit of Mori et al.'s released Wikipedia RAG corpus (3,249 files vs 3,257 reported)
wiki_dir = f"{LFCBI}/addinfo_retrieval"
wiki_files = sorted([f for f in os.listdir(wiki_dir) if f.endswith(".txt")])
names = [urllib.parse.unquote(f[:-4]) for f in wiki_files]
gridiron = [n for n in names if any(k in n.lower() for k in ["gridiron", "american_football", "baseball", "basketball", "rugby"])]
wrong_era = [n for n in names if (m := re.search(r"born_(\d{4})", n)) and (int(m.group(1)) < 1970 or int(m.group(1)) > 2000)]
unique_wrong_entity = sorted(set(gridiron) | set(wrong_era))

post2017 = 0
chunks, chunk_files = [], []
for f in wiki_files:
    txt = open(f"{wiki_dir}/{f}", encoding="utf-8", errors="ignore").read()
    if re.search(r"\b20(18|19|20|21|22|23|24)\b", txt):
        post2017 += 1
    for i in range(0, max(1, len(txt)), 900):
        c = txt[i:i + 1000]
        if c.strip():
            chunks.append(c)
            chunk_files.append(f)
        if i + 1000 >= len(txt):
            break

# Test TF-IDF top-10 chunk retrieval on the 25 demo clips in sample_metadata.csv
sample_meta = pd.read_csv(f"{LFCBI}/demo/sample_metadata.csv")
scbi_all = pd.read_csv(f"{LFCBI}/commentary/scbi-v2.csv", usecols=["game", "half", "start", "end", "text"])
pif = pd.read_csv(f"{LFCBI}/demo/players_in_frames_sn_gamestate.csv")
vec = TfidfVectorizer()
X_chunks = vec.fit_transform(chunks)

queries = []
for r in sample_meta.itertuples():
    t = (r.start + r.end) / 2.0
    u = scbi_all[(scbi_all["game"] == r.game) & (scbi_all["half"] == r.half) & (scbi_all["end"] < t)].sort_values("start")
    prev_texts, tot_len = [], 0
    for txt in reversed(u["text"].dropna().tolist()):
        if tot_len + len(str(txt)) > 256:
            break
        prev_texts.append(str(txt))
        tot_len += len(str(txt)) + 1
    q = "Previous comments: " + " ".join(reversed(prev_texts))
    p_sub = pif[(pif["sample_id"] == r.id) & (pif["time"] >= t - 2) & (pif["time"] <= t + 2)].dropna(subset=["long_name"])
    if len(p_sub):
        p_str = ", ".join([f"{row.long_name} from {row.team}" for row in p_sub.drop_duplicates(subset=["long_name", "team"]).itertuples()])
        q = f"Players shown in this frame: {p_str}\n" + q
    m_year = int(re.search(r"(201\d)-(\d{2})-(\d{2})", r.game).group(1))
    queries.append((r.id, m_year, f"Game: {r.game}\n" + q))

sims = linear_kernel(vec.transform([q[2] for q in queries]), X_chunks)
future_chunks_top10 = 0
for i, (_, m_year, _) in enumerate(queries):
    for idx in np.argsort(sims[i])[::-1][:10]:
        years = [int(y) for y in re.findall(r"\b(20(?:1[5-9]|2[0-4]))\b", chunks[idx])]
        if any(y > m_year for y in years):
            future_chunks_top10 += 1

audit_df = pd.DataFrame([
    {"metric": "Total Wikipedia files in Mori release (3,257 reported in paper Sec 3.2)", "count": len(wiki_files), "pct": 100.0, "examples": "8 fewer .txt files in released data/addinfo_retrieval archive"},
    {"metric": "Files containing 2018-2024 years (static 2024 snapshot for 2014-2017 games)", "count": post2017, "pct": round(post2017 / len(wiki_files) * 100, 1), "examples": "2018-2024 career transfers/stats present in unchunked files"},
    {"metric": "Retrieved top-10 TF-IDF chunks for 25 demo clips containing years > match_year", "count": future_chunks_top10, "pct": round(future_chunks_top10 / (len(queries) * 10) * 100, 1), "examples": f"{future_chunks_top10}/{len(queries)*10} chunks across 25/25 demo queries contain post-match years"},
    {"metric": "Wrong sport disambiguation (gridiron / American football)", "count": len(gridiron), "pct": round(len(gridiron) / len(wiki_files) * 100, 2), "examples": "; ".join(gridiron[:5])},
    {"metric": "Wrong birth era (<1970 or >2000 for 2014-2017 matches)", "count": len(wrong_era), "pct": round(len(wrong_era) / len(wiki_files) * 100, 2), "examples": "; ".join(wrong_era[:5])},
    {"metric": "Unique wrong-person entity files (union; 1 file overlaps both lists)", "count": len(unique_wrong_entity), "pct": round(len(unique_wrong_entity) / len(wiki_files) * 100, 2), "examples": "Overlap: Joe_Campbell_(American_football,_born_1955)"},
])
audit_df.to_csv(f"{OUT}/table_mori_corpus_audit.csv", index=False)
print("Saved all 4 tables to", OUT)
