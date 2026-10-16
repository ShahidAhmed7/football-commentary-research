# Data Sources

This file records where every raw input used by
`preliminary_analysis/preliminary_analysis_walkthrough.ipynb` (and the two
`preliminary_analysis/step0*.py` scripts) came from.

Raw data is **not** part of this repository. Everything lives under `data/`,
which is listed in `.gitignore`. Do not commit anything from `data/`.

## Layout

```text
data/
├── lfcbi/
│   ├── data/                 # Mori et al. system-data.zip, extracted (Section 1)
│   ├── mori_code/            # Mori et al. source files (Section 2)
│   └── soccer-bg-script/     # Mori et al. annotation repo files (Section 3)
├── soccernet_labels/         # <league>/<season>/<game>/Labels-v2.json, 500 games (Section 4)
├── pff/
│   └── 10517.jsonl           # 200-frame PFF FC tracking excerpt (Section 5)
└── skillcorner/
    ├── possessions_on_ball.csv   # upstream name: 1925299_phases_of_play.csv (Section 6)
    └── dynamic_events.csv        # upstream name: 1925299_dynamic_events.csv (Section 6)
```

With files in these places, the notebook's path cell prints `found` for all
seven inputs and needs no environment variables. The variables `LFCBI_DIR`,
`MORI_CODE_DIR`, `SBS_DIR`, `SN_LABELS_DIR`, `PFF_SAMPLE`, `SKC_POP` and
`SKC_DE` override the defaults.

## How the records below were checked

- **Date obtained.** Every file was downloaded on **2026-10-15** (local machine
  time). The date comes from the modification times of the downloaded files,
  not from a separate download log.
- **Version pinning.** Each local file was matched against the upstream source
  by content, not by assumption: git blob hashes against the GitHub or Hugging
  Face tree, or CRC-32 against the source zip.
- **Directory digests.** These are computed with the recipe below. They let you
  confirm a fresh download is byte-identical to the copy the notebook was run on.

  ```bash
  cd data/<dir>
  find . -type f ! -name '.DS_Store' -print0 | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 | shasum -a 256
  ```

- **Licenses.** Only license files or metadata that were opened directly are
  quoted. Anything else is marked **[UNVERIFIED]**.

---

## 1. LFCBI / SCBI commentary corpus and system data (`data/lfcbi/data/`)

| Field | Value |
| :--- | :--- |
| Provider | Mori et al., ACL 2025 System Demonstrations, pp. 394–404 |
| Source | Google Drive folder linked as "Code repository" in footnote 2 of the paper: `https://drive.google.com/drive/folders/1_EqBtLr9YCnRDlB4IS9p69PhnZffTmPx`. The file is in its `archives/` subfolder (`https://drive.google.com/drive/folders/1ZCM9u9L-M1eaVtytMF4MP2amoSxWaZu7`). |
| File downloaded | `system-data.zip`, Drive file ID `1o1fdlSVqh4Y2nG2vzmsBkL_tOOQMBWZM` (`https://drive.google.com/uc?id=1o1fdlSVqh4Y2nG2vzmsBkL_tOOQMBWZM&export=download`) |
| Version | Google Drive has no version tag. The pin is the archive checksum: `system-data.zip` SHA-256 `c3d5e2dc568b1e55ca8b26b08b3b14097d7548bb5de70b24d342641ab329d539`, 86,660,295 bytes, 3,706 zip entries (3,464 files). Newest entry is dated 2025-02-19. |
| Date obtained | 2026-10-15 |
| Extraction | `unzip system-data.zip` produces a `data/` folder, which is placed at `data/lfcbi/data/`. All 3,464 extracted files were checked against the zip's sizes and CRC-32 values, with 0 mismatches. |
| Contents used | `commentary/scbi-v2.csv`, `commentary/gpt-3.5-turbo-1106_500game_1_llm_annotation.csv` (kept for the pilot-to-GPT join in Step 2), `spotting/`, `addinfo_retrieval/` (3,249 Wikipedia `.txt` files), `demo/`, `from_video/`, `reference_documents/` |
| Key file checksums (SHA-256) | `scbi-v2.csv`: `380f2305579b566af0554259388e823b3b36eb0a2d4b26c78c13997cfa7f1819` (135,281,406 B)<br>`gpt-3.5-turbo-1106_500game_1_llm_annotation.csv`: `5e8facf75a0351e61cf640717238df4298f0a9e407b0672ef5706b10b4720548` (49,325,723 B) |
| Directory digest | 3,463 files (excluding `.DS_Store`): `cbadd0d9ffb070f23b2e3b5eeca76333139ca32f256613c4c6d06dde34731d86` |
| License / terms | **No license stated.** There is no LICENSE or README file in the zip, and no LICENSE file in the companion GitHub repo (Section 2). Treat it as all rights reserved by the authors unless they say otherwise. The transcripts were derived from SoccerNet broadcast audio, so SoccerNet's terms for the underlying videos probably apply too **[UNVERIFIED]**. Do not redistribute. |

## 2. Mori et al. source code (`data/lfcbi/mori_code/`)

| Field | Value |
| :--- | :--- |
| Repo | `https://github.com/zaemon1251-hesty/soccer-bg-commentary` (files from `src/soccer_bg_commentary/`, `scripts/`, and `README.md`, flattened into one folder) |
| Version | **Not a single commit.** 17 of the 18 files match commits `aed9ae73b28186ef0847ee3ee24e203f1f0d1445` (2025-03-29, "api server"), `0a5f639df3ef3d476825cd86b8e305b0950698b4` and `1b968dd02122ef3b58a3776889a148ee16556dd4` (both 2025-03-31) byte for byte. `demo.py` matches the blob from commit `12790357f5aabf67002bdf011bcd103d8d8218bf` (2025-03-26), which was replaced in `aed9ae73`. No commit on any branch contains all 18 files. `main` HEAD on the download date was `f1a8706ff3ba859883991dcb6d24500e7d684afe` (2025-07-24). The notebook only reads `addinfo_retrieval.py` (to check the chunk splitter) and `constants.py` (prompt text). Both of those match `aed9ae73`. |
| Date obtained | 2026-10-15 |
| Directory digest | 18 files: `ed26d7e6dfd007074bc4b8f3d7d69c304c0025a760d7176b6d28ca3cb54ee979` |
| License / terms | **No license.** The GitHub API reports no license, and there is no LICENSE file at HEAD or at `aed9ae73`. Without a license, default copyright applies. Use it for reading and reference only, and do not redistribute. |
| Clean re-fetch | `git clone https://github.com/zaemon1251-hesty/soccer-bg-commentary && git -C soccer-bg-commentary checkout aed9ae73b28186ef0847ee3ee24e203f1f0d1445`. This reproduces every file except `demo.py`. |

## 3. Mori et al. annotation scripts (`data/lfcbi/soccer-bg-script/`)

| Field | Value |
| :--- | :--- |
| Repo | `https://github.com/zaemon1251-hesty/soccer-bg-script`, default branch `dev` |
| Version | Commit `501a781c8afd8039924e5f7092a0d3e404dad374` (2025-06-13), which was HEAD on the download date. 20 of the 21 local files match that tree byte for byte. The 21st file, `paths.txt`, is a local list of the files fetched and is **not** from upstream. |
| Files used | `annotation_tool/annotation-description.txt`, `annotation-human.md`, `annotation_guideline.md`, `haruki_annotation-target.txt`, `sakou_42_1_target_prompt.txt`, `scripts/experiment-llm-annotation/exp0-1.sh`, `scripts/stable/build_scbi.sh`, plus the rest listed in `paths.txt` |
| Date obtained | 2026-10-15 |
| Directory digest | 21 files: `4ea4825b022aa524a71d942377036b798dc825c3bf7a1a40df213c17204c9fcc` |
| License / terms | **No license.** The GitHub API reports none, and the tree at `501a781c` has no LICENSE file. Default copyright applies. |

## 4. SoccerNet-v2 action-spotting labels (`data/soccernet_labels/`)

| Field | Value |
| :--- | :--- |
| Source | Hugging Face dataset `SoccerNet/SN-Labels`: `https://huggingface.co/datasets/SoccerNet/SN-Labels` |
| Version | Revision `8e01649fe968da9f541c91019b65b3028b2425fc` (still `main` on 2026-10-15, last modified 2026-09-07) |
| Files | Only `Labels-v2.json`, for all **500** games: 6 leagues (`england_epl`, `europe_uefa-champions-league`, `france_ligue-1`, `germany_bundesliga`, `italy_serie-a`, `spain_laliga`) × seasons `2014-2015`, `2015-2016`, `2016-2017`. All 500 match that revision's git blob IDs, and the revision has exactly 500 `Labels-v2.json` files. Videos and other label files (`Labels-caption.json`, `Labels-v3.json`, `Labels-cameras.json`) were not downloaded. |
| Download method | Downloaded with `huggingface_hub` into a local folder, restricted to `*/Labels-v2.json`. This is inferred from the `.cache/huggingface` metadata the download left behind, which records the revision above for every file. |
| Date obtained | 2026-10-15 |
| Directory digest | 500 files: `ee7c00dcb4d5a9398e4644da24d0002f3f9a6a21fbe7d5d618d50c78f01735e7` |
| License / terms | The dataset is **not gated** (`gated: false` in the Hugging Face API) and needs no account. **No license is declared**: there is no dataset card, and the only tag is `region:us`. Access to SoccerNet *videos* requires signing the SoccerNet NDA. Whether the NDA or other SoccerNet terms also cover these label files is **[UNVERIFIED]**. Do not redistribute. |

## 5. PFF FC tracking excerpt (`data/pff/10517.jsonl`)

| Field | Value |
| :--- | :--- |
| Content | 200 consecutive tracking frames from PFF FC game `10517` (FIFA World Cup 2022 Final, Argentina vs France). Tracking format version `4.1.0`, with per-player `visibility` (`VISIBLE`/`ESTIMATED`) fields. |
| Source | The kloppy test fixture `kloppy/tests/files/pff_10517.jsonl` in `https://github.com/PySport/kloppy`. The local file is byte-identical: git blob `9e61443659a0bec8adcf570608026e3f31d86483`, 1,003,941 bytes. |
| Version | Fixture added in kloppy commit `9692ec306b07fb1c6e63032515388ef78cf233b8` (2025-01-26, "Add support for loading PFF FC Tracking data (#409)"). It was unchanged at kloppy `master` HEAD `ad453c40116e254a7d475b857c46d4af164a3a54` on the download date. |
| Raw URL | `https://raw.githubusercontent.com/PySport/kloppy/9692ec306b07fb1c6e63032515388ef78cf233b8/kloppy/tests/files/pff_10517.jsonl` |
| SHA-256 | `8bf1ce4963da1d6c9a1644fff4b4b2cf421a8779ce3f6004861713902bfc63c4` |
| Date obtained | 2026-10-15 |
| License / terms | The kloppy repository is BSD-3-Clause (GitHub API). The tracking data itself comes from PFF FC's public World Cup 2022 release, whose terms of use were **not** opened in this work **[UNVERIFIED]**. Results computed from this excerpt describe one 200-frame sample only. |

## 6. SkillCorner open data, one match (`data/skillcorner/`)

| Field | Value |
| :--- | :--- |
| Repo | `https://github.com/SkillCorner/opendata` |
| Version | Commit `4340d274572876239c154c90bc507a9b3250a656` (2026-09-14, "Refresh four matches to dynamic events v3"), which was `master` HEAD on the download date |
| Files (match `1925299`) | `data/matches/1925299/1925299_phases_of_play.csv` → `possessions_on_ball.csv` (131,104 B, git blob `20c06b378f9a7181ab0e02de7c4a37779316386a`)<br>`data/matches/1925299/1925299_dynamic_events.csv` → `dynamic_events.csv` (5,043,601 B, git blob `09f226d38a3d9955d38ec32529a24c30c7958d22`)<br>Both are byte-identical to the upstream tree. |
| Naming note | The local name `possessions_on_ball.csv` is historical. Upstream, this file is a **phases-of-play** table (the notebook's "483 phases"). The name is kept because the notebook default path points to it. |
| SHA-256 | `possessions_on_ball.csv`: `a1682a736a9cf1c311cb799f0a949f7e023dca4ebaa9dd294b138e8929271274`<br>`dynamic_events.csv`: `2557acbdcba6547d764b9f7c277c3d7e81a4d58692d6c20c2bfa28625ae0cce3` |
| Date obtained | 2026-10-15 |
| License / terms | MIT License, "Copyright (c) 2020 SkillCorner" (`LICENSE` at the commit above) |
| Scope | One match out of the repository's 10 A-League matches. Use it as illustrative context only. |

---

## Known gaps

- **No upstream version tags.** The Mori Drive archive has none, so the zip
  checksum above is the only pin. If the authors replace the file, the checksum
  will no longer match.
- **`mori_code/` is a mixed snapshot.** See Section 2.
- **Unopened terms.** The PFF data-use terms and SoccerNet's terms for label
  files were not opened. Both are marked [UNVERIFIED].
- **Unreleased pilot inputs.** The December 2023 pilot inputs referenced in
  `soccer-bg-script/scripts/experiment-llm-annotation/exp0-1.sh`
  (`denoised_1_tokenized_224p_all.csv`,
  `gpt-3.5-turbo-1106_42_1_llm_annotation-20231217.csv`) are **not** in any
  public release. This is why the full 100-item pilot-vs-GPT κ is `NOT COMPUTED`
  in the notebook.
