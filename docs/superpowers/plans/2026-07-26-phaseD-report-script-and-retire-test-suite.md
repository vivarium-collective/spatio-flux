# Phase D: Report Script + Retire `test_suite.py` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status (2026-07-26): EXECUTED (core), 2 items open.** `scripts/build_report.py`
> reproduces `report.html` by REUSING `generate_html_report` (merges per-study
> `charts/` + study-sourced descriptions/timing) — verified end-to-end (19-scenario
> `report/index.html`, 152 KB, cards/imgs/gifs/descriptions). `scripts/reproduce.py`
> runs the investigation then builds the report; README repointed to it. Simpler than
> the plan's Task-1 refactor: no `render_report` extraction needed — full renderer reuse.
> **OPEN:** (1) **deleting `test_suite.py` is blocked** — `experiments/figure_A.py` and
> `figure_B.py` import it (paper-figure scripts); needs a decision (port them / retire
> them / keep a shim) before deletion. (2) **CI publish (Task 5) not wired** — needs a
> push + a GitHub Actions run.

**Goal:** Produce `report.html` in the current look — the published page at `vivarium-collective.github.io/spatio-flux/report/index.html` — but driven by the investigation's study artifacts instead of an `out/` scan, and retire `test_suite.py` so the investigation is the single source of truth.

**Architecture:** Refactor `spatio_flux/library/tools.py::generate_html_report` into a **data-in renderer** (`render_report(report_data)`) that consumes a `report_data` structure and reuses the existing CSS / sticky-TOC / per-scenario-card / JSON-state-viewer HTML verbatim. A `scripts/build_report.py` collects `report_data` by walking `investigations/spatio-flux-test-suite/` + each `studies/<slug>/charts|viz` + `<slug>_timing.json` + the composite `description`. A `scripts/reproduce.py` runs every study (the investigation) then builds the report — replacing `test_suite.py`, which is deleted. CI publishes both the report and the native `vivarium-workbench-publish` bundle.

**Tech Stack:** existing `spatio_flux/library/tools.py` HTML/CSS/JSON-viewer code (reused), `pyyaml`, `pytest`, GitHub Actions.

## Global Constraints

- **Worktree** `~/code/spatio-flux--workbench-modernization`, branch `workbench-modernization`. Verify branch/HEAD before commits.
- **Depends on Phase B** — the report consumes `studies/<slug>/charts/*` produced by the analysis flush. Run Phase B first (or against `out0/` fixtures for the renderer tests).
- **Byte-for-look fidelity, not byte-for-byte** — the report must have the same sections, cards, artifacts, TOC, About/ecosystem/how-to blocks, and meta pills as the current `report.html`. Reuse the existing CSS strings and viewer JS in `tools.py` unchanged; only the *data source* changes (study artifacts, not an `out/` directory scan).
- **`test_suite.py` is deleted in this phase** (per the approved decision to retire it) — but only after `reproduce.py` + the report reproduce its function and any still-used helpers (`STANDARD_FIELD_COLORS`, plot configs) are relocated into `spatio_flux/analysis/` or `plots/`.
- **Descriptions come from the `@composite_generator`** (`entry.description`) as today (`test_suite.py:627-633`), now looked up per study from its `baseline.composite` ref.

## What the current report renders per scenario (preserve exactly)

From `generate_html_report` (`tools.py:888-1193`): hero header; sticky-TOC two-column layout; About/overview (callout grid, process-families table, Vivarium-2.0 ecosystem list, references, meta pills: total sim time + timestamp); "how to read the bigraph" block; then **per scenario**: `<h2>` anchor + one-line description, runtime line (`total (process: …s, framework: …s)`), interactive JSON state viewer (from `<slug>_state.json`), bigraph diagram (`<slug>_viz.png`), PNG plots, GIFs; then "Other Generated Files" + "Total Simulation Time".

---

### Task 1: Extract a data-in report renderer

**Files:**
- Create: `spatio_flux/library/report_render.py`
- Modify: `spatio_flux/library/tools.py` (have `generate_html_report` delegate to the new renderer, preserving its current signature/behavior)
- Test: `tests/test_report.py`

**Interfaces:**
- Produces: `render_report(report_data: dict) -> str` (full `report.html` string). `report_data`:
  ```python
  {"title": str, "total_sim_time": float, "generated": str,
   "scenarios": [{"slug", "description", "timing": {"process","framework","elapsed"},
                  "state_json_path": str|None, "diagram_png": str|None,
                  "pngs": [paths], "gifs": [paths], "other": [paths]}]}
  ```
  The renderer reuses the exact CSS/TOC/card/JSON-viewer helpers currently in `tools.py` (`_json_viewer_css_lines`, `_render_json_api_viewer_block`, `_spatio_flux_intro_html`, `_how_to_read_bigraph_html`, `_json_viewer_js`, `_safe_id`, `test_files` grouping) — move or import them, don't rewrite.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
def test_render_report_has_all_sections():
    from spatio_flux.library.report_render import render_report
    data = {"title": "Spatio-Flux Test Suite Report", "total_sim_time": 1.0,
            "generated": "2026-07-26", "scenarios": [
                {"slug": "monod_kinetics", "description": "Monod kinetics.",
                 "timing": {"process": 0.5, "framework": 0.5, "elapsed": 1.0},
                 "state_json_path": None, "diagram_png": None,
                 "pngs": ["monod_kinetics.png"], "gifs": [], "other": []}]}
    html = render_report(data)
    assert "Spatio-Flux Test Suite Report" in html
    assert 'id="' in html and "monod_kinetics" in html
    assert "monod_kinetics.png" in html
    assert "process:" in html  # runtime line preserved
```

- [ ] **Step 2: Run to verify it fails.**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_report.py -q`
Expected: FAIL — `report_render` missing.

- [ ] **Step 3: Implement `report_render.py`** by lifting the assembly loop from `generate_html_report` (`tools.py:995-1181`) into `render_report(report_data)`, sourcing per-scenario content from `report_data["scenarios"]` instead of scanning `output_dir`. Keep every helper call identical. Then make `tools.generate_html_report` build `report_data` from its `output_dir` scan and call `render_report` — preserving its current behavior so nothing else breaks.

- [ ] **Step 4: Run to verify it passes.**

Run: `PYTHONPATH=$PWD ~/code/spatio-flux/.venv/bin/python -m pytest tests/test_report.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spatio_flux/library/report_render.py spatio_flux/library/tools.py tests/test_report.py
git commit -m "refactor(report): data-in render_report; generate_html_report delegates"
```

---

### Task 2: `build_report.py` — collect report_data from studies

**Files:**
- Create: `scripts/build_report.py`
- Test: `tests/test_report.py` (extend)

**Interfaces:**
- Produces: `collect_report_data(ws_root) -> dict` (the `report_data` above) + `main(out_path)`. Walks `investigations/spatio-flux-test-suite/investigation.yaml` for scenario order; for each study reads `studies/<slug>/charts/` for `<slug>_viz.png` / `_state.json` / `*.png` / `*.gif`, `<slug>_timing.json` for timing, and the composite `description` from `pbg_superpowers.composite_generator._REGISTRY[baseline.composite].description`.

- [ ] **Step 1: Write the failing test** (collect from a fixture studies dir built from `out0/`)

```python
def test_collect_report_data_orders_by_investigation(tmp_path):
    from scripts.build_report import collect_report_data
    # minimal fake workspace: investigation.yaml (2 slugs) + studies/<slug>/charts/<slug>.png + _timing.json
    ... # build tmp_path fixture
    data = collect_report_data(str(tmp_path))
    assert [s["slug"] for s in data["scenarios"]][:2] == ["monod_kinetics", "ecoli_core_dfba"]
    assert data["scenarios"][0]["pngs"]
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `build_report.py`.** `collect_report_data` reads the investigation `studies:` order, globs each study's `charts/`, classifies files (diagram `_viz.png`, `_state.json`, other `*.png`, `*.gif`) exactly as `test_files` does in `tools.py:1007-1027`, sums timing into `total_sim_time`. `main(out)` = `render_report(collect_report_data("."))` → write `out`.

- [ ] **Step 4: Run to verify it passes**, then generate against the real workspace (after Phase B):
`.venv-serve/bin/python scripts/build_report.py --out report/index.html` and open it — confirm 19 scenario cards with the current look.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_report.py tests/test_report.py
git commit -m "feat(report): build_report.py collects report_data from study artifacts"
```

---

### Task 3: `reproduce.py` — run the investigation, then build the report

**Files:**
- Create: `scripts/reproduce.py`
- Test: `tests/test_report.py` (smoke, `@pytest.mark.slow`)

- [ ] **Step 1: Write the failing test** (reproduce a single cheap scenario + report exists)

```python
@pytest.mark.slow
def test_reproduce_one_scenario_builds_report(tmp_path):
    # run reproduce.py --only monod_kinetics into a temp workspace; assert report/index.html exists
    ...
```

- [ ] **Step 2–3: Implement `reproduce.py`.** Iterates the investigation's `studies:` (or `--only <slug>`), calls `spatio_flux/runners/run_study.py` for each (Phase B runner), then `scripts.build_report.main("report/index.html")`. This is the retirement replacement for `test_suite.py main()`.

```python
"""Reproduce the whole spatio-flux test suite from the investigation:
run every study, then build the report. Replaces test_suite.py."""
import subprocess, sys, yaml, os
from scripts.build_report import main as build_report

def main(only=None):
    inv = yaml.safe_load(open("investigations/spatio-flux-test-suite/investigation.yaml"))
    slugs = [only] if only else inv["studies"]
    for slug in slugs:
        subprocess.run([sys.executable, "spatio_flux/runners/run_study.py", slug], check=True)
    build_report("report/index.html")
```

- [ ] **Step 4: Run** (serving venv, one scenario). **Step 5: Commit.**

```bash
git add scripts/reproduce.py tests/test_report.py
git commit -m "feat(report): reproduce.py runs investigation + builds report (test_suite replacement)"
```

---

### Task 4: Retire `test_suite.py`

**Files:**
- Delete: `spatio_flux/experiments/test_suite.py`
- Modify: relocate any still-referenced constants (`STANDARD_FIELD_COLORS`, per-scenario plot configs already captured in Phase-B `FLUSH_SPEC`) into `spatio_flux/plots/colors.py` or `spatio_flux/analysis/`
- Test: `tests/test_report.py` (extend)

- [ ] **Step 1: Write the failing test** (nothing imports test_suite; README/docs point at reproduce.py)

```python
def test_test_suite_module_removed():
    import importlib.util
    assert importlib.util.find_spec("spatio_flux.experiments.test_suite") is None

def test_no_references_to_test_suite():
    import subprocess
    r = subprocess.run(["grep","-rIl","experiments.test_suite","spatio_flux","scripts"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.stdout.strip() == "", f"stale references: {r.stdout}"
```

- [ ] **Step 2: Run to verify it fails** (module still present).

- [ ] **Step 3:** Move `STANDARD_FIELD_COLORS` (`test_suite.py:64-76`) into `spatio_flux/plots/colors.py`; confirm Phase-B `FLUSH_SPEC` already owns every scenario's plot config (it does — Phase B Task 3). Delete `spatio_flux/experiments/test_suite.py`. Update `README.md` run instructions to `uv run python scripts/reproduce.py` + serving the dashboard.

- [ ] **Step 4: Run to verify it passes** — the two removal tests + the full suite (`pytest tests/ -q`).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: retire test_suite.py; investigation + reproduce.py are the source of truth"
```

---

### Task 5: CI — publish report + native dashboard bundle

**Files:**
- Create/Modify: `.github/workflows/publish.yml`
- Test: workflow dry-run / manual

- [ ] **Step 1:** Add a job that, on push to main (or manual dispatch): sets up the serving venv, runs `scripts/reproduce.py` (or downloads cached `charts/`), runs `scripts/build_report.py --out report/index.html`, and `vivarium-workbench-publish --workspace . --out site/` for the native read-only bundle; publishes both to `gh-pages` (report at `report/index.html` to preserve the existing URL, native bundle alongside).
- [ ] **Step 2:** Verify the published `report/index.html` matches the current look and the native dashboard bundle loads. Record + commit.

---

## Self-Review

**Spec coverage (design §4.5):** custom report script reproducing current look → Tasks 1–2; native dashboard published too → Task 5; retire `test_suite.py` + `reproduce.py` shim → Tasks 3–4. ✓

**Placeholder scan:** renderer reuses named existing helpers (no HTML rewritten); the two fixture-building `...` lines in tests are the standard "construct tmp workspace" step whose contents are specified in prose (investigation.yaml + studies/<slug>/charts + _timing.json). ✓

**Type/name consistency:** `report_data` shape identical across `render_report` (Task 1), `collect_report_data` (Task 2), `reproduce.py` (Task 3). ✓

**Dependencies:** Tasks 2–5 consume Phase B artifacts (`charts/`, `_timing.json`); the renderer (Task 1) is testable standalone against `out0/`.
