"""Pins the numbers quoted in README.md to the notebook outputs and to
results/02_five_seeds.json, so the page cannot drift from what was actually run.
Run: python -m pytest tests/   (no torch needed; nothing is retrained here)
"""
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
RES = json.loads((ROOT / "results" / "02_five_seeds.json").read_text())
SEEDS = [str(s) for s in RES["setup"]["seeds"]]


def notebook_outputs(name):
    nb = json.loads((ROOT / name).read_text())
    return "\n".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code"
                     for o in c.get("outputs", []))


def readme_table_row(label):
    """Numbers in the README summary-table row that starts with `label`."""
    for line in README.splitlines():
        if line.startswith(f"| {label}"):
            return [float(x) for x in re.findall(r"\d+\.\d{4}", line)]
    raise AssertionError(f"row '{label}' not in README")


# --- module 02: the summary table --------------------------------------------------

def test_summary_table_matches_results():
    rows = {"equal weights, lambda = 1": "equal_lambda_1",
            "tuned constant, lambda = 1000": "tuned_lambda_1000",
            "gradient-based (annealing)": "gradient_based"}
    for label, key in rows.items():
        mean, sd = readme_table_row(label)
        s = RES["summary"][key]
        assert round(s["mean"], 4) == mean, (label, s["mean"])
        assert round(s["sd_sample"], 4) == sd, (label, s["sd_sample"])   # sample sd, n = 5


def test_sd_is_sample_not_population():
    e = RES["summary"]["gradient_based"]["per_seed"]
    assert round(statistics.stdev(e), 4) == 0.0182
    assert round(statistics.pstdev(e), 4) != 0.0182


def test_twenty_times_more_repeatable():
    s = RES["summary"]
    ratio = s["gradient_based"]["sd_sample"] / s["tuned_lambda_1000"]["sd_sample"]
    assert 18 < ratio < 22


def test_per_seed_table():
    expected = {"equal_lambda_1": [0.4250, 0.4301, 0.3771, 0.5791, 0.2216],
                "tuned_lambda_1000": [0.0097, 0.0118, 0.0106, 0.0106, 0.0097],
                "gradient_based": [0.0128, 0.0089, 0.0483, 0.0394, 0.0121]}
    for key, vals in expected.items():
        assert [round(RES["runs"][key][s]["rel_l2"], 4) for s in SEEDS] == vals
    lam = [round(RES["runs"]["gradient_based"][s]["lambda_final"]) for s in SEEDS]
    assert lam == [4332, 5269, 3880, 3786, 5080]
    assert 3750 <= min(lam) and max(lam) <= 5300


def test_paired_reading():
    p = RES["paired"]["tuned_vs_gradient_based"]
    assert p["wins_for_first"] == 4
    d = p["per_seed"]
    assert d[1] > 0                                   # seed 2: gradient-based wins
    assert sum(abs(x) < 0.004 for x in d) == 3        # three seeds within 0.004
    g = RES["runs"]["gradient_based"]
    assert 0.04 <= g["3"]["rel_l2"] <= 0.05 and 0.03 <= g["4"]["rel_l2"] <= 0.05


def test_loss_is_not_the_error_across_all_runs():
    runs = [(m, s, r["pde_loss"], r["rel_l2"]) for m in RES["runs"] for s, r in RES["runs"][m].items()]
    assert len(runs) == 15
    best = min(runs, key=lambda t: t[2])
    assert best[0] == "equal_lambda_1" and best[1] == "5"
    assert round(best[2], 2) == 0.17 and round(100 * best[3]) == 22
    tuned = [r["pde_loss"] for r in RES["runs"]["tuned_lambda_1000"].values()]
    assert round(min(tuned), 2) == 0.30 and round(max(tuned), 2) == 0.60


def test_notebook_single_run_matches_seed_1():
    out = notebook_outputs("02_loss_balancing.ipynb")
    assert "0.4076" in out and "0.4435" in out and "0.4250" in out and "0.0097" in out
    r = RES["runs"]
    assert round(r["equal_lambda_1"]["1"]["pde_loss"], 4) == 0.4076
    assert round(r["tuned_lambda_1000"]["1"]["pde_loss"], 4) == 0.4435
    assert round(r["equal_lambda_1"]["1"]["rel_l2"], 4) == 0.4250
    assert round(r["gradient_based"]["1"]["lambda_final"]) == 4332


# --- module 01: the Cole-Hopf check --------------------------------------------------

def test_module_01_error_range():
    out = notebook_outputs("01_burgers_from_scratch.ipynb")
    errs = [float(m) for m in re.findall(r"^\s*[01]\.\d{2}\s+(\d\.\d{4})\s+\d", out, re.M)]
    assert len(errs) == 4, errs
    assert min(errs) == 0.0008 and max(errs) == 0.0040     # "between 0.08% and 0.4%"
    assert "0.08% and 0.4%" in README
