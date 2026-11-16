"""Pins the twenty-seed numbers in README.md and in the module-02 notebook to
results/02_twenty_seeds.json, and that file to results/02_five_seeds.json and the per-seed
parts.  Nothing is retrained here.   Run: python -m pytest tests/
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
RES = json.loads((ROOT / "results" / "02_twenty_seeds.json").read_text())
FIVE = json.loads((ROOT / "results" / "02_five_seeds.json").read_text())
PARTS = ROOT / "results" / "parts_twenty"
METHODS = ["equal_lambda_1", "tuned_lambda_1000", "gradient_based"]
SEEDS = [str(s) for s in range(1, 21)]


def err(m):
    return [RES["runs"][m][s]["rel_l2"] for s in SEEDS]


def test_runs_come_from_the_stored_files():
    assert list(RES["runs"]) == METHODS and RES["setup"]["seeds"] == list(range(1, 21))
    for m in METHODS:
        for s in SEEDS[:5]:
            assert RES["runs"][m][s] == FIVE["runs"][m][s]
        for s in SEEDS[5:]:
            assert RES["runs"][m][s] == json.loads((PARTS / f"seed_{s}.json").read_text())["runs"][m]


def test_control_seed_reproduces_exactly():
    control = json.loads((PARTS / "seed_1.json").read_text())["runs"]
    assert RES["control_seed_1_identical"] is True
    assert all(control[m] == FIVE["runs"][m]["1"] for m in METHODS)


def test_summary_is_recomputed_from_runs():
    for m in METHODS:
        e, s = err(m), RES["summary"][m]
        assert s["per_seed"] == e
        assert s["mean"] == statistics.mean(e) and s["sd_sample"] == statistics.stdev(e)
        assert s["min"] == min(e) and s["max"] == max(e)
        assert s["n_under_2pct"] == sum(x < 0.02 for x in e)


def test_headline_table_in_readme_and_notebook():
    labels = {"equal_lambda_1": ("equal weights (λ = 1)", "equal weights, $\\lambda=1$"),
              "tuned_lambda_1000": ("tuned constant (λ = 1000)", "tuned constant, $\\lambda=1000$"),
              "gradient_based": ("gradient-based annealing", "gradient-based")}
    nb = json.loads((ROOT / "02_loss_balancing.ipynb").read_text())
    md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    for m, (readme_label, nb_label) in labels.items():
        s = RES["summary"][m]
        nums = f"{s['mean']:.4f} | {s['sd_sample']:.4f} | {s['min']:.4f} | {s['max']:.4f} |"
        assert f"| {readme_label} | {nums} {s['n_under_2pct']} of 20 |" in README
        assert f"| {nb_label} | {nums}" in md
    assert [RES["summary"][m]["n_under_2pct"] for m in METHODS] == [0, 10, 17]
    assert "21% to 74%" in md and "p = 0.26" in md


def test_per_seed_rows_in_readme():
    r = RES["runs"]
    for s in SEEDS[5:]:
        row = (f"| {s} | {r['equal_lambda_1'][s]['rel_l2']:.4f} | {r['tuned_lambda_1000'][s]['rel_l2']:.4f} | "
               f"{r['gradient_based'][s]['rel_l2']:.4f} | {r['gradient_based'][s]['lambda_final']:.0f} |")
        assert row in README, row


def test_the_tuned_constant_is_not_repeatable():
    t = err("tuned_lambda_1000")
    assert (round(min(t[:5]), 4), round(max(t[:5]), 4)) == (0.0097, 0.0118)
    assert sum(x < 0.016 for x in t[5:]) == 5
    assert sum(0.02 < x < 0.0735 for x in t[5:]) == 10 and round(max(t), 3) == 0.073
    assert round(statistics.stdev(t), 4) == 0.0187 and round(statistics.stdev(t[:5]), 4) == 0.0009
    assert "1.05% ± 0.09" in README
    assert round(100 * statistics.mean(t[:5]), 2) == 1.05 and round(100 * statistics.stdev(t[:5]), 2) == 0.09


def test_annealing_reading():
    g = err("gradient_based")
    assert [s for s, x in zip(SEEDS, g) if x > 0.02] == ["3", "4", "8"]
    b = RES["by_block"]["gradient_based"]["seeds_6_20"]
    assert round(100 * b["mean"], 2) == 1.40 and round(100 * b["sd_sample"], 2) == 0.51
    assert "1.40% ± 0.51" in README
    lam = RES["annealing_lambda_final"]
    assert (round(lam["min"]), round(lam["max"])) == (2866, 6199)
    assert "λ = 2866 to 6199" in README


def test_paired_reading():
    p = RES["paired"]["tuned_vs_gradient_based"]
    t, g = err("tuned_lambda_1000"), err("gradient_based")
    assert p["per_seed"] == [a - b for a, b in zip(t, g)]
    assert p["wins_for_first"] == 7 and p["n"] == 20          # annealing wins 13 of 20
    assert round(p["mean_difference"], 4) == 0.0090 and round(p["sd_sample"], 4) == 0.0241
    assert round(p["sign_test_p_two_sided"], 2) == 0.26
    assert "+0.0090 ± 0.0241" in README and "p = 0.26" in README
    assert sum(d < 0 for d in p["per_seed"][:5]) == 4          # the five-seed reading, for contrast


def test_loss_is_not_the_error_across_sixty_runs():
    runs = [(m, s, r["pde_loss"], r["rel_l2"]) for m in METHODS for s, r in RES["runs"][m].items()]
    assert len(runs) == 60
    best = min(runs, key=lambda x: x[2])
    assert best[:2] == ("equal_lambda_1", "6")
    assert round(best[2], 2) == 0.09 and round(100 * best[3]) == 21
    tuned = [r["pde_loss"] for r in RES["runs"]["tuned_lambda_1000"].values()]
    assert round(min(tuned), 2) == 0.17 and round(max(tuned), 1) == 5.4
    assert RES["lowest_pde_loss_run"] == {"method": best[0], "seed": 6, "pde_loss": best[2], "rel_l2": best[3]}


def test_readme_says_what_changed():
    assert "Correction, 16 November 2026" in README
    assert "Five seeds picked a winner that twenty do not confirm" in README
    assert "A tuned constant beats the adaptive scheme" not in README.replace("(a tuned", "(A tuned").split("## Per-seed check")[0]
