"""Pins the README section "Exercises 2 and 3, run" to results/02_lambda_basin.json, and
that file to results/02_five_seeds.json and to the per-run parts.  Nothing is retrained
here (50 runs, about half an hour).  Run: python -m pytest tests/
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = " ".join((ROOT / "README.md").read_text().split())      # whitespace collapsed
RES = json.loads((ROOT / "results" / "02_lambda_basin.json").read_text())
FIVE = json.loads((ROOT / "results" / "02_five_seeds.json").read_text())
SEEDS = [str(s) for s in RES["setup"]["seeds"]]
GRID = RES["grid"]
GOOD = RES["good_threshold"]


def err(lam, seed):
    return RES["runs"][str(lam)][seed]["rel_l2"]


def sweep_row(lam):
    s = RES["sweep"][str(lam)]
    under = sum(e < GOOD for e in s["per_seed"])
    return (f"| {lam} | {s['mean']:.4f} | {s['sd_sample']:.4f} | {s['min']:.4f} | "
            f"{s['max']:.4f} | {under} of {len(SEEDS)} |")


def endpoint_row(seed):
    e = RES["endpoint"]["per_seed"][seed]
    return (f"| {seed} | {e['lambda']:.0f} | {e['annealed']['rel_l2']:.4f} | "
            f"{e['constant_from_step_one']['rel_l2']:.4f} | {err(1000, seed):.4f} |")


def test_control_column_reproduces_the_stored_runs():
    """lambda = 1000 was trained again by the new script; it must give back the 22 September runs."""
    c = RES["control_lambda_1000"]
    assert c["identical"] and c["max_abs_diff"] == 0.0
    for s in SEEDS:
        assert err(1000, s) == FIVE["runs"]["tuned_lambda_1000"][s]["rel_l2"]
        assert RES["runs"]["1000"][s]["pde_loss"] == FIVE["runs"]["tuned_lambda_1000"][s]["pde_loss"]
    assert "bit for bit" in README


def test_lambda_1_column_and_annealed_runs_are_the_stored_ones():
    for s in SEEDS:
        assert RES["runs"]["1"][s] == FIVE["runs"]["equal_lambda_1"][s]
        e = RES["endpoint"]["per_seed"][s]
        assert e["annealed"] == FIVE["runs"]["gradient_based"][s]
        assert e["lambda"] == FIVE["runs"]["gradient_based"][s]["lambda_final"]
        assert e["constant_from_step_one"]["lambda_final"] == e["lambda"]


def test_parts_are_the_runs():
    parts = ROOT / "results" / "parts_basin"
    for lam in GRID[1:]:
        for s in SEEDS:
            assert json.loads((parts / f"seed_{s}_lam_{lam}.json").read_text()) == RES["runs"][str(lam)][s]
    for s in SEEDS:
        assert (json.loads((parts / f"seed_{s}_lam_endpoint.json").read_text())
                == RES["endpoint"]["per_seed"][s]["constant_from_step_one"])
    assert all(r["torch_threads"] == 1 for lam in GRID[1:] for r in RES["runs"][str(lam)].values())


def test_sweep_summary_is_computed_from_the_runs():
    assert GRID == [1, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
    for lam in GRID:
        e = [err(lam, s) for s in SEEDS]
        s_ = RES["sweep"][str(lam)]
        assert s_["per_seed"] == e
        assert abs(s_["mean"] - statistics.mean(e)) < 1e-12
        assert abs(s_["sd_sample"] - statistics.stdev(e)) < 1e-12        # sample sd, n = 5


def test_readme_tables():
    for lam in GRID:
        assert sweep_row(lam) in README, sweep_row(lam)
    for s in SEEDS:
        assert endpoint_row(s) in README, endpoint_row(s)


def test_basin_is_a_decade_wide_and_lopsided():
    assert RES["lambdas_under_2pct_on_every_seed"] == [1000]
    under = {lam: sum(err(lam, s) < GOOD for s in SEEDS) for lam in GRID}
    assert [lam for lam in GRID if under[lam] == 4] == [300, 3000]
    assert all(under[lam] == 0 for lam in (1, 10, 30, 100, 10000, 30000, 100000))
    m = {lam: 100 * RES["sweep"][str(lam)]["mean"] for lam in GRID}
    assert [f"{m[lam]:.1f}" for lam in (300, 1000, 3000)] == ["1.6", "1.0", "1.6"]
    assert "at a mean of 1.6 % against 1.0 %" in README
    assert [f"{m[lam]:.0f}" for lam in (1, 10)] == ["41", "13"] and f"{m[30]:.1f}" == "4.6"
    assert "41 % at λ = 1, 13 % at 10, 4.6 % at 30" in README
    assert f"{m[10000]:.1f}" == "4.2" and f"{m[30000]:.0f}" == "83"
    assert "4.2 % at 10 000, then 83 % at 30 000" in README
    regime = [err(lam, s) for lam in (30, 100, 300, 1000, 3000, 10000) for s in SEEDS]
    assert 0.009 < min(regime) and max(regime) < 0.07
    assert "between 0.9 % and 7 % on every seed" in README
    assert sum(err(30000, s) > err(1, s) for s in SEEDS) == 4          # worse than no weighting
    assert "on four seeds of five" in README
    assert [RES["per_seed"][s]["best_lambda"] for s in SEEDS] == [1000, 1000, 1000, 1000, 3000]
    assert (f"{err(3000, '5'):.4f}", f"{err(1000, '5'):.4f}") == ("0.0093", "0.0097")
    assert "Seed 5 prefers 3000 (0.0093 against 0.0097)" in README


def test_zero_function_at_one_hundred_thousand():
    zero = RES["pde_loss_of_zero_function"]
    assert f"{zero:.0f}" == "6896"
    for s in SEEDS:
        r = RES["runs"]["100000"][s]
        assert 6895 < r["pde_loss"] < 6896 and r["pde_loss"] < zero
        assert r["bc_loss"] < 2e-6 and r["rel_l2"] > 0.9996
    assert "PDE loss between 6895 and 6896 (u = 0 scores 6896" in README


def test_exercise_3_constant_at_the_annealed_endpoint():
    e = RES["endpoint"]
    c, a = e["constant"]["per_seed"], e["annealed"]["per_seed"]
    tuned = [err(1000, s) for s in SEEDS]
    assert [x < y for x, y in zip(c, a)] == [False, False, True, True, True] and e["constant_wins"] == 3
    assert (f"{100 * c[0]:.1f}", f"{100 * a[0]:.1f}", f"{c[0] / a[0]:.1f}") == ("5.9", "1.3", "4.6")
    assert "ends at 5.9 % error where the annealed run reached 1.3 %: 4.6 times worse" in README
    assert (f"{100 * c[1]:.1f}", f"{100 * a[1]:.1f}") == ("3.0", "0.9")
    assert [f"{100 * x:.1f}" for x in (c[2], a[2], c[3], a[3])] == ["2.8", "4.8", "2.7", "3.9"]
    assert "(2.8 % against 4.8 %, 2.7 % against 3.9 %)" in README
    assert all(x > 2 * t for x, t in zip(c[2:4], tuned[2:4])) and [f"{100 * t:.1f}" for t in tuned[2:4]] == ["1.1", "1.1"]
    assert (f"{100 * e['constant']['mean']:.1f}", f"{100 * e['annealed']['mean']:.1f}") == ("3.1", "2.4")
    assert f"{a[4] - c[4]:.4f}" == "0.0009"
    assert (f"{e['diff_mean']:+.3f}", f"{e['diff_sd']:.3f}") == ("+0.007", "0.027")
    assert "+0.007 ± 0.027" in README
    assert all(x > t for x, t in zip(c, tuned))                        # tuned 1000 wins on every seed
    assert 2.9 < e["constant"]["mean"] / statistics.mean(tuned) < 3.1
    assert sum(x < GOOD for x in c) == 1
    lam = [e["per_seed"][s]["lambda"] for s in SEEDS]
    assert (f"{min(lam):.0f}", f"{max(lam):.0f}") == ("3786", "5269")
    assert 5.5 < 30000 / max(lam) < 6.5 and 7.5 < 30000 / min(lam) < 8.5
    assert all(l > RES["per_seed"][s]["best_lambda"] for l, s in zip(lam, SEEDS))
    assert "neither sentence survives as written" in README


if __name__ == "__main__":
    for lam in GRID:
        print(sweep_row(lam))
    print()
    for s in SEEDS:
        print(endpoint_row(s))
    print()
    print(json.dumps({k: RES[k] for k in ("control_lambda_1000", "lambdas_under_2pct_on_every_seed", "per_seed")}, indent=1))
    e = RES["endpoint"]
    print({k: e[k] for k in ("constant", "annealed", "diff_mean", "diff_sd", "constant_wins")})
