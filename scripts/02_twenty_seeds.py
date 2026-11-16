"""Extend the module-02 comparison from 5 seeds to 20 and write results/02_twenty_seeds.json.

Seeds 1..5 are taken from results/02_five_seeds.json. Seeds 6..20 are trained here with the
unchanged training loop of scripts/02_five_seeds.py, one file per seed in
results/parts_twenty/. Seed 1 can be trained again as a control: it must reproduce the stored
run exactly, and the merge step checks that it does.

Run:  python scripts/02_twenty_seeds.py 6 7 8    train these seeds (about two minutes per seed
                                                 alone on a laptop CPU; several processes can
                                                 run side by side, each with its own seeds)
      python scripts/02_twenty_seeds.py          merge everything into the results file
"""
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIVE = ROOT / "results" / "02_five_seeds.json"
PARTS = ROOT / "results" / "parts_twenty"
OUT = ROOT / "results" / "02_twenty_seeds.json"
SEEDS = list(range(1, 21))
GOOD = 0.02


def train_seeds(seeds):
    spec = importlib.util.spec_from_file_location("five", ROOT / "scripts" / "02_five_seeds.py")
    five = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(five)
    PARTS.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        f = PARTS / f"seed_{seed}.json"
        if f.exists():
            continue
        t0 = time.time()
        runs = {m: five.train(seed, lam) for m, lam in five.METHODS.items()}
        f.write_text(json.dumps({"seed": seed, "runs": runs, "wall_time_s": time.time() - t0}))
        print(seed, {m: round(r["rel_l2"], 4) for m, r in runs.items()}, flush=True)


def stats(e):
    return {"mean": statistics.mean(e), "sd_sample": statistics.stdev(e), "min": min(e),
            "max": max(e), "n_under_2pct": sum(x < GOOD for x in e), "per_seed": e}


def sign_test_two_sided(wins, n):
    k = max(wins, n - wins)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def merge():
    five = json.loads(FIVE.read_text())
    methods = list(five["runs"])
    runs = {m: {str(s): five["runs"][m][str(s)] for s in five["setup"]["seeds"]} for m in methods}
    control = json.loads((PARTS / "seed_1.json").read_text())["runs"]
    control_identical = all(control[m] == five["runs"][m]["1"] for m in methods)
    for s in range(6, 21):
        part = json.loads((PARTS / f"seed_{s}.json").read_text())["runs"]
        for m in methods:
            runs[m][str(s)] = part[m]
    err = {m: [runs[m][str(s)]["rel_l2"] for s in SEEDS] for m in methods}
    t, g = err["tuned_lambda_1000"], err["gradient_based"]
    diff = [a - b for a, b in zip(t, g)]                       # tuned minus gradient-based
    wins = sum(d < 0 for d in diff)
    flat = [(m, s, r["pde_loss"], r["rel_l2"]) for m in methods for s, r in runs[m].items()]
    low = min(flat, key=lambda x: x[2])
    lam = [runs["gradient_based"][str(s)]["lambda_final"] for s in SEEDS]
    setup = dict(five["setup"], seeds=SEEDS)
    out = {
        "problem": five["problem"], "setup": setup, "error_metric": five["error_metric"],
        "source": "seeds 1-5 from results/02_five_seeds.json; seeds 6-20 from results/parts_twenty/",
        "control_seed_1_identical": control_identical,
        "summary": {m: stats(err[m]) for m in methods},
        "by_block": {m: {"seeds_1_5": stats(err[m][:5]), "seeds_6_20": stats(err[m][5:])}
                     for m in methods},
        "paired": {"tuned_vs_gradient_based": {
            "per_seed": diff, "wins_for_first": wins, "n": len(diff),
            "mean_difference": statistics.mean(diff), "sd_sample": statistics.stdev(diff),
            "sign_test_p_two_sided": sign_test_two_sided(wins, len(diff))}},
        "lowest_pde_loss_run": {"method": low[0], "seed": int(low[1]), "pde_loss": low[2],
                                "rel_l2": low[3]},
        "annealing_lambda_final": {"min": min(lam), "max": max(lam), "per_seed": lam},
        "runs": runs,
    }
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_seeds([int(a) for a in sys.argv[1:]])
    else:
        o = merge()
        print("control seed 1 identical:", o["control_seed_1_identical"])
        for m, s in o["summary"].items():
            print(f"{m:18s} mean {s['mean']:.4f} sd {s['sd_sample']:.4f} min {s['min']:.4f} "
                  f"max {s['max']:.4f} under 2%: {s['n_under_2pct']}/20")
        for m, b in o["by_block"].items():
            print(f"{m:18s} 1-5: {b['seeds_1_5']['mean']:.4f} +- {b['seeds_1_5']['sd_sample']:.4f}"
                  f" ({b['seeds_1_5']['n_under_2pct']}/5)   6-20: {b['seeds_6_20']['mean']:.4f} +- "
                  f"{b['seeds_6_20']['sd_sample']:.4f} ({b['seeds_6_20']['n_under_2pct']}/15)")
        p = o["paired"]["tuned_vs_gradient_based"]
        print("tuned wins", p["wins_for_first"], "of", p["n"], "mean diff", round(p["mean_difference"], 4),
              "sd", round(p["sd_sample"], 4), "sign test p", round(p["sign_test_p_two_sided"], 3))
        print("lowest pde loss:", o["lowest_pde_loss_run"])
        print("lambda final:", round(o["annealing_lambda_final"]["min"]), round(o["annealing_lambda_final"]["max"]))
