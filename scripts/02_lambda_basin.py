"""Exercises 2 and 3 of module 02, run instead of left to the reader.

Exercise 2: sweep the constant weight lambda over 10^0 .. 10^5 and see how wide the
basin of good values is.  Exercise 3: fix lambda from step one at the value the
gradient-based scheme ended at, with no annealing, and see whether it trains.

Same network, points, budget and training loop as scripts/02_five_seeds.py -- its
train() is imported, not copied -- for seeds 1..5.  The lambda = 1 column and the
annealed runs are read from results/02_five_seeds.json; the lambda = 1000 column is
trained again here as a control and must reproduce the stored runs.

Run:  python scripts/02_lambda_basin.py --job SEED a|b    (ten jobs, one process each)
      python scripts/02_lambda_basin.py --merge
      python scripts/02_lambda_basin.py                   (everything, in sequence)
One run is five to seven minutes on one CPU thread with ten running side by side on a
laptop (OMP_NUM_THREADS=1); 50 runs in all, about half an hour.
Writes results/02_lambda_basin.json.
"""
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "02_lambda_basin.json"
PARTS = ROOT / "results" / "parts_basin"
GRID = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)   # lambda = 1 is already stored
GOOD = 0.02                                                    # "good" = under 2 % error


def five_seeds_module():
    spec = importlib.util.spec_from_file_location("five_seeds", ROOT / "scripts" / "02_five_seeds.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stored():
    return json.loads((ROOT / "results" / "02_five_seeds.json").read_text())


def run_one(fs, seed, label):
    """label is a grid value ("3000") or "endpoint" (where the annealed run of this seed ended)."""
    import torch
    lam = (stored()["runs"]["gradient_based"][str(seed)]["lambda_final"]
           if label == "endpoint" else float(label))
    t0 = time.time()
    r = fs.train(seed, lam)
    r.update(wall_time_s=time.time() - t0, torch_threads=torch.get_num_threads())
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"seed_{seed}_lam_{label}.json").write_text(json.dumps(r))
    print(f"seed {seed}  lambda {label:>8s} ({lam:9.1f})  rel L2 {r['rel_l2']:.4f}  "
          f"pde {r['pde_loss']:.4f}  bc {r['bc_loss']:.2e}  {r['wall_time_s']:4.0f}s", flush=True)


def jobs(seed):
    labels = [str(g) for g in GRID] + ["endpoint"]
    return {"a": labels[:5], "b": labels[5:]}


def stats(e):
    return {"mean": statistics.mean(e), "sd_sample": statistics.stdev(e),
            "min": min(e), "max": max(e), "per_seed": e}


def merge(fs):
    seeds = fs.SEEDS
    five = stored()
    part = lambda s, lab: json.loads((PARTS / f"seed_{s}_lam_{lab}.json").read_text())
    runs = {"1": {str(s): five["runs"]["equal_lambda_1"][str(s)] for s in seeds}}
    for g in GRID:
        runs[str(g)] = {str(s): part(s, str(g)) for s in seeds}
    lams = [1] + list(GRID)
    err = lambda lab, s: runs[str(lab)][str(s)]["rel_l2"]
    sweep = {str(l): stats([err(l, s) for s in seeds]) for l in lams}

    # control: the lambda = 1000 column against the runs stored on 2026-09-22
    old = [five["runs"]["tuned_lambda_1000"][str(s)]["rel_l2"] for s in seeds]
    new = [err(1000, s) for s in seeds]
    control = {"stored": old, "rerun": new, "identical": old == new,
               "max_abs_diff": max(abs(a - b) for a, b in zip(old, new))}

    per_seed = {}
    for s in seeds:
        e = {l: err(l, s) for l in lams}
        best = min(e, key=e.get)
        good = [l for l in lams if e[l] < GOOD]
        per_seed[str(s)] = {"best_lambda": best, "best_rel_l2": e[best],
                            "lambdas_under_2pct": good,
                            "lambdas_within_2x_of_best": [l for l in lams if e[l] <= 2 * e[best]]}
    under_all = [l for l in lams if all(err(l, s) < GOOD for s in seeds)]

    endpoint = {}
    for s in seeds:
        c, a = part(s, "endpoint"), five["runs"]["gradient_based"][str(s)]
        endpoint[str(s)] = {"lambda": a["lambda_final"],
                            "constant_from_step_one": c, "annealed": a,
                            "diff_constant_minus_annealed": c["rel_l2"] - a["rel_l2"]}
    d = [endpoint[str(s)]["diff_constant_minus_annealed"] for s in seeds]
    ce = [endpoint[str(s)]["constant_from_step_one"]["rel_l2"] for s in seeds]

    import torch
    xf, yf, _, _ = fs.make_data(seed=0)
    zero = fs.source(xf, yf).pow(2).mean().item()      # PDE loss of u = 0 on the collocation points
    out = {
        "problem": five["problem"], "setup": dict(five["setup"], torch=torch.__version__),
        "error_metric": five["error_metric"],
        "grid": lams, "good_threshold": GOOD,
        "note": "lambda = 1 runs and the annealed runs are those of 02_five_seeds.json; "
                "every other run was trained by this script, one CPU thread per process",
        "control_lambda_1000": control,
        "pde_loss_of_zero_function": zero,
        "sweep": sweep,
        "lambdas_under_2pct_on_every_seed": under_all,
        "per_seed": per_seed,
        "endpoint": {"per_seed": endpoint, "constant": stats(ce),
                     "annealed": stats([endpoint[str(s)]["annealed"]["rel_l2"] for s in seeds]),
                     "diff_mean": statistics.mean(d), "diff_sd": statistics.stdev(d),
                     "constant_wins": sum(x < 0 for x in d)},
        "runs": runs,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    for l in lams:
        s_ = sweep[str(l)]
        print(f"lambda {l:>6d}  mean {s_['mean']:.4f}  sd {s_['sd_sample']:.4f}  "
              f"min {s_['min']:.4f}  max {s_['max']:.4f}")
    print("control identical:", control["identical"], " max diff", control["max_abs_diff"])
    print("endpoint, constant - annealed:", [round(x, 4) for x in d])


def main():
    args = sys.argv[1:]
    fs = five_seeds_module()
    if args[:1] == ["--job"]:
        seed = int(args[1])
        for label in jobs(seed)[args[2]]:
            run_one(fs, seed, label)
        return 0
    if args[:1] != ["--merge"]:
        for seed in fs.SEEDS:
            for half in ("a", "b"):
                for label in jobs(seed)[half]:
                    run_one(fs, seed, label)
    merge(fs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
