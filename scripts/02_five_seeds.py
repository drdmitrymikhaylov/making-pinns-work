"""Re-run the five-seed table of module 02 and write it to results/02_five_seeds.json.

The notebook 02_loss_balancing.ipynb shows one seed in its outputs and quotes the
five-seed table (mean / sd / min / max) as text. This script is what produces that
table: the same network, data, budget and three weightings as the notebook, for
seeds 1..5, with the per-seed errors kept so the table can be checked and the two
weightings compared pair by pair on the same seed.

Run:  python scripts/02_five_seeds.py          (about ten minutes on a laptop CPU)
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

A1, A2, K = 1.0, 4.0, 1.0          # same problem as the notebook
ITERS, LAM_TUNED, ALPHA = 4000, 1000.0, 0.9
SEEDS = (1, 2, 3, 4, 5)
OUT = Path(__file__).resolve().parents[1] / "results" / "02_five_seeds.json"


def exact(x, y):
    return torch.sin(A1 * np.pi * x) * torch.sin(A2 * np.pi * y)


def source(x, y):
    return (K ** 2 - (A1 * np.pi) ** 2 - (A2 * np.pi) ** 2) * exact(x, y)


class MLP(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers, d_in = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def pde_residual(model, x, y):
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    u = model(x, y)
    g = lambda a, b: torch.autograd.grad(a, b, torch.ones_like(a), create_graph=True)[0]
    return g(g(u, x), x) + g(g(u, y), y) + K ** 2 * u - source(x, y)


def grad_norm(loss, model):
    gs = torch.autograd.grad(loss, list(model.parameters()), retain_graph=True,
                             allow_unused=True)
    return torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None))


def make_data(n_f=4000, n_b=400, seed=0):
    torch.manual_seed(seed)
    xf = torch.rand(n_f, 1) * 2 - 1
    yf = torch.rand(n_f, 1) * 2 - 1
    s = torch.rand(n_b, 1) * 2 - 1
    o = torch.ones(n_b // 4, 1)
    xb = torch.cat([s[:n_b // 4], s[n_b // 4:n_b // 2], -o, o])
    yb = torch.cat([-o, o, s[n_b // 2:3 * n_b // 4], s[3 * n_b // 4:]])
    return xf, yf, xb, yb


def l2_error(model, n=200):
    g = torch.linspace(-1, 1, n)
    X, Y = torch.meshgrid(g, g, indexing="ij")
    x, y = X.reshape(-1, 1), Y.reshape(-1, 1)
    with torch.no_grad():
        pred = model(x, y)
    truth = exact(x, y)
    return (torch.norm(pred - truth) / torch.norm(truth)).item()


def train(seed, lam=None):
    """lam=None -> gradient-based (learning-rate annealing); else constant weight.
    The seed sets the network initialisation; the point sets are the notebook's (seed 0)."""
    torch.manual_seed(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xf, yf, xb, yb = make_data(seed=0)
    adaptive = lam is None
    w = 1.0 if adaptive else lam
    for it in range(1, ITERS + 1):
        l_f = pde_residual(model, xf, yf).pow(2).mean()
        l_b = model(xb, yb).pow(2).mean()
        if adaptive and it % 100 == 0:
            gf, gb = grad_norm(l_f, model), grad_norm(l_b, model)
            if gb > 0:
                w = ALPHA * w + (1 - ALPHA) * (gf / gb).item()
        opt.zero_grad()
        (l_f + w * l_b).backward()
        opt.step()
    return {"pde_loss": l_f.item(), "bc_loss": l_b.item(), "lambda_final": w,
            "rel_l2": l2_error(model)}


METHODS = {"equal_lambda_1": 1.0, "tuned_lambda_1000": LAM_TUNED, "gradient_based": None}
PARTS = OUT.parent / "parts"


def run_seeds(seeds):
    """Train the three weightings for each seed; returns {method: {seed: result}}."""
    runs = {m: {} for m in METHODS}
    t0 = time.time()
    for seed in seeds:
        for m, lam in METHODS.items():
            r = train(seed, lam)
            runs[m][str(seed)] = r
            print(f"seed {seed}  {m:18s}  rel L2 {r['rel_l2']:.4f}  "
                  f"pde {r['pde_loss']:.4f}  lambda {r['lambda_final']:.0f}  "
                  f"{time.time() - t0:5.0f}s", flush=True)
    return runs, time.time() - t0


def main():
    # `--seed N` trains one seed and writes results/parts/seed_N.json (run the five in
    # parallel, one process each); `--merge` combines the parts; no argument does it all.
    args = sys.argv[1:]
    if args[:1] == ["--seed"]:
        seed = int(args[1])
        runs, wall = run_seeds([seed])
        PARTS.mkdir(parents=True, exist_ok=True)
        (PARTS / f"seed_{seed}.json").write_text(json.dumps({"runs": runs, "wall_time_s": wall}))
        return 0
    if args[:1] == ["--merge"]:
        runs, wall = {m: {} for m in METHODS}, 0.0
        for seed in SEEDS:
            part = json.loads((PARTS / f"seed_{seed}.json").read_text())
            for m in METHODS:
                runs[m].update(part["runs"][m])
            wall = max(wall, part["wall_time_s"])
    else:
        runs, wall = run_seeds(SEEDS)

    summary = {}
    for m in METHODS:
        e = [runs[m][str(s)]["rel_l2"] for s in SEEDS]
        summary[m] = {"mean": statistics.mean(e), "sd_sample": statistics.stdev(e),
                      "sd_population": statistics.pstdev(e), "min": min(e), "max": max(e),
                      "per_seed": e}

    def paired(a, b):
        d = [runs[a][str(s)]["rel_l2"] - runs[b][str(s)]["rel_l2"] for s in SEEDS]
        return {"diff_mean": statistics.mean(d), "diff_sd": statistics.stdev(d),
                "wins_for_first": sum(x < 0 for x in d), "per_seed": d}

    out = {
        "problem": "2D Helmholtz, manufactured solution sin(pi x) sin(4 pi y), k=1",
        "setup": {"iters": ITERS, "seeds": list(SEEDS), "network": "MLP 4x64 tanh",
                  "optimiser": "Adam lr 1e-3", "n_f": 4000, "n_b": 400,
                  "points_seed": 0, "tuned_lambda": LAM_TUNED, "annealing_alpha": ALPHA,
                  "torch": torch.__version__, "device": "cpu"},
        "error_metric": "relative L2 on a 200x200 grid against the exact solution",
        "summary": summary,
        "paired": {"tuned_vs_gradient_based": paired("tuned_lambda_1000", "gradient_based"),
                   "tuned_vs_equal": paired("tuned_lambda_1000", "equal_lambda_1"),
                   "gradient_based_vs_equal": paired("gradient_based", "equal_lambda_1")},
        "runs": runs,
        "wall_time_s": wall,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    for m, s in summary.items():
        print(f"{m:18s} mean {s['mean']:.4f}  sd {s['sd_sample']:.4f}  "
              f"min {s['min']:.4f}  max {s['max']:.4f}")
    p = out["paired"]["tuned_vs_gradient_based"]
    print(f"tuned constant beats gradient-based on {p['wins_for_first']}/{len(SEEDS)} seeds, "
          f"mean diff {p['diff_mean']:+.4f}")


if __name__ == "__main__":
    sys.exit(main())
