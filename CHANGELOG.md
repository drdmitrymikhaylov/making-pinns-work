# Changelog

## 2026-09-22

- Module 02's five-seed table is now produced, not just quoted: `scripts/02_five_seeds.py` re-runs the three weightings for seeds 1–5 and writes `results/02_five_seeds.json` (per-seed errors, sample sd, paired differences, final adaptive λ). The re-run reproduces the README table to four decimals. New README subsection "Checking the numbers": per-seed table; the tuned constant beats annealing on 4/5 seeds (loses on seed 2), three seeds agree within 0.004 and the mean gap comes from two seeds at 4–5 % error; annealing settles at λ ≈ 3800–5300 (4–5× the tuned 1000) and is no better for it; ranked by PDE loss the best of all 15 runs is a 22 %-error model. `tests/test_readme_numbers.py` (8 tests) pins the README to the results file and to the notebook outputs, including module 01's 0.08–0.4 % Cole–Hopf range. "Related" links point at the renamed repos.
