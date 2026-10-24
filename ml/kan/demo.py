"""Tiny end-to-end demo of the KAN, MLP, and SINDy baselines.

Run from this directory:
    python3 demo.py
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kan import KAN, export_results, mse, r2_score, train
from baselines import MLP, sindy, sindy_predict

random.seed(0)


# Synthetic residual: a nonlinear correction that a KAN should recover.
# y = 0.5*x0^2 - 0.3*sin(x1) + 0.1*x0*x1
def f(x0, x1):
    return 0.5 * x0 * x0 - 0.3 * math.sin(x1) + 0.1 * x0 * x1


N = 200
X = [[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)] for _ in range(N)]
Y = [[f(x[0], x[1])] for x in X]

print("=== KAN ===")
kan = KAN([2, 4, 1], num_basis=8, degree=3,
          grid_range=(-2.0, 2.0), standardize=True, l1=1e-4)
train(kan, X, Y, epochs=150, lr=2e-1, verbose=True)

kan_mse = mse(kan.forward(X), Y)
kan_r2 = r2_score(kan.forward(X), Y)
print(f"KAN train mse = {kan_mse:.6e}  r2 = {kan_r2:.6f}")

def _count_mask(group):
    kept = total = 0
    for row in group:
        for value in row:
            if isinstance(value, list):
                kept += sum(1 for flag in value if flag)
                total += len(value)
            else:
                kept += 1 if value else 0
                total += 1
    return kept, total


# Pruning is destructive (zeroes small weights), but the zeroed weights can be
# relearned by training afterwards; no mask is enforced.
masks = kan.prune(1e-2)
kept = total = 0
for layer_mask in masks:
    for group in layer_mask:
        k, t = _count_mask(group)
        kept += k
        total += t
print(f"pruned {total - kept}/{total} edge parameters; retraining 20 epochs")
train(kan, X, Y, epochs=20, lr=1e-2, verbose=False)
kan_mse = mse(kan.forward(X), Y)
kan_r2 = r2_score(kan.forward(X), Y)
print(f"KAN post-prune mse = {kan_mse:.6e}  r2 = {kan_r2:.6f}")

edge0 = kan.symbolic_edge(0, 0, 0)
edge1 = kan.symbolic_edge(0, 1, 0)
print("KAN edge 0->0 coefficients:", edge0.coeffs, f"(r2={edge0.r2:.4f})")
print("KAN edge 1->0 coefficients:", edge1.coeffs, f"(r2={edge1.r2:.4f})")
print("(edge polynomials are in standardized input units)")

print("\n=== MLP ===")
mlp = MLP([2, 8, 1])
mlp.train(X, Y, epochs=200, lr=5e-1, verbose=True)
mlp_mse = mse(mlp.forward(X), Y)
mlp_r2 = r2_score(mlp.forward(X), Y)
print(f"MLP train mse = {mlp_mse:.6e}  r2 = {mlp_r2:.6f}")

print("\n=== SINDy ===")
Xi, names, exprs = sindy(X, Y, threshold=0.05, degree=3)
print("features:", names)
print("expression:", exprs[0])
sp = sindy_predict(X, Xi, degree=3)
sindy_mse = mse(sp, Y)
sindy_r2 = r2_score(sp, Y)
print(f"sindy mse = {sindy_mse:.6e}  r2 = {sindy_r2:.6f}")

results = {
    "n_samples": N,
    "target": "0.5*x0^2 - 0.3*sin(x1) + 0.1*x0*x1",
    "kan": {
        "dims": kan.dims,
        "num_basis": 8,
        "degree": 3,
        "grid_range": list(kan.grid_range),
        "standardize": kan.standardize,
        "l1": kan.l1,
        "num_parameters": kan.num_parameters(),
        "train_mse": kan_mse,
        "train_r2": kan_r2,
        "edges": [
            {"layer": 0, "i": 0, "j": 0,
             "coeffs": edge0.coeffs, "r2": edge0.r2, "rmse": edge0.rmse},
            {"layer": 0, "i": 1, "j": 0,
             "coeffs": edge1.coeffs, "r2": edge1.r2, "rmse": edge1.rmse},
        ],
    },
    "mlp": {"train_mse": mlp_mse, "train_r2": mlp_r2},
    "sindy": {
        "degree": 3,
        "features": names,
        "expression": exprs[0],
        "train_mse": sindy_mse,
        "train_r2": sindy_r2,
    },
}
out_path = export_results(Path(__file__).with_name("kan_result.json"), results)
print(f"\nwrote {out_path}")
