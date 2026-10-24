"""Baselines required by ARCHITECTURE.md: a plain MLP and a SINDy-style
sparse regression.  Uses numpy for fast vectorized training."""

import random
from itertools import combinations_with_replacement

import numpy as np

__all__ = [
    "MLP",
    "_poly_features",
    "mse",
    "poly_terms",
    "poly_transform",
    "sindy",
    "sindy_predict",
]


def mse(pred, target):
    n = len(pred)
    m = len(pred[0])
    s = 0.0
    for i in range(n):
        for j in range(m):
            d = pred[i][j] - target[i][j]
            s += d * d
    return s / (n * m)


class MLP:
    """Small fully-connected MLP trained with full-batch gradient descent.

    All weights/biases are stored as numpy arrays for speed.  ``forward``
    accepts lists of lists and returns a list of lists, keeping the same
    interface as the original pure-Python version.
    """

    def __init__(self, dims, seed: int = 0):
        self.dims = dims
        self.W = []
        self.b = []
        rng = np.random.default_rng(seed)
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            scale = np.sqrt(2.0 / fan_in)  # He init for ReLU
            w = rng.normal(0.0, scale, size=(dims[i], dims[i + 1]))
            b = np.zeros(dims[i + 1])
            self.W.append(w)
            self.b.append(b)
        self._cache = None

    # ------------------------------------------------------------------
    # forward / backward / update
    # ------------------------------------------------------------------
    def forward(self, X):
        X = np.asarray(X, dtype=np.float64)
        acts = [X]
        n_layers = len(self.W)
        for li in range(n_layers):
            prev = acts[-1]
            z = prev @ self.W[li] + self.b[li]
            if li < n_layers - 1:
                z = np.maximum(z, 0.0)  # ReLU
            acts.append(z)
        self._cache = acts
        return acts[-1].tolist()

    def backward(self, dY):
        acts = self._cache
        L = len(self.W)
        dY = np.asarray(dY, dtype=np.float64)
        grads_W = [None] * L
        grads_b = [None] * L
        dA = dY
        for li in range(L - 1, -1, -1):
            prev = acts[li]
            n = prev.shape[0]
            if li == L - 1:
                dZ = dA  # output layer: no activation derivative
            else:
                a = acts[li + 1]
                dZ = dA * (a > 0)  # ReLU derivative
            grads_W[li] = prev.T @ dZ  # dY already carries 1/(n*m)
            grads_b[li] = dZ.sum(axis=0)
            if li > 0:
                dA = dZ @ self.W[li].T
        self._grads = (grads_W, grads_b)

    def update(self, lr):
        grads_W, grads_b = self._grads
        for li in range(len(self.W)):
            self.W[li] -= lr * grads_W[li]
            self.b[li] -= lr * grads_b[li]

    # ------------------------------------------------------------------
    # state save / restore (for early stopping)
    # ------------------------------------------------------------------
    def _save_state(self):
        return {
            "W": [w.copy() for w in self.W],
            "b": [b.copy() for b in self.b],
        }

    def _load_state(self, state):
        self.W = [w.copy() for w in state["W"]]
        self.b = [b.copy() for b in state["b"]]

    def train(self, X, Y, epochs=200, lr=1e-2, verbose=True,
              X_val=None, Y_val=None, patience=None):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n = X.shape[0]
        m = Y.shape[1]

        best_state = None
        best_val = float("inf")
        patience_counter = 0

        Y_val_arr = None
        if Y_val is not None:
            Y_val_arr = np.asarray(Y_val, dtype=np.float64)

        for epoch in range(1, epochs + 1):
            pred = self.forward(X)
            pred_arr = np.asarray(pred, dtype=np.float64)
            diff = pred_arr - Y
            loss = float((diff * diff).sum() / (n * m))
            # d(MSE)/d(pred) = 2 * (pred - target) / (n * m)
            dY = (2.0 * diff / (n * m)).tolist()
            self.backward(dY)
            self.update(lr)

            if X_val is not None and Y_val_arr is not None:
                val_pred = self.forward(X_val)
                val_pred_arr = np.asarray(val_pred, dtype=np.float64)
                val_diff = val_pred_arr - Y_val_arr
                val_n = val_pred_arr.shape[0]
                val_loss = float((val_diff * val_diff).sum() / (val_n * m))
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = self._save_state()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience is not None and patience_counter >= patience:
                        if verbose:
                            print(f"[mlp] epoch {epoch:4d}  mse = {loss:.6e}  "
                                  f"(early stop, best val mse={best_val:.6e})")
                        break

            if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
                print(f"[mlp] epoch {epoch:4d}  mse = {loss:.6e}")

        if best_state is not None:
            self._load_state(best_state)

        return self


def poly_terms(n_features, degree=2):
    """All monomials in `n_features` variables with total degree <= `degree`.

    Returns a list of ``(exponents, name)`` pairs, ordered by total degree
    (then lexicographically), with the constant term last.  For example
    ``poly_terms(2, 2)`` yields ``x0, x1, x0^2, x0*x1, x1^2, 1``.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    terms = []
    for total in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), total):
            exponents = [0] * n_features
            for idx in combo:
                exponents[idx] += 1
            parts = []
            for i, e in enumerate(exponents):
                if e == 1:
                    parts.append(f"x{i}")
                elif e > 1:
                    parts.append(f"x{i}^{e}")
            terms.append((tuple(exponents), "*".join(parts)))
    terms.append((tuple([0] * n_features), "1"))
    return terms


def poly_transform(row, terms):
    """Evaluate a monomial library (from :func:`poly_terms`) on one row."""
    values = []
    for exponents, _name in terms:
        value = 1.0
        for i, e in enumerate(exponents):
            if e:
                value *= row[i] ** e
        values.append(value)
    return values


def _poly_features(X, degree=2):
    """Build a polynomial feature library up to `degree` (any degree >= 1)."""
    if not X:
        raise ValueError("X must be non-empty")
    n = len(X)
    d = len(X[0])
    terms = poly_terms(d, degree)
    feats = [poly_transform(X[ni], terms) for ni in range(n)]
    return feats, [name for _exponents, name in terms]


def sindy_predict(X, Xi, degree=2):
    """Evaluate a fitted SINDy model on X."""
    if not X:
        return []
    terms = poly_terms(len(X[0]), degree)
    out_dim = len(Xi)
    pred = []
    for row in X:
        vals = poly_transform(row, terms)
        pred.append([sum(Xi[j][a] * vals[a] for a in range(len(vals)))
                     for j in range(out_dim)])
    return pred


def _solve(A, b, ridge=1e-10):
    """Gauss-Jordan solve with a small ridge and a pivot guard."""
    n = len(A)
    if n == 0:
        return []
    scale = max(1.0, max(abs(A[i][i]) for i in range(n)))
    M = [A[i][:] for i in range(n)]
    for i in range(n):
        M[i][i] += ridge * scale
    M = [M[i] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-14:
            # Singular direction: set its coefficient to zero and move on.
            M[col] = [0.0] * (n + 1)
            M[col][col] = 1.0
            continue
        M[col], M[pivot] = M[pivot], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col:
                factor = M[r][col]
                if factor != 0.0:
                    M[r] = [M[r][t] - factor * M[col][t]
                            for t in range(n + 1)]
    return [M[i][n] for i in range(n)]


def _format_expression(coeffs, names):
    parts = []
    for c, name in zip(coeffs, names):
        if c != 0.0:
            parts.append(f"{c:+.4f}*{name}")
    return " ".join(parts).lstrip("+ ") if parts else "0"


def sindy(X, Y, threshold=0.05, degree=2):
    """Sparse identification via sequential thresholded least squares (STLSQ)."""
    if not X:
        raise ValueError("X must be non-empty")
    F, names = _poly_features(X, degree)
    m = len(F[0])
    n = len(F)

    # Normal equations: (F^T F) Xi = F^T Y.
    FtF = [[sum(F[ni][a] * F[ni][b] for ni in range(n))
            for b in range(m)] for a in range(m)]

    out_dim = len(Y[0])
    Xi = []
    for j in range(out_dim):
        rhs = [sum(F[ni][a] * Y[ni][j] for ni in range(n)) for a in range(m)]
        Xi.append(_solve([row[:] for row in FtF], rhs))

    # Thresholding rounds with re-solve on the surviving support.
    for _ in range(10):
        for j in range(out_dim):
            for a in range(m):
                if abs(Xi[j][a]) < threshold:
                    Xi[j][a] = 0.0
            support = [a for a in range(m) if Xi[j][a] != 0.0]
            if not support:
                continue
            A = [[F[ni][a] for a in support] for ni in range(n)]
            AtA = [[sum(A[ni][p] * A[ni][q] for ni in range(n))
                    for q in range(len(support))] for p in range(len(support))]
            Aty = [sum(A[ni][p] * Y[ni][j] for ni in range(n))
                   for p in range(len(support))]
            sol = _solve(AtA, Aty)
            new_xi = [0.0] * m
            for idx, a in enumerate(support):
                new_xi[a] = sol[idx]
            Xi[j] = new_xi

    expressions = [_format_expression(Xi[j], names) for j in range(out_dim)]
    return Xi, names, expressions


def linear_regression(X, Y):
    """Ordinary least squares: returns (weights, bias) for single-output Y.

    Supports multi-output by fitting each output independently. Returns a
    predict function via ``linear_predict``.
    """
    n = len(X)
    d = len(X[0])
    m = len(Y[0])
    # Add bias column.
    # Solve (X^T X + ridge) w = X^T y per output.
    # Use normal equations with small ridge for stability.
    ridge = 1e-8
    # Build Gram matrix (d+1) x (d+1).
    G = [[0.0] * (d + 1) for _ in range(d + 1)]
    B = [[0.0] * m for _ in range(d + 1)]
    for i in range(n):
        row = [1.0] + list(X[i])
        for a in range(d + 1):
            for b in range(d + 1):
                G[a][b] += row[a] * row[b]
            for k in range(m):
                B[a][k] += row[a] * Y[i][k]
    for a in range(d + 1):
        G[a][a] += ridge
    # Gaussian elimination.
    # Augment.
    A = [G[a][:] + [B[a][k] for k in range(m)] for a in range(d + 1)]
    # Solve for each k.
    # Eliminate.
    for col in range(d + 1):
        # pivot
        piv = max(range(col, d + 1), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        if abs(pv) < 1e-30:
            pv = 1e-30
        for j in range(col, d + 1 + m):
            A[col][j] /= pv
        for r in range(d + 1):
            if r == col:
                continue
            factor = A[r][col]
            for j in range(col, d + 1 + m):
                A[r][j] -= factor * A[col][j]
    # weights: A[a][d+1+k]
    W = [[A[a][d + 1 + k] for k in range(m)] for a in range(d + 1)]
    # W[0] is bias, W[1:] are feature weights.
    return W


def linear_predict(X, W):
    out = []
    for row in X:
        pred = [W[0][k] for k in range(len(W[0]))]
        for i in range(len(row)):
            for k in range(len(W[0])):
                pred[k] += W[i + 1][k] * row[i]
        out.append(pred)
    return out
