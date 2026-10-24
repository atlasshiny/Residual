"""A small, dependency-free Kolmogorov-Arnold Network (KAN).

A KAN replaces the fixed activation of an MLP with a learnable univariate
function on every edge.  Each edge is a B-spline plus a linear residual, so
the whole model is a sum of 1-D functions that can later be read off as a
closed-form expression.

This implementation is intentionally tiny: pure Python, no autograd, analytic
gradients.  It is meant for the residual-correction use case in
ARCHITECTURE.md section 5, not for large-scale training.
"""

import json
import math
import random

__all__ = [
    "KAN",
    "KANLayer",
    "PolynomialFit",
    "export_results",
    "mse",
    "r2_score",
    "train",
]


def _linspace(a, b, n):
    if n == 1:
        return [0.5 * (a + b)]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _bspline_knots(a, b, num_basis, degree):
    """Open uniform (clamped) knot vector for `num_basis` B-splines."""
    if num_basis < degree + 1:
        raise ValueError("num_basis must be >= degree + 1")
    n = num_basis
    k = degree
    knots = [a] * (k + 1)
    interior = n - k - 1
    if interior > 0:
        for i in range(1, interior + 1):
            knots.append(a + (b - a) * i / (interior + 1))
    knots += [b] * (k + 1)
    return knots


def _bspline_basis(x, knots, degree):
    """Cox-de Boor recursion. Returns a list of `num_basis` values.

    The knot vector is clamped, so `knots[0] == knots[1] == ...` at the left
    edge and `knots[-1] == knots[-2] == ...` at the right edge.  A plain
    half-open-interval implementation drops the final basis at the right
    boundary (`N_{n-1}(b) == 0`); to keep a partition of unity we evaluate the
    limit from inside the domain, i.e. treat the resulting 0/0 ratios as 1.
    """
    lo, hi = knots[0], knots[-1]
    if x < lo:
        x = lo
    elif x > hi:
        x = hi

    m = len(knots)

    # Degree 0: indicator of [knots[i], knots[i+1]).  At the right edge the
    # last span is degenerate ([b, b]); seed it explicitly so the recursion
    # below can carry it to N_{n-1}(b) = 1.
    basis = [0.0] * (m - 1)
    for i in range(m - 1):
        if knots[i] <= x < knots[i + 1]:
            basis[i] = 1.0
    if x == hi:
        basis[m - 2] = 1.0

    for d in range(1, degree + 1):
        new_len = m - d - 1
        new_basis = [0.0] * new_len
        for i in range(new_len):
            left = 0.0
            den = knots[i + d] - knots[i]
            if den != 0.0:
                left = (x - knots[i]) / den * basis[i]
            elif x == knots[i]:
                # 0/0 limit convention (only reachable on a clamped edge).
                left = basis[i]

            right = 0.0
            den = knots[i + d + 1] - knots[i + 1]
            if den != 0.0:
                right = (knots[i + d + 1] - x) / den * basis[i + 1]
            elif x == knots[i + d + 1]:
                # 0/0 limit convention: carries N_{n-1} to 1 at x == b.
                right = basis[i + 1]

            new_basis[i] = left + right
        basis = new_basis

    # len(basis) == len(knots) - degree - 1 == num_basis
    return basis


def _bspline_deriv(x, knots, degree, eps=1e-5):
    """Finite-difference derivative of the basis vector.

    Central differences are used inside the domain; at (or immediately next
    to) an edge a one-sided difference is used instead so that clamping does
    not silently produce a zero derivative.
    """
    lo, hi = knots[0], knots[-1]
    if x < lo:
        x = lo
    elif x > hi:
        x = hi

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    h = eps * max(1.0, hi - lo)

    if x - h < lo:
        b0 = _bspline_basis(lo, knots, degree)
        b1 = _bspline_basis(lo + h, knots, degree)
        return [(b1[k] - b0[k]) / h for k in range(len(b0))]

    if x + h > hi:
        b0 = _bspline_basis(hi - h, knots, degree)
        b1 = _bspline_basis(hi, knots, degree)
        return [(b1[k] - b0[k]) / h for k in range(len(b1))]

    left = _bspline_basis(x - h, knots, degree)
    right = _bspline_basis(x + h, knots, degree)
    return [(right[k] - left[k]) / (2.0 * h) for k in range(len(left))]


class KANLayer:
    def __init__(self, in_dim, out_dim, num_basis=8, degree=3,
                 grid_range=(-1.0, 1.0)):
        lo, hi = float(grid_range[0]), float(grid_range[1])
        if not hi > lo:
            raise ValueError("grid_range must satisfy high > low")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis
        self.degree = degree
        self.grid_range = (lo, hi)
        self.knots = _bspline_knots(lo, hi, num_basis, degree)
        # coeff[i][j][k], linear[i][j], bias[j]
        self.coeff = [[[random.uniform(-0.05, 0.05) for _ in range(num_basis)]
                       for _ in range(out_dim)] for _ in range(in_dim)]
        self.linear = [[random.uniform(-0.05, 0.05) for _ in range(out_dim)]
                       for _ in range(in_dim)]
        self.bias = [0.0 for _ in range(out_dim)]
        self.mask = None
        self._cache = None

    def forward(self, X):
        n = len(X)
        basis_cache = []  # basis_cache[n][i] = list of K values
        for sample in X:
            row = []
            for i in range(self.in_dim):
                row.append(_bspline_basis(sample[i], self.knots, self.degree))
            basis_cache.append(row)
        self._cache = (X, basis_cache)

        Y = [[0.0 for _ in range(self.out_dim)] for _ in range(n)]
        for ni in range(n):
            for j in range(self.out_dim):
                y = self.bias[j]
                for i in range(self.in_dim):
                    xi = X[ni][i]
                    y += self.linear[i][j] * xi
                    for k in range(self.num_basis):
                        y += self.coeff[i][j][k] * basis_cache[ni][i][k]
                Y[ni][j] = y
        return Y

    def backward(self, dY):
        X, basis_cache = self._cache
        n = len(X)
        grad_coeff = [[[0.0 for _ in range(self.num_basis)]
                       for _ in range(self.out_dim)] for _ in range(self.in_dim)]
        grad_linear = [[0.0 for _ in range(self.out_dim)]
                       for _ in range(self.in_dim)]
        grad_bias = [0.0 for _ in range(self.out_dim)]
        grad_X = [[0.0 for _ in range(self.in_dim)] for _ in range(n)]

        for ni in range(n):
            # The basis derivative depends only on (sample, input), so
            # compute it once per input instead of once per output j.
            derivs = [_bspline_deriv(X[ni][i], self.knots, self.degree)
                      for i in range(self.in_dim)]
            for j in range(self.out_dim):
                dy = dY[ni][j]
                grad_bias[j] += dy
                for i in range(self.in_dim):
                    xi = X[ni][i]
                    grad_linear[i][j] += dy * xi
                    for k in range(self.num_basis):
                        grad_coeff[i][j][k] += dy * basis_cache[ni][i][k]
                        grad_X[ni][i] += dy * self.coeff[i][j][k] * derivs[i][k]
                    grad_X[ni][i] += dy * self.linear[i][j]

        self._grads = (grad_coeff, grad_linear, grad_bias)
        return grad_X

    def update(self, lr, l1=0.0):
        grad_coeff, grad_linear, grad_bias = self._grads
        for i in range(self.in_dim):
            for j in range(self.out_dim):
                for k in range(self.num_basis):
                    c = self.coeff[i][j][k]
                    g = grad_coeff[i][j][k]
                    if l1:
                        g += l1 * (1.0 if c > 0.0 else (-1.0 if c < 0.0 else 0.0))
                    self.coeff[i][j][k] = c - lr * g
                w = self.linear[i][j]
                g = grad_linear[i][j]
                if l1:
                    g += l1 * (1.0 if w > 0.0 else (-1.0 if w < 0.0 else 0.0))
                self.linear[i][j] = w - lr * g
        for j in range(self.out_dim):
            self.bias[j] -= lr * grad_bias[j]

    def prune(self, threshold=1e-2, inplace=True):
        """Magnitude pruning.

        Returns ``(coeff_mask, linear_mask)`` where True means "kept".  With
        ``inplace=False`` the masks are computed without touching the weights.
        Pruning is destructive; calling :meth:`train` (or the module-level
        ``train``) afterwards relearns the remaining/zeroed weights, since no
        mask is enforced during training.
        """
        coeff_mask = [[[abs(self.coeff[i][j][k]) >= threshold
                        for k in range(self.num_basis)]
                       for j in range(self.out_dim)] for i in range(self.in_dim)]
        linear_mask = [[abs(self.linear[i][j]) >= threshold
                        for j in range(self.out_dim)] for i in range(self.in_dim)]
        self.mask = (coeff_mask, linear_mask)
        if inplace:
            for i in range(self.in_dim):
                for j in range(self.out_dim):
                    for k in range(self.num_basis):
                        if not coeff_mask[i][j][k]:
                            self.coeff[i][j][k] = 0.0
                    if not linear_mask[i][j]:
                        self.linear[i][j] = 0.0
        return self.mask

    def edge(self, i, j, x):
        """Evaluate the univariate edge function i -> j at x."""
        y = self.linear[i][j] * x
        basis = _bspline_basis(x, self.knots, self.degree)
        for k in range(self.num_basis):
            y += self.coeff[i][j][k] * basis[k]
        return y


class KAN:
    def __init__(self, dims, num_basis=8, degree=3, grid_range=(-2.0, 2.0),
                 standardize=True, l1=0.0):
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output sizes")
        lo, hi = float(grid_range[0]), float(grid_range[1])
        if not hi > lo:
            raise ValueError("grid_range must satisfy high > low")
        self.dims = list(dims)
        self.grid_range = (lo, hi)
        self.standardize = bool(standardize)
        self.l1 = float(l1)
        self.x_mean = None
        self.x_std = None
        self.layers = [KANLayer(dims[i], dims[i + 1], num_basis, degree,
                                grid_range=self.grid_range)
                       for i in range(len(dims) - 1)]

    # ------------------------------------------------------------------ scaler
    def fit_scaler(self, X):
        """Compute per-feature z-score statistics from X."""
        if not X:
            raise ValueError("cannot fit a scaler on an empty batch")
        d = len(X[0])
        n = len(X)
        mean = [0.0] * d
        for row in X:
            if len(row) != d:
                raise ValueError("ragged input batch")
            for i in range(d):
                mean[i] += row[i]
        mean = [v / n for v in mean]
        var = [0.0] * d
        for row in X:
            for i in range(d):
                diff = row[i] - mean[i]
                var[i] += diff * diff
        std = [math.sqrt(v / n) for v in var]
        # Constant features: leave them centered at zero.
        std = [s if s > 1e-12 else 1.0 for s in std]
        self.x_mean, self.x_std = mean, std
        return mean, std

    def transform(self, X):
        """Apply the fitted z-score transform (no-op if disabled/unfitted)."""
        if not self.standardize or self.x_mean is None:
            return [list(row) for row in X]
        return [[(row[i] - self.x_mean[i]) / self.x_std[i]
                 for i in range(len(row))] for row in X]

    def _prepare(self, X):
        if not self.standardize:
            return [list(row) for row in X]
        if not X:
            return []
        if self.x_mean is None:
            self.fit_scaler(X)
        return self.transform(X)

    # ------------------------------------------------------------------ model
    def forward(self, X):
        Xs = self._prepare(X)
        for layer in self.layers:
            Xs = layer.forward(Xs)
        return Xs

    def backward(self, dY):
        for layer in reversed(self.layers):
            dY = layer.backward(dY)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr, self.l1)

    def prune(self, threshold=1e-2, inplace=True):
        return [layer.prune(threshold, inplace=inplace) for layer in self.layers]

    def num_parameters(self):
        total = 0
        for layer in self.layers:
            total += layer.in_dim * (layer.out_dim * layer.num_basis
                                     + layer.out_dim)
            total += layer.out_dim
        return total

    def symbolic_edge(self, layer_idx, i, j, samples=64, poly_degree=3):
        """Fit edge (i -> j) of a layer to a polynomial.

        The fit is performed over the layer's configured ``grid_range``.  When
        input standardization is enabled these are standardized units (the
        coordinates the spline actually sees), not raw feature units.

        Returns a :class:`PolynomialFit` with ``coeffs`` (ascending powers),
        ``r2`` and ``rmse``.
        """
        layer = self.layers[layer_idx]
        lo, hi = layer.grid_range
        xs = _linspace(lo, hi, samples)
        ys = [layer.edge(i, j, x) for x in xs]

        p = poly_degree
        n = len(xs)
        # Normal equations: S[k] = sum_n x_n^k, T[a] = sum_n x_n^a y_n.
        S = [0.0] * (2 * p + 1)
        T = [0.0] * (p + 1)
        powers = [[1.0] * (2 * p + 1) for _ in range(n)]
        for ni, x in enumerate(xs):
            pw = powers[ni]
            for k in range(1, 2 * p + 1):
                pw[k] = pw[k - 1] * x
            for k in range(2 * p + 1):
                S[k] += pw[k]
            for a in range(p + 1):
                T[a] += pw[a] * ys[ni]

        A = [[S[a + b] for b in range(p + 1)] for a in range(p + 1)]
        coeffs = _solve_linear(A, T)

        ss_res = 0.0
        mean_y = sum(ys) / n
        ss_tot = 0.0
        for ni, x in enumerate(xs):
            yhat = 0.0
            for a, c in enumerate(coeffs):
                yhat += c * powers[ni][a]
            d = ys[ni] - yhat
            ss_res += d * d
            e = ys[ni] - mean_y
            ss_tot += e * e
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else (1.0 if ss_res == 0.0 else 0.0)
        rmse = math.sqrt(ss_res / n)
        return PolynomialFit(coeffs, r2, rmse)


class PolynomialFit:
    """Result of :meth:`KAN.symbolic_edge`.

    Behaves like the coefficient list for backwards compatibility, while also
    exposing fit quality (``r2`` / ``rmse``).
    """

    __slots__ = ("coeffs", "r2", "rmse")

    def __init__(self, coeffs, r2, rmse):
        self.coeffs = list(coeffs)
        self.r2 = r2
        self.rmse = rmse

    def __iter__(self):
        return iter(self.coeffs)

    def __getitem__(self, idx):
        return self.coeffs[idx]

    def __len__(self):
        return len(self.coeffs)

    def __repr__(self):
        return (f"PolynomialFit(coeffs={self.coeffs!r}, "
                f"r2={self.r2:.6f}, rmse={self.rmse:.3e})")


def _solve_linear(A, b, ridge=1e-10):
    """Small dense Gauss-Jordan solve with partial pivoting.

    A tiny relative ridge and a pivot guard keep the symbolic fit finite even
    on a degenerate edge (e.g. a constant function, or collinear samples).
    """
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
            # Singular direction: pin this coefficient to zero.
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


def mse(pred, target):
    n = len(pred)
    m = len(pred[0])
    s = 0.0
    for i in range(n):
        for j in range(m):
            d = pred[i][j] - target[i][j]
            s += d * d
    return s / (n * m)


def r2_score(pred, target):
    n = len(pred)
    m = len(pred[0])
    means = [sum(target[i][j] for i in range(n)) / n for j in range(m)]
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        for j in range(m):
            d = pred[i][j] - target[i][j]
            e = target[i][j] - means[j]
            ss_res += d * d
            ss_tot += e * e
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _kan_state_copy(model):
    """Deep copy of KAN trainable parameters (for early stopping restore)."""
    return {
        "x_mean": list(model.x_mean) if model.x_mean is not None else None,
        "x_std": list(model.x_std) if model.x_std is not None else None,
        "layers": [
            {
                "coeff": [[list(c) for c in layer.coeff[i]]
                          for i in range(layer.in_dim)],
                "linear": [[v for v in row] for row in layer.linear],
                "bias": list(layer.bias),
            }
            for layer in model.layers
        ],
    }


def _kan_state_restore(model, state):
    """Restore KAN trainable parameters from a saved state dict."""
    model.x_mean = list(state["x_mean"]) if state["x_mean"] is not None else None
    model.x_std = list(state["x_std"]) if state["x_std"] is not None else None
    for layer, ls in zip(model.layers, state["layers"]):
        for i in range(layer.in_dim):
            for j in range(layer.out_dim):
                for k in range(layer.num_basis):
                    layer.coeff[i][j][k] = ls["coeff"][i][j][k]
                layer.linear[i][j] = ls["linear"][i][j]
        for j in range(layer.out_dim):
            layer.bias[j] = ls["bias"][j]


def train(model, X, Y, epochs=200, lr=1e-2, verbose=True,
          X_val=None, Y_val=None, patience=None):
    """Full-batch gradient descent on MSE with optional early stopping.

    If ``X_val``/``Y_val`` are provided and ``patience`` is set, training
    stops when validation loss hasn't improved for ``patience`` epochs, and
    the best (lowest validation loss) model state is restored.
    """
    n = len(X)
    m = len(Y[0])

    best_state = None
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        pred = model.forward(X)
        loss = mse(pred, Y)
        # d(mean squared error)/d(pred) = 2 * (pred - target) / (n * m)
        dY = [[2.0 * (pred[i][j] - Y[i][j]) / (n * m)
               for j in range(m)] for i in range(n)]
        model.backward(dY)
        model.update(lr)

        if X_val is not None and Y_val is not None:
            val_pred = model.forward(X_val)
            val_loss = mse(val_pred, Y_val)
            if val_loss < best_val:
                best_val = val_loss
                best_state = _kan_state_copy(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience is not None and patience_counter >= patience:
                    if verbose:
                        print(f"epoch {epoch:4d}  mse = {loss:.6e}  "
                              f"(early stop, best val mse={best_val:.6e})")
                    break

        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            extra = ""
            if X_val is not None and Y_val is not None:
                extra = f"  val_mse={mse(model.forward(X_val), Y_val):.6e}"
            print(f"epoch {epoch:4d}  mse = {loss:.6e}{extra}")

    if best_state is not None:
        _kan_state_restore(model, best_state)

    return model


def export_results(path, results):
    """Write a JSON-serialisable results dict to `path`."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    return str(path)
