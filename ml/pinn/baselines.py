"""Baselines required by ARCHITECTURE.md: a plain MLP and a SINDy-style
sparse regression.  Uses PyTorch for the MLP.
"""

import random
from itertools import combinations_with_replacement
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    pred = np.asarray(pred)
    target = np.asarray(target)
    return np.mean((pred - target)**2)

class MLP(nn.Module):
    """Small fully-connected MLP trained with PyTorch.
    Maintains a list-of-lists interface for compatibility.
    """
    def __init__(self, dims, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            out = self.net(x_tensor)
        return out.numpy().tolist()

    def fit(self, X, Y, epochs=200, lr=1e-2, verbose=True,
              X_val=None, Y_val=None, patience=None, weight_decay=0.0):
        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)
        
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
        else:
            X_val_t, Y_val_t = None, None

        for epoch in range(1, epochs + 1):
            self.train()
            optimizer.zero_grad()
            pred = self.net(X_t)
            loss = criterion(pred, Y_t)
            loss.backward()
            optimizer.step()
            
            if X_val_t is not None:
                self.eval()
                with torch.no_grad():
                    v_pred = self.net(X_val_t)
                    v_loss = criterion(v_pred, Y_val_t)
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        best_state = self.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience is not None and patience_counter >= patience:
                            break
            
            if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
                print(f"[mlp] epoch {epoch:4d}  mse = {loss.item():.6e}")
        
        if best_state is not None:
            self.load_state_dict(best_state)
        return self

# --- SINDy remains numpy-based as it is not a gradient-descent problem ---

def linear_regression(X, Y):
    n = len(X)
    d = len(X[0])
    m = len(Y[0])
    ridge = 1e-8
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
    A = [G[a][:] + [B[a][k] for k in range(m)] for a in range(d + 1)]
    for col in range(d + 1):
        piv = max(range(col, d + 1), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        if abs(pv) < 1e-30: pv = 1e-30
        for j in range(col, d + 1 + m):
            A[col][j] /= pv
        for r in range(d + 1):
            if r == col: continue
            factor = A[r][col]
            for j in range(col, d + 1 + m):
                A[r][j] -= factor * A[col][j]
    W = [[A[a][d + 1 + k] for k in range(m)] for a in range(d + 1)]
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

def poly_terms(n_features, degree=2):
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
    values = []
    for exponents, _name in terms:
        value = 1.0
        for i, e in enumerate(exponents):
            if e:
                value *= row[i] ** e
        values.append(value)
    return values

def _poly_features(X, degree=2):
    if not X:
        raise ValueError("X must be non-empty")
    n = len(X)
    d = len(X[0])
    terms = poly_terms(d, degree)
    feats = [poly_transform(X[ni], terms) for ni in range(n)]
    return feats, [name for _exponents, name in terms]

def sindy_predict(X, Xi, degree=2):
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
    if not X:
        raise ValueError("X must be non-empty")
    F, names = _poly_features(X, degree)
    m = len(F[0])
    n = len(F)
    FtF = [[sum(F[ni][a] * F[ni][b] for ni in range(n))
            for b in range(m)] for a in range(m)]
    out_dim = len(Y[0])
    Xi = []
    for j in range(out_dim):
        rhs = [sum(F[ni][a] * Y[ni][j] for ni in range(n)) for a in range(m)]
        Xi.append(_solve([row[:] for row in FtF], rhs))
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
