try:
    import torch
    from torch import nn, optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None
    optim = None
import copy


class PINN(nn.Module if HAS_TORCH else object):
    """
    Physics-Informed Neural Network for orbital residual correction.
    The model predicts the position residuals delta_r = f(t).
    The physics constraint is that the derivative of delta_r must match delta_v.
    """

    def __init__(self, n_features=1, out_dim=3, hidden_dims=None, lr=1e-3, l2=1e-4):
        if hidden_dims is None:
            hidden_dims = [32, 32]
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for the PINN model. Install with 'pip install .[gpu]' or 'uv sync --extra gpu'."
            )
        super().__init__()
        self.lr = lr
        self.l2 = l2

        layers = []
        curr_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(
                nn.Tanh()
            )  # Tanh is preferred for PINNs due to smooth derivatives
            curr_dim = h
        layers.append(nn.Linear(curr_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x, return_tensor=False):
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32)

        # We only use the no_grad block if return_tensor is False AND we are in eval mode
        if not return_tensor and not self.training:
            self.eval()
            with torch.no_grad():
                out = self.net(x)
                return out.detach().numpy()

        return self.net(x)

    def get_derivative(self, x):
        """Compute df/dt using autograd.
        Assumes x is a tensor of shape [batch, 1].
        """
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.clone().detach().requires_grad_(True)
        y = self.forward(x, return_tensor=True)
        grads = []
        for i in range(y.shape[1]):
            grad = torch.autograd.grad(y[:, i].sum(), x, create_graph=True)[0]
            grads.append(grad)
        return torch.stack(grads, dim=1).squeeze(-1)


def train_pinn(
    model,
    X_fit,
    Y_pos,
    Y_vel,
    epochs=500,
    lr=1e-3,
    l2=1e-4,
    X_val=None,
    Y_pos_val=None,
    Y_vel_val=None,
    patience=20,
):
    """
    Train PINN with a composite loss: MSE(pos) + lambda * MSE(vel).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    # Convert to tensors
    X_fit = torch.tensor(X_fit, dtype=torch.float32)
    Y_pos = torch.tensor(Y_pos, dtype=torch.float32)
    Y_vel = torch.tensor(Y_vel, dtype=torch.float32)

    if X_val is not None:
        X_val = torch.tensor(X_val, dtype=torch.float32)
        Y_pos_val = torch.tensor(Y_pos_val, dtype=torch.float32)
        Y_vel_val = torch.tensor(Y_vel_val, dtype=torch.float32)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 1. Position loss
        pred_pos = model(X_fit, return_tensor=True)
        loss_pos = criterion(pred_pos, Y_pos)

        # 2. Velocity loss (Physics: v = dr/dt)
        pred_vel = model.get_derivative(X_fit)
        loss_vel = criterion(pred_vel, Y_vel)

        total_loss = loss_pos + loss_vel
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"[pinn] epoch {epoch:4d}  loss = {total_loss.item():.6e} (pos={loss_pos.item():.6e}, vel={loss_vel.item():.6e})"
            )

        # Validation
        if X_val is not None:
            model.eval()
            v_pos = model(X_val, return_tensor=True)
            v_vel = model.get_derivative(X_val)

            with torch.no_grad():
                val_loss = criterion(v_pos, Y_pos_val) + criterion(v_vel, Y_vel_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience is not None and patience_counter >= patience:
                        break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
