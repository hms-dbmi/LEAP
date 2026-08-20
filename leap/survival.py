import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class SurvivalNN(torch.nn.Module):
    """MLP mapping a slide embedding to a scalar log-hazard.

    LeakyReLU activations with dropout after each hidden layer; the output layer is linear.
    """

    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                torch.nn.Linear(in_dim, h_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=dropout),
            ]
            in_dim = h_dim
        layers.append(torch.nn.Linear(in_dim, 1))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def cox_ph_loss(log_risk, time, event):
    """Negative Cox partial log-likelihood (Breslow ties), averaged over observed events.

    Sorting by time descending makes each subject's risk set the leading prefix, so the
    normalising term is a cumulative log-sum-exp. Risk scores are standardised within the
    batch for numerical stability. A batch containing no event has an undefined partial
    likelihood and yields NaN.
    """
    order = torch.argsort(time, descending=True)
    log_risk = log_risk[order]
    event = event[order]

    log_risk = (log_risk - log_risk.mean()) / (log_risk.std() + 1e-8)
    log_cumsum = torch.logcumsumexp(log_risk, dim=0)
    losses = (log_risk - log_cumsum) * event
    return -torch.sum(losses) / torch.sum(event)


def train_survival_model(
    features,
    targets,
    hidden_dims=(128, 64),
    lr=1e-4,
    epochs=100,
    batch_size=64,
    dropout=0.2,
    l2=1e-3,
    patience=10,
    grad_clip=1.0,
    device="cpu",
    verbose=True,
):
    """Fit a SurvivalNN on slide embeddings with the Cox partial likelihood.

    Parameters
    ----------
    features: (N, D) array or tensor of slide embeddings.
    targets: dict with 'Time' and 'Event' arrays of length N.
    l2: ridge penalty applied to all parameters, added to the loss.
    patience: epochs without an improvement in training loss before stopping early.

    Returns the trained model in eval mode.
    """
    X = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    time = torch.as_tensor(np.asarray(targets["Time"]), dtype=torch.float32, device=device)
    event = torch.as_tensor(np.asarray(targets["Event"]), dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(X, time, event), batch_size=batch_size, shuffle=True)
    model = SurvivalNN(X.shape[1], list(hidden_dims), dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_loss, no_improve, epoch_loss = float("inf"), 0, float("nan")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_time, batch_event in loader:
            optimizer.zero_grad()
            risk = model(batch_X).squeeze(-1)
            loss = cox_ph_loss(risk, batch_time, batch_event)
            loss = loss + l2 * torch.sum(
                torch.square(torch.cat([p.view(-1) for p in model.parameters()]))
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step(epoch_loss)
        if epoch_loss < best_loss:
            best_loss, no_improve = epoch_loss, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    survival head: early stop at epoch {epoch + 1}/{epochs}")
                break

    if verbose:
        print(f"    survival head trained, final epoch loss {epoch_loss:.4f}")
    return model.eval()


@torch.no_grad()
def risk_scores(model, features, device):
    """Standardised log-hazard, one value per row of `features`."""
    X = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    risk = model(X).squeeze(-1).cpu().numpy()
    return (risk - risk.mean()) / (risk.std() + 1e-12)


@torch.no_grad()
def penultimate_activations(model, features, device):
    """Activations of the survival head's last hidden layer, (N, hidden_dims[-1]).

    Replays every layer except the final Linear, so it follows `hidden_dims` without
    hard-coding a size.
    """
    h = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    for layer in list(model.model.children())[:-1]:
        h = layer(h)
    return h.cpu().numpy()
