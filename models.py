"""Models for the comparison table: the baseline, the Assignment 2 forest (as a regressor), and
the feed-forward neural network.

Every model has the same two methods, so the pipeline can treat them alike:
    fit(X, y)  -> self        X: rows x features, y: forward returns (decimals)
    predict(X) -> 1-D array   one expected forward return per row
The network's fit() also takes the validation block (X_val, y_val) for early stopping.

PyTorch is only imported when the network is used, so the baseline and the forest (and the rest of
the app) run on a machine without it.

"Up" means a strictly positive forward return. Direction is +1 for up and -1 otherwise.

Run `python models.py [prices.csv]` for a quick look. It scores VALIDATION data only: the test
block is touched once, at the very end, by the final evaluation.
"""
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from features import FEATURES


# ------------------------------------------------------------------------ helpers
def to_x(df, cols=None):
    """Feature matrix (float numpy array) from a frame. Use this for rows that have no label yet."""
    return df[list(FEATURES if cols is None else cols)].to_numpy(dtype=float)


def to_xy(df, cols=None):
    """(X, y): features and the forward-return label. Pass the NORMALIZED frames for the network;
    trees do not care about scale, but giving every model the same inputs keeps the comparison fair."""
    return to_x(df, cols), df["label"].to_numpy(dtype=float)


def _fit_check(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError(f"X must be 2-D with one row per label (got X {X.shape}, y {y.shape}).")
    if len(y) == 0:
        raise ValueError("Cannot fit on zero rows.")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("X and y must not contain NaN or infinite values.")
    return X, y


# ----------------------------------------------------------------------- baseline
class MeanBaseline:
    """The number to beat: ignore the features and always predict the TRAINING mean return.

    Direction: always the majority direction of the training labels (the walkthrough's
    "always predict the majority class"). Nothing from validation or test is ever used."""

    name = "Baseline (training mean)"

    def fit(self, X, y):
        _, y = _fit_check(X, y)
        self.mean_ = float(y.mean())
        self.up_share_ = float((y > 0).mean())
        self.majority_up_ = self.up_share_ >= 0.5
        return self

    def _need_fit(self):
        if not hasattr(self, "mean_"):
            raise RuntimeError("Call fit() first.")

    def predict(self, X):
        self._need_fit()
        return np.full(len(X), self.mean_)

    def predict_direction(self, X):
        self._need_fit()
        return np.full(len(X), 1 if self.majority_up_ else -1)


# ------------------------------------------------------------------------- forest
class ForestRegressor:
    """The Assignment 2 random forest, switched from a yes/no classifier to a regressor.

    The hyperparameters are Assignment 2's defaults (40 trees, depth 6, 3 features per split,
    at least 25 rows per leaf, seed 7), so the comparison with the network is against the same
    forest you already built. Only the target changed: it now predicts the expected forward return
    instead of P(up). It never sees validation or test rows."""

    name = "Random forest (Assignment 2)"

    def __init__(self, n_trees=40, max_depth=6, mtry=3, min_leaf=25, seed=7):
        self.n_trees, self.max_depth, self.mtry = int(n_trees), int(max_depth), int(mtry)
        self.min_leaf, self.seed = int(min_leaf), int(seed)

    def fit(self, X, y):
        X, y = _fit_check(X, y)
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_trees, max_depth=self.max_depth,
            max_features=min(self.mtry, X.shape[1]), min_samples_leaf=self.min_leaf,
            bootstrap=True, random_state=self.seed, n_jobs=-1,
        ).fit(X, y)
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"Expected X with {self.n_features_} columns, got shape {X.shape}.")
        return self.model_.predict(X)

    def importances(self, cols=None):
        """How much each feature was used, as {name: share}. Shares add up to 1."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() first.")
        cols = list(FEATURES if cols is None else cols)
        return dict(zip(cols, (float(v) for v in self.model_.feature_importances_)))


# ------------------------------------------------------------------ neural network
def parse_hidden(sizes):
    """'32, 16' (or a list / tuple of ints) -> (32, 16). For the GUI's hidden-layer-sizes box."""
    parts = [p.strip() for p in sizes.replace(";", ",").split(",") if p.strip()] if isinstance(sizes, str) else list(sizes)
    try:
        out = tuple(int(str(p)) for p in parts)          # int("3.5") fails on purpose
    except ValueError:
        raise ValueError(f"Hidden layer sizes must be whole numbers like 32, 16 (got {sizes!r}).") from None
    if not out or any(s < 1 for s in out):
        raise ValueError(f"Hidden layer sizes must be one or more positive whole numbers (got {sizes!r}).")
    return out


def _torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError("The neural network needs PyTorch. Install it with: pip install torch") from e
    return torch


class NeuralNet:
    """Feed-forward neural network that predicts the expected forward return.

        inputs -> Linear(hidden[0]) -> ReLU -> Linear(hidden[1]) -> ReLU -> ... -> Linear(1)

    Defaults are the walkthrough's: hidden layers 32 and 16, ReLU, one LINEAR output (it predicts a
    return, not a yes/no), MSE loss, Adam at learning rate 0.001, early stopping on validation.

    Early stopping: after every epoch the network is scored on the validation block. Training
    stops once the validation loss has not improved for `patience` epochs, and the weights from
    the BEST epoch are restored (not the last one). The validation block is only ever scored, never
    trained on. That also means validation numbers for this model are slightly optimistic, because
    validation chose its best epoch; the honest comparison is on the test block.

    Train and validation loss are both the plain MSE over the whole block, measured after each
    epoch, so the two curves are directly comparable. Pass `on_epoch(epoch, train_loss, val_loss)`
    to watch training live (for the GUI plot); return True from it to stop early (a Cancel button).

    After fit(): history_ (dict of lists: epoch, train_loss, val_loss), best_epoch_, best_val_loss_,
    epochs_run_, stopped_early_, stop_reason_, model_, optimizer_.
    """

    name = "Neural network (ANN)"

    def __init__(self, hidden=(32, 16), lr=0.001, epochs=200, batch_size=64, patience=10,
                 seed=7, min_delta=0.0):
        self.hidden = parse_hidden(hidden)
        self.lr, self.min_delta = float(lr), float(min_delta)
        self.epochs, self.batch_size, self.patience, self.seed = int(epochs), int(batch_size), int(patience), int(seed)
        if not self.lr > 0:
            raise ValueError("Learning rate must be positive.")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("Epochs, batch size and patience must each be at least 1.")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative.")

    def fit(self, X, y, X_val=None, y_val=None, on_epoch=None):
        torch = _torch()
        X, y = _fit_check(X, y)
        have_val = X_val is not None
        if have_val != (y_val is not None):
            raise ValueError("Give both X_val and y_val, or neither.")
        if have_val:
            X_val, y_val = _fit_check(X_val, y_val)
            if X_val.shape[1] != X.shape[1]:
                raise ValueError(f"X_val has {X_val.shape[1]} columns but X has {X.shape[1]}.")

        nn = torch.nn
        Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)   # copies
        if have_val:
            Xv, yv = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

        widths = [X.shape[1], *self.hidden]
        with torch.random.fork_rng(devices=[]):              # seed the weights without touching the global RNG
            torch.manual_seed(self.seed)
            layers = []                                      # layers draw their starting weights when
            for a, b in zip(widths[:-1], widths[1:]):        # they are CREATED, so create them in here
                layers += [nn.Linear(a, b), nn.ReLU()]
            layers.append(nn.Linear(widths[-1], 1))          # linear output: no activation
            model = nn.Sequential(*layers)
        loss_fn = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        shuffler = torch.Generator().manual_seed(self.seed)

        hist = {"epoch": [], "train_loss": [], "val_loss": []}
        best_val, best_epoch, best_state, since_best = float("inf"), 0, None, 0
        reason = f"reached the maximum of {self.epochs} epochs"
        n = len(Xt)
        for epoch in range(1, self.epochs + 1):
            model.train()
            order = torch.randperm(n, generator=shuffler)
            for i in range(0, n, self.batch_size):
                idx = order[i:i + self.batch_size]
                opt.zero_grad()
                loss_fn(model(Xt[idx]).squeeze(1), yt[idx]).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                tr = loss_fn(model(Xt).squeeze(1), yt).item()
                va = loss_fn(model(Xv).squeeze(1), yv).item() if have_val else None
            if not np.isfinite(tr):
                raise RuntimeError("Training diverged (the loss is not a finite number). Try a smaller learning rate.")
            hist["epoch"].append(epoch)
            hist["train_loss"].append(tr)
            hist["val_loss"].append(va)

            if have_val:
                if va < best_val - self.min_delta:
                    best_val, best_epoch, since_best = va, epoch, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    since_best += 1
            if on_epoch is not None and on_epoch(epoch, tr, va):
                reason = "stopped by the user"
                break
            if have_val and since_best >= self.patience:
                reason = f"early stopping: validation loss did not improve for {self.patience} epochs"
                break

        if best_state is not None:
            model.load_state_dict(best_state)                # back to the best epoch, not the last
        else:
            best_epoch = len(hist["epoch"])
        model.eval()
        self.model_, self.optimizer_, self.history_ = model, opt, hist
        self.best_epoch_, self.best_val_loss_ = best_epoch, (best_val if have_val else None)
        self.epochs_run_, self.stop_reason_ = len(hist["epoch"]), reason
        self.stopped_early_ = reason.startswith("early stopping")
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() first.")
        torch = _torch()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"Expected X with {self.n_features_} columns, got shape {X.shape}.")
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy().astype(np.float64)


# --------------------------------------------------------- per-epoch loss, emitted
def format_history(net, max_lines=40):
    """The per-epoch train and validation loss as text, in the walkthrough's style:

        epoch   1   train 0.00412   val 0.00408
        epoch  12   train 0.00338   val 0.00381  <- best
        stop: early stopping: validation loss did not improve for 10 epochs; weights from epoch 12 kept

    Every epoch is listed up to `max_lines`; a longer run is thinned evenly, always keeping the best
    and the last epoch. (For live updates while training, pass `on_epoch` to NeuralNet.fit.)"""
    h, n = net.history_, len(net.history_["epoch"])
    keep = range(n)
    if n > max_lines:
        step = int(np.ceil(n / (max_lines - 2)))
        keep = sorted(set(range(0, n, step)) | {net.best_epoch_ - 1, n - 1})
    lines = []
    for i in keep:
        va = h["val_loss"][i]
        mark = "  <- best" if va is not None and h["epoch"][i] == net.best_epoch_ else ""
        lines.append(f"epoch {h['epoch'][i]:>3}   train {h['train_loss'][i]:.5f}   "
                     f"val {'  n/a  ' if va is None else format(va, '.5f')}{mark}")
    kept = f"; weights from epoch {net.best_epoch_} kept" if net.best_val_loss_ is not None else ""
    lines.append(f"stop: {net.stop_reason_}{kept}")
    return "\n".join(lines)


def plot_loss_curves(ax, history, best_epoch=None):
    """Draw train and validation loss on a matplotlib Axes (works for the Tkinter canvas, or for
    saving a PNG). Call it again with the growing history from an `on_epoch` callback for a live plot."""
    ep = history["epoch"]
    ax.plot(ep, history["train_loss"], label="train", color="#005030")
    if all(v is not None for v in history["val_loss"]) and len(ep):
        ax.plot(ep, history["val_loss"], label="validation", color="#B52B4C")
    if best_epoch:
        ax.axvline(best_epoch, linestyle="--", linewidth=1, color="#6E6A61", label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Network training")
    ax.legend()
    return ax


if __name__ == "__main__":
    import os

    from data import load_prices
    from features import build_features, normalize, split_by_date

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_market_A.csv")
    train, val, test, info = split_by_date(build_features(load_prices(path), 20))
    tr_n, va_n, _, _, _ = normalize(train, val, test)          # the test block is scaled but never scored here
    (Xtr, ytr), (Xva, yva) = to_xy(tr_n), to_xy(va_n)
    print(f"train {info['train_start']} .. {info['train_end']} ({len(ytr):,} rows)   "
          f"validation {info['val_start']} .. {info['val_end']} ({len(yva):,} rows)\n")

    forest, ann = ForestRegressor(), NeuralNet()
    try:
        _torch()
        lineup = [MeanBaseline(), forest, ann]
    except ImportError as e:
        print(f"({e}; skipping the network)\n")
        lineup = [MeanBaseline(), forest]

    print(f"{'model':<30}{'val MAE':>10}{'val RMSE':>10}{'direction acc':>15}")
    for m in lineup:
        m.fit(Xtr, ytr, Xva, yva) if isinstance(m, NeuralNet) else m.fit(Xtr, ytr)
        p = m.predict(Xva)
        d = m.predict_direction(Xva) if hasattr(m, "predict_direction") else np.where(p > 0, 1, -1)
        acc = float((d == np.where(yva > 0, 1, -1)).mean())
        print(f"{m.name:<30}{np.abs(p - yva).mean():>10.5f}{np.sqrt(((p - yva) ** 2).mean()):>10.5f}{acc:>14.1%}")
    print("\nforest feature importances:", {k: round(v, 3) for k, v in forest.importances().items()})
    if hasattr(ann, "history_"):
        print("\nnetwork training, per epoch:\n" + format_history(ann))
        print("note: validation chose the network's best epoch, so its validation numbers are a little "
              "optimistic. Compare models fairly on the test block, once, at the end.")
