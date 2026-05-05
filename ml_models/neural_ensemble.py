"""
ml_models/neural_ensemble.py — Neural network meta-learner for ensemble calibration.

Why a neural ensemble on top of XGBoost?
  Each disease model (diabetes.pkl, anemia.pkl, etc.) is trained independently on
  its own labelled dataset.  Their raw probability outputs can be systematically
  biased — one model might always over-predict, another under-predict — and they
  don't account for correlations between diseases (e.g. high diabetes risk often
  correlates with elevated kidney risk).

  The NeuralEnsemble learns to correct these biases by training a small MLP on
  the HELD-OUT XGBoost outputs alongside ground-truth labels.  After training,
  it maps [p_diabetes, p_anemia, p_infection] → [calibrated_diabetes, calibrated_anemia,
  calibrated_infection] using weights that encode cross-disease correlations.

Architecture (per report type):
  Input  — N floats, one raw XGBoost probability per disease
  Hidden — Dense(32, ReLU) → Dense(16, ReLU)
  Output — Dense(N, Sigmoid) — one calibrated probability per disease

Pass-through fallback:
  When ensemble weights don't exist (file not found), the system falls back to
  returning the raw XGBoost probabilities unchanged.  This means the system works
  immediately after training the individual disease models — the ensemble is an
  optional calibration layer that improves accuracy but is not required.

Persistence:
  Weights are stored as plain numpy arrays serialised with joblib into
  ml_models/xgboost/ensemble_{report_type}.pkl — the same directory as the
  individual model .pkl files.  No heavy framework (TensorFlow, PyTorch) is
  needed; the pure-numpy MLP avoids adding several GB of dependencies to the
  Docker image.
"""

import os
import logging
from typing import Dict, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)


# ── Activation functions ──────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit: max(0, x).

    Used in the hidden layers of the MLP.  ReLU is preferred over sigmoid/tanh
    for hidden layers because it does not saturate for large positive inputs,
    which keeps gradients flowing during backpropagation and speeds up training.

    Args:
        x — numpy array of any shape.

    Returns:
        Array of same shape with negative values zeroed out.
    """
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation: 1 / (1 + e^(-x)).

    Used in the OUTPUT layer only, because each output must be a probability
    in [0, 1].  The clip(-20, 20) prevents overflow in np.exp() for extreme
    inputs (e.g. if a raw logit is -1000, e^1000 would be inf without clipping).

    Args:
        x — numpy array of any shape.

    Returns:
        Array of same shape with values in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ── MLP implementation ────────────────────────────────────────────────────────

class _MLP:
    """
    Minimal 3-layer feedforward neural network implemented in pure numpy.

    Layer structure:
      Input(n_in) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(n_out, Sigmoid)

    Weights are standard numpy float32 arrays so they serialise efficiently with
    joblib.  No framework dependency means the Docker image stays lean (~200 MB
    smaller than adding PyTorch).

    Weight initialisation uses He (Kaiming) uniform scaling:
      scale = sqrt(2 / fan_in)
    This is the recommended initialiser for ReLU networks — it keeps the variance
    of activations stable across layers and prevents vanishing/exploding gradients
    at the start of training.
    """

    def __init__(self, n_in: int, n_out: int, seed: int = 42):
        """
        Allocate and initialise all weight matrices and bias vectors.

        Args:
            n_in  — input dimension (= number of diseases for the report type).
            n_out — output dimension (= n_in, one calibrated score per disease).
            seed  — random seed for reproducibility.  Set to 42 by default so
                    two identical training runs produce the same initial weights.
        """
        rng = np.random.default_rng(seed)

        # He initialisation scales for each layer based on its input size.
        scale1 = np.sqrt(2.0 / n_in)
        scale2 = np.sqrt(2.0 / 32)
        scale3 = np.sqrt(2.0 / 16)

        # Layer 1: (n_in → 32)
        self.W1 = rng.standard_normal((n_in, 32)).astype(np.float32) * scale1
        self.b1 = np.zeros(32, dtype=np.float32)

        # Layer 2: (32 → 16)
        self.W2 = rng.standard_normal((32, 16)).astype(np.float32) * scale2
        self.b2 = np.zeros(16, dtype=np.float32)

        # Output layer: (16 → n_out)
        self.W3 = rng.standard_normal((16, n_out)).astype(np.float32) * scale3
        self.b3 = np.zeros(n_out, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run a forward pass through all three layers.

        The @ operator is numpy's matrix multiplication.  For input shape (1, n_in):
          h1  = ReLU(x  @ W1 + b1)   shape (1, 32)
          h2  = ReLU(h1 @ W2 + b2)   shape (1, 16)
          out = σ(h2 @ W3 + b3)      shape (1, n_out)

        Args:
            x — array of shape (n_in,) or (batch, n_in).

        Returns:
            Array of shape (n_out,) or (batch, n_out) — probabilities in (0,1).
        """
        h1 = _relu(x @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        return _sigmoid(h2 @ self.W3 + self.b3)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
        batch_size: int = 32,
    ) -> None:
        """
        Train the MLP with mini-batch SGD and binary cross-entropy loss.

        Binary cross-entropy (BCE) is the standard loss for multi-label binary
        classification — each output is independently treated as a Bernoulli
        random variable:
          BCE = -mean( y*log(out) + (1-y)*log(1-out) )

        Backpropagation is computed analytically (no autograd):
          ∂L/∂out = (out - y) / batch_size       (BCE gradient through sigmoid)
          ∂L/∂W3  = h2.T @ dout                  (chain rule for W3)
          ∂L/∂h2  = dout @ W3.T; zero where h2<=0 (ReLU dead neuron mask)
          ... same pattern for W2, W1

        The epsilon (1e-7) prevents log(0) when a sigmoid output is exactly 0 or 1,
        which can happen for very confident predictions.

        Args:
            X          — shape (n_samples, n_in)  — XGBoost proba outputs.
            y          — shape (n_samples, n_out) — binary ground-truth labels.
            epochs     — number of full passes over the training data.
            lr         — learning rate (SGD step size).
            batch_size — number of samples per gradient update.
        """
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        n = len(X)

        for epoch in range(epochs):
            # Shuffle indices each epoch to prevent the model from memorising
            # the order of samples (reduces overfitting on small datasets).
            idx = np.random.permutation(n)
            total_loss = 0.0

            for start in range(0, n, batch_size):
                batch_idx = idx[start: start + batch_size]
                Xb, yb = X[batch_idx], y[batch_idx]

                # ── Forward pass ──────────────────────────────────────────
                h1  = _relu(Xb @ self.W1 + self.b1)
                h2  = _relu(h1 @ self.W2 + self.b2)
                out = _sigmoid(h2 @ self.W3 + self.b3)

                # ── Loss (binary cross-entropy) ───────────────────────────
                eps = 1e-7   # numerical stability guard for log(0)
                loss = -np.mean(
                    yb * np.log(out + eps) + (1 - yb) * np.log(1 - out + eps)
                )
                total_loss += loss

                # ── Backpropagation ───────────────────────────────────────
                # Gradient of BCE through sigmoid output layer collapses to (out - y).
                dout = (out - yb) / len(Xb)
                dW3  = h2.T @ dout
                db3  = dout.sum(axis=0)

                # Gradient through layer 2 + ReLU dead neuron mask.
                dh2 = dout @ self.W3.T
                dh2[h2 <= 0] = 0.0   # ReLU derivative: 0 for non-activated neurons
                dW2 = h1.T @ dh2
                db2 = dh2.sum(axis=0)

                # Gradient through layer 1 + ReLU dead neuron mask.
                dh1 = dh2 @ self.W2.T
                dh1[h1 <= 0] = 0.0
                dW1 = Xb.T @ dh1
                db1 = dh1.sum(axis=0)

                # ── SGD weight update ─────────────────────────────────────
                self.W3 -= lr * dW3;  self.b3 -= lr * db3
                self.W2 -= lr * dW2;  self.b2 -= lr * db2
                self.W1 -= lr * dW1;  self.b1 -= lr * db1

            # Log every 50 epochs so training progress is visible in the console.
            if (epoch + 1) % 50 == 0:
                logger.info(f"  Epoch {epoch + 1}/{epochs}  loss={total_loss:.4f}")


# ── Public NeuralEnsemble class ───────────────────────────────────────────────

class NeuralEnsemble:
    """
    Public wrapper that owns a _MLP meta-learner and provides inference,
    training, save, and load methods.

    The separation between NeuralEnsemble (public API) and _MLP (internal
    computation) exists so that future callers can swap the underlying network
    (e.g. to a PyTorch model) without changing the interface used by predict.py.

    Typical inference flow:
        ensemble = NeuralEnsemble.load("ensemble_blood.pkl")
        calibrated = ensemble.predict(raw_xgb_probas)   # np.ndarray (n_diseases,)

    Typical training flow:
        ensemble = NeuralEnsemble(n_diseases=3)
        ensemble.fit(X_train, y_train, epochs=200, lr=0.01)
        ensemble.save("ml_models/xgboost/ensemble_blood.pkl")
    """

    def __init__(self, n_diseases: int):
        """
        Create an untrained ensemble for the given number of diseases.

        Args:
            n_diseases — number of individual disease models for the report type.
                         Must match len(REPORT_MODEL_MAP[report_type]).
        """
        self.n_diseases = n_diseases
        self._mlp: Optional[_MLP] = None
        self._trained = False   # False → predict() uses identity pass-through

    def predict(self, xgb_probas: np.ndarray) -> np.ndarray:
        """
        Produce calibrated disease risk scores from raw XGBoost probabilities.

        If the ensemble HAS been trained (_trained=True and _mlp is set):
          passes xgb_probas through the MLP forward pass → calibrated scores.
        If the ensemble has NOT been trained (no weights loaded):
          returns xgb_probas unchanged (identity pass-through).

        The pass-through ensures the system is functional immediately after
        training the individual XGBoost models — the ensemble is optional.

        Args:
            xgb_probas — 1-D array of shape (n_diseases,), in the order defined
                         by REPORT_MODEL_MAP.  Values are in [0, 1].

        Returns:
            np.ndarray of shape (n_diseases,) — calibrated probabilities in (0,1).
        """
        x = np.array(xgb_probas, dtype=np.float32).reshape(1, -1)

        if self._trained and self._mlp is not None:
            return self._mlp.forward(x)[0]

        # Untrained — log at DEBUG level (not WARNING) because this is expected
        # on a fresh installation before the ensemble has been trained.
        logger.debug(
            "NeuralEnsemble: no trained weights found — "
            "returning raw XGBoost probabilities."
        )
        return x[0]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
        batch_size: int = 32,
    ) -> None:
        """
        Train a new _MLP meta-learner on held-out XGBoost output + ground truth.

        Always creates a fresh _MLP (previous weights are discarded).  This means
        calling fit() twice re-trains from random initial weights, not from the
        current weights — there is no warm-starting.

        Args:
            X          — shape (n_samples, n_diseases) — XGBoost proba columns.
            y          — shape (n_samples, n_diseases) — binary ground-truth labels.
            epochs     — training epochs (default 200, ~seconds on a single CPU).
            lr         — learning rate (default 0.01 — conservative for small data).
            batch_size — mini-batch size (default 32; use smaller values if N < 50).
        """
        self._mlp = _MLP(n_in=self.n_diseases, n_out=self.n_diseases)
        self._mlp.fit(X, y, epochs=epochs, lr=lr, batch_size=batch_size)
        self._trained = True
        logger.info(
            f"NeuralEnsemble trained: {self.n_diseases} inputs → "
            f"{self.n_diseases} outputs"
        )

    def save(self, path: str) -> None:
        """
        Serialise the ensemble weights to disk with joblib.

        The payload dict stores everything needed to reconstruct the ensemble:
          n_diseases — so load() can validate the architecture matches the report type.
          trained    — so load() knows whether to use forward() or pass-through.
          mlp        — the _MLP object with all W1/b1/W2/b2/W3/b3 arrays.

        joblib is preferred over pickle for numpy arrays because it uses
        memory-mapped files which are faster to load and support compression.

        Args:
            path — absolute or relative path for the output .pkl file.
                   Parent directory is created if it doesn't exist.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "n_diseases": self.n_diseases,
            "trained":    self._trained,
            "mlp":        self._mlp,
        }
        joblib.dump(payload, path)
        logger.info(f"NeuralEnsemble saved → {path}")

    @classmethod
    def load(cls, path: str) -> "NeuralEnsemble":
        """
        Load a previously saved ensemble from disk.

        Reconstructs the NeuralEnsemble with the saved n_diseases, _trained flag,
        and _mlp weight arrays.  The loaded object is ready for predict() calls.

        Args:
            path — path to the .pkl file written by save().

        Returns:
            A NeuralEnsemble instance with _trained=True and _mlp populated.
        """
        payload = joblib.load(path)
        obj = cls(n_diseases=payload["n_diseases"])
        obj._mlp     = payload["mlp"]
        obj._trained = payload["trained"]
        logger.info(f"NeuralEnsemble loaded ← {path}")
        return obj


# ── Module-level ensemble cache ───────────────────────────────────────────────

# Keyed by report_type string (e.g. "blood", "lipid").
# Populated lazily on first predict() call and reused for all subsequent tasks
# in the same worker process.
_ENSEMBLE_CACHE: Dict[str, NeuralEnsemble] = {}


def load_ensemble(report_type: str, ensemble_path: str, n_diseases: int) -> NeuralEnsemble:
    """
    Return a cached NeuralEnsemble for the given report type, loading if needed.

    Cache hit  → returns the already-loaded ensemble (fast path, no I/O).
    Cache miss → checks if the .pkl file exists:
      File found   → NeuralEnsemble.load() reads and caches it.
      File missing → Creates a fresh untrained instance (identity pass-through)
                     and caches it so subsequent calls also take the fast path.

    The caching is process-scoped (Python module-level dict), not thread-safe.
    This is fine because Celery workers run with worker_concurrency=1 (single
    thread).  If concurrency is ever increased, a threading.Lock would be needed.

    Args:
        report_type   — cache key and log context (e.g. "blood").
        ensemble_path — absolute path to the ensemble .pkl file.
        n_diseases    — number of disease models; used to create the untrained
                        pass-through instance when the file doesn't exist.

    Returns:
        NeuralEnsemble — either trained (loaded from disk) or untrained (pass-through).
    """
    if report_type in _ENSEMBLE_CACHE:
        return _ENSEMBLE_CACHE[report_type]

    if os.path.exists(ensemble_path):
        ensemble = NeuralEnsemble.load(ensemble_path)
    else:
        logger.info(
            f"No ensemble weights at {ensemble_path} — "
            f"using identity pass-through for '{report_type}'."
        )
        ensemble = NeuralEnsemble(n_diseases=n_diseases)

    _ENSEMBLE_CACHE[report_type] = ensemble
    return ensemble
