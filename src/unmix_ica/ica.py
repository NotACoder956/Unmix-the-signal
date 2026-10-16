"""A compact, from-scratch symmetric FastICA implementation.

The implementation uses NumPy for linear algebra only. It does not call a
ready-made ICA implementation from scikit-learn, MNE, or another package.
Arrays use the convention (channels/components, samples).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ICAResult:
    """Outputs and diagnostics from a FastICA fit."""

    sources: np.ndarray
    unmixing_matrix: np.ndarray
    whitening_matrix: np.ndarray
    eigenvalues: np.ndarray
    n_iter: int
    converged: bool
    convergence_history: list[float]


def center_signals(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center each row/channel and return (centered data, channel means)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array shaped (channels, samples).")
    if X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("X must have at least 2 channels and 2 samples.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    means = np.mean(X, axis=1, keepdims=True)
    return X - means, means


def pca_whiten(
    X: np.ndarray, n_components: int | None = None, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center X and perform PCA whitening with covariance/eigendecomposition.

    Returns (Z, whitening_matrix, dewhitening_matrix, sorted_eigenvalues).
    Z has approximately identity covariance. No sklearn whitening helper is used.
    """
    X_centered, _ = center_signals(X)
    n_channels, n_samples = X_centered.shape
    if n_components is None:
        n_components = n_channels
    if not 1 <= n_components <= n_channels:
        raise ValueError("n_components must be between 1 and the number of channels.")

    covariance = (X_centered @ X_centered.T) / n_samples
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order][:n_components]
    eigenvectors = eigenvectors[:, order][:, :n_components]

    if np.any(eigenvalues <= eps):
        raise ValueError(
            "The mixture covariance is rank-deficient. Use linearly independent mixtures."
        )

    whitening_matrix = (eigenvectors / np.sqrt(eigenvalues)[None, :]).T
    dewhitening_matrix = eigenvectors * np.sqrt(eigenvalues)[None, :]
    Z = whitening_matrix @ X_centered
    return Z, whitening_matrix, dewhitening_matrix, eigenvalues


def symmetric_decorrelation(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project rows of W onto the orthogonal group using eigendecomposition."""
    gram = W @ W.T
    values, vectors = np.linalg.eigh(gram)
    values = np.maximum(values, eps)
    inv_sqrt = (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.T
    return inv_sqrt @ W


class FastICA:
    """Symmetric FastICA with a tanh non-linearity, implemented from scratch."""

    def __init__(
        self,
        n_components: int | None = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int = 42,
        alpha: float = 1.0,
    ) -> None:
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if tol <= 0:
            raise ValueError("tol must be positive.")
        if not 0.5 <= alpha <= 2.0:
            raise ValueError("alpha should be in the stable range [0.5, 2.0].")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.alpha = alpha
        self.result_: ICAResult | None = None
        self.mean_: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to mixed signals X and return estimated independent components."""
        X_centered, means = center_signals(X)
        self.mean_ = means
        n_channels, n_samples = X_centered.shape
        n_components = self.n_components or n_channels
        if n_components > n_channels:
            raise ValueError("n_components cannot exceed the number of channels.")

        Z, whitening, _, eigenvalues = pca_whiten(X_centered, n_components=n_components)
        rng = np.random.default_rng(self.random_state)
        W = symmetric_decorrelation(rng.normal(size=(n_components, n_components)))
        history: list[float] = []
        converged = False

        for iteration in range(1, self.max_iter + 1):
            projected = W @ Z
            activated = np.tanh(self.alpha * projected)
            derivative_mean = np.mean(
                self.alpha * (1.0 - activated**2), axis=1
            )
            W_new = (activated @ Z.T) / n_samples - derivative_mean[:, None] * W
            W_new = symmetric_decorrelation(W_new)
            # Sign-insensitive convergence: an independent component has no
            # meaningful orientation, so W and -W represent the same solution.
            delta = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0))
            history.append(float(delta))
            W = W_new
            if delta < self.tol:
                converged = True
                break

        sources = W @ Z
        # Include whitening so the stored matrix maps centered mixtures to sources.
        self.result_ = ICAResult(
            sources=sources,
            unmixing_matrix=W @ whitening,
            whitening_matrix=whitening,
            eigenvalues=eigenvalues,
            n_iter=iteration,
            converged=converged,
            convergence_history=history,
        )
        return sources

    def fit(self, X: np.ndarray) -> "FastICA":
        self.fit_transform(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform additional samples using the fitted unmixing matrix."""
        if self.result_ is None or self.mean_ is None:
            raise RuntimeError("Call fit or fit_transform before transform.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != self.mean_.shape[0]:
            raise ValueError("X must have the same channel count as the fitted data.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")
        X_centered = X - self.mean_
        return self.result_.unmixing_matrix @ X_centered
