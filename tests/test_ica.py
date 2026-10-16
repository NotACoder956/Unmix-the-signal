import numpy as np

from unmix_ica.ica import FastICA, center_signals, pca_whiten
from unmix_ica.audio import mix_sources
from unmix_ica.evaluate import align_sources, correlation_matrix


def test_center_signals_removes_channel_means():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 1000)) + np.array([[2.0], [-3.0], [0.7]])
    centered, means = center_signals(X)
    assert np.allclose(centered.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(means[:, 0], [2.0, -3.0, 0.7], atol=0.1)


def test_pca_whitening_produces_identity_covariance():
    rng = np.random.default_rng(2)
    S = np.vstack([
        rng.laplace(size=5000),
        rng.uniform(-1, 1, size=5000),
        np.sign(rng.normal(size=5000)) + 0.05 * rng.normal(size=5000),
    ])
    A = np.array([[1.0, 0.4, 0.2], [0.2, 1.0, 0.5], [0.4, 0.1, 1.0]])
    X = A @ S
    Z, _, _, _ = pca_whiten(X)
    covariance = Z @ Z.T / Z.shape[1]
    assert np.allclose(covariance, np.eye(3), atol=1e-8)


def test_fastica_recovers_independent_non_gaussian_sources():
    rng = np.random.default_rng(4)
    count = 12000
    sources = np.vstack([
        rng.laplace(size=count),
        rng.uniform(-1.5, 1.5, size=count),
        np.sign(rng.normal(size=count)) + 0.08 * rng.normal(size=count),
    ])
    A = np.array([[1.0, 0.45, 0.2], [0.25, 1.0, 0.5], [0.4, 0.2, 1.0]])
    mixtures = mix_sources(sources, A)
    estimates = FastICA(max_iter=500, tol=1e-6, random_state=42).fit_transform(mixtures)
    aligned, _, _ = align_sources(sources, estimates)
    corr = correlation_matrix(sources, aligned)
    assert np.mean(np.diag(corr)) > 0.97


def test_mix_rejects_rank_deficient_matrix():
    sources = np.ones((2, 20))
    bad_matrix = np.array([[1.0, 1.0], [2.0, 2.0]])
    try:
        mix_sources(sources, bad_matrix)
    except ValueError as exc:
        assert "full column rank" in str(exc)
    else:
        raise AssertionError("Expected a rank-deficient mixing matrix to fail")
