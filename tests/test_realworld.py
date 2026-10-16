import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from unmix_ica.audio import write_wav
from unmix_ica.realworld import align_recordings, estimate_relative_lag, run_real_world_pipeline


def _three_mixtures(n=16000, sample_rate=8000, seed=7):
    rng = np.random.default_rng(seed)
    # Independent non-Gaussian signals: Laplace, uniform, and sparse pulses.
    s1 = rng.laplace(size=n)
    s2 = rng.uniform(-1, 1, size=n)
    s3 = rng.normal(size=n) * (rng.random(n) < 0.15)
    sources = np.vstack([s1, s2, s3])
    mixing = np.array([[1.0, 0.3, 0.2], [0.2, 1.1, 0.4], [0.3, 0.1, 1.0]])
    mixtures = mixing @ sources
    mixtures /= np.max(np.abs(mixtures)) * 1.2
    return mixtures


def _shift_right(x, samples):
    return np.concatenate([np.zeros(samples), x[:-samples]]) if samples else x.copy()


def _shift_left(x, samples):
    return np.concatenate([x[samples:], np.zeros(samples)]) if samples else x.copy()


def test_estimate_relative_lag_positive_and_negative():
    fs = 8000
    mixtures = _three_mixtures(n=fs * 5)
    ref = mixtures[0]
    for delay in (240, -160):
        candidate = _shift_right(ref, delay) if delay > 0 else _shift_left(ref, abs(delay))
        estimated = estimate_relative_lag(ref, candidate, fs, max_lag_seconds=0.5, sync_window_seconds=4)
        assert abs(estimated - delay) <= 16  # alignment is coarsely estimated at 1 kHz


def test_align_recordings_removes_relative_start_offsets():
    fs = 8000
    mixtures = _three_mixtures(n=fs * 4)
    recordings = [mixtures[0], _shift_right(mixtures[1], 200), _shift_left(mixtures[2], 120)]
    aligned, lags = align_recordings(recordings, fs, max_lag_seconds=0.5, sync_window_seconds=3)
    assert aligned.shape[0] == 3
    assert aligned.shape[1] >= fs * 3
    assert abs(lags[1] - 200) <= 16
    assert abs(lags[2] + 120) <= 16


def test_real_world_pipeline_writes_outputs_and_diagnostics(tmp_path: Path):
    fs = 8000
    mixtures = _three_mixtures(n=fs * 3)
    inputs = []
    for i, mix in enumerate(mixtures, start=1):
        path = tmp_path / f"device_{i}.wav"
        write_wav(path, fs, mix)
        inputs.append(path)
    out = tmp_path / "result"
    summary = run_real_world_pipeline(inputs, out, target_sample_rate=fs, max_iter=300)
    assert summary["number_of_device_mixtures"] == 3
    assert (out / "component_01_raw.wav").exists()
    assert (out / "component_03_raw.wav").exists()
    assert (out / "components_raw.npy").exists()
    assert (out / "aligned_device_recordings.npy").exists()
    assert (out / "01_device_recordings.png").exists()
    assert (out / "02_ica_components.png").exists()
    assert (out / "03_correlation_diagnostics.png").exists()
    details = json.loads((out / "diagnostics.json").read_text())
    assert any("SI-SDR" in limitation for limitation in details["limitations"])
    assert details["inputs"][0]["input_sample_rate_hz"] == fs


def test_align_rejects_too_few_channels():
    try:
        align_recordings([np.ones(10)], sample_rate=1000)
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("Expected ValueError for one input recording")


def test_real_world_pipeline_resamples_mixed_input_rates(tmp_path: Path):
    from scipy.signal import resample_poly

    fs = 8000
    mixtures = _three_mixtures(n=fs * 3, sample_rate=fs)
    rates = [fs, 12000, fs]
    inputs = []
    for i, (mix, rate) in enumerate(zip(mixtures, rates), start=1):
        audio = resample_poly(mix, rate, fs) if rate != fs else mix
        path = tmp_path / f"device_{i}.wav"
        write_wav(path, rate, audio)
        inputs.append(path)
    out = tmp_path / "resampled_result"
    summary = run_real_world_pipeline(inputs, out, target_sample_rate=fs, max_iter=300)
    assert [row["input_sample_rate_hz"] for row in summary["inputs"]] == rates
    assert summary["target_sample_rate_hz"] == fs
    assert summary["aligned_duration_seconds"] > 2.8
