import numpy as np
from pathlib import Path
from unmix_ica.audio import read_wav, write_wav, normalize_source


def test_wav_round_trip(tmp_path: Path):
    sr = 8000
    t = np.arange(sr // 2) / sr
    audio = 0.4 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    write_wav(path, sr, audio)
    read_sr, decoded = read_wav(path)
    assert read_sr == sr
    assert decoded.shape == audio.shape
    assert np.corrcoef(audio, decoded)[0, 1] > 0.999


def test_normalize_source_sets_rms():
    x = np.random.default_rng(10).normal(size=1000)
    y = normalize_source(x, target_rms=0.3)
    assert abs(float(np.sqrt(np.mean(y**2))) - 0.3) < 1e-10
    assert abs(float(np.mean(y))) < 1e-12
