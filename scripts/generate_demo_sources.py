#!/usr/bin/env python3
"""Generate reproducible speech, music, and ambient source WAVs.

Speech is synthesized locally with eSpeak when it is installed. If eSpeak is
unavailable, a deterministic formant-like voiced signal is generated instead.
Music and ambience are generated procedurally, so no external downloads are
needed. These are demonstration sources, not real-world field recordings.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from unmix_ica.audio import read_wav, write_wav, normalize_source

SAMPLE_RATE = 8000
DURATION = 7.0
N = int(SAMPLE_RATE * DURATION)


def speech_like_fallback(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Build a deterministic voiced/formant-like waveform as a no-TTS fallback."""
    fundamental = 125 + 12 * np.sin(2 * np.pi * 0.7 * t) + 7 * np.sin(2 * np.pi * 2.1 * t)
    phase = 2 * np.pi * np.cumsum(fundamental) / SAMPLE_RATE
    voiced = sum((1.0 / h) * np.sin(h * phase) for h in range(1, 16))
    formants = (0.55 + 0.45 * np.sin(2 * np.pi * 2.7 * t)) * np.sin(2 * np.pi * 720 * t)
    formants += (0.35 + 0.30 * np.sin(2 * np.pi * 1.9 * t + 1)) * np.sin(2 * np.pi * 1450 * t)
    syllable = 0.15 + 0.85 * (np.sin(2 * np.pi * 3.2 * t) > -0.45)
    breath = rng.normal(0, 0.04, size=t.size)
    return syllable * (0.65 * voiced + 0.3 * formants) + breath * (1 - syllable)


def make_speech(out: Path, t: np.ndarray, rng: np.random.Generator) -> None:
    espeak = shutil.which("espeak") or shutil.which("espeak-ng")
    if espeak:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "speech.wav"
            text = (
                "This demonstration mixes speech, music, and ambient sound. "
                "Independent component analysis will try to separate the sources."
            )
            subprocess.run([espeak, "-s", "145", "-w", str(raw), text], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            sr, audio = read_wav(raw, target_sample_rate=SAMPLE_RATE)
            if len(audio) < len(t):
                audio = np.pad(audio, (0, len(t) - len(audio)))
            else:
                audio = audio[:len(t)]
    else:
        audio = speech_like_fallback(t, rng)
    write_wav(out, SAMPLE_RATE, normalize_source(audio, target_rms=0.2))


def make_music(t: np.ndarray, out: Path) -> None:
    # A small synthetic instrumental phrase with melody, soft chords, and bass.
    bpm = 96
    beat = 60 / bpm
    melody = [261.63, 329.63, 392.00, 523.25, 392.00, 329.63, 293.66, 349.23,
              440.00, 587.33, 440.00, 349.23, 261.63, 329.63, 392.00, 523.25]
    note_index = np.floor(t / (beat / 2)).astype(int) % len(melody)
    note_freq = np.asarray(melody)[note_index]
    note_phase = 2 * np.pi * np.cumsum(note_freq) / SAMPLE_RATE
    env_phase = (t % (beat / 2)) / (beat / 2)
    attack_release = np.minimum(1.0, env_phase * 12) * np.minimum(1.0, (1 - env_phase) * 8)
    lead = (np.sin(note_phase) + 0.35 * np.sin(2 * note_phase) + 0.12 * np.sin(3 * note_phase)) * attack_release
    chord_freqs = [130.81, 164.81, 196.00, 146.83]
    chord_index = np.floor(t / (2 * beat)).astype(int) % len(chord_freqs)
    chord_phase = 2 * np.pi * np.cumsum(np.asarray(chord_freqs)[chord_index]) / SAMPLE_RATE
    chord = 0.45 * (np.sin(chord_phase) + 0.25 * np.sin(2 * chord_phase))
    bass = 0.35 * np.sin(2 * np.pi * (65.41 + 8 * np.sin(2 * np.pi * 0.2 * t)) * t)
    pulse = np.exp(-((t % beat) / 0.045) ** 2) * 0.12
    audio = lead + chord + bass + pulse
    write_wav(out, SAMPLE_RATE, normalize_source(audio, target_rms=0.2))


def make_ambience(t: np.ndarray, out: Path, rng: np.random.Generator) -> None:
    # Pink-ish rain bed plus sparse droplets. All randomness is seeded.
    white = rng.normal(0, 1, size=t.size)
    # One-pole smoothing creates a softer, lower-frequency noise bed.
    smooth = np.empty_like(white)
    smooth[0] = white[0]
    for i in range(1, len(white)):
        smooth[i] = 0.93 * smooth[i - 1] + 0.07 * white[i]
    rain = 0.65 * white + 0.55 * smooth
    drops = np.zeros_like(t)
    count = 95
    positions = rng.integers(0, len(t), size=count)
    for pos in positions:
        width = int(rng.integers(12, 55))
        end = min(len(t), pos + width)
        if end > pos:
            x = np.arange(end - pos)
            drops[pos:end] += rng.uniform(0.3, 1.0) * np.exp(-x / max(3, width / 5)) * np.sin(2 * np.pi * rng.uniform(600, 2600) * x / SAMPLE_RATE)
    slow_wind = 0.25 * np.sin(2 * np.pi * 0.18 * t) * rng.normal(0, 1, len(t))
    audio = rain + 0.25 * drops + slow_wind
    write_wav(out, SAMPLE_RATE, normalize_source(audio, target_rms=0.2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "originals")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t = np.arange(N, dtype=np.float64) / SAMPLE_RATE
    make_speech(args.output_dir / "speech.wav", t, rng)
    make_music(t, args.output_dir / "music.wav")
    make_ambience(t, args.output_dir / "ambient_rain.wav", rng)
    print(f"Wrote three {DURATION:.1f}-second, {SAMPLE_RATE} Hz demo sources to {args.output_dir}")


if __name__ == "__main__":
    main()
