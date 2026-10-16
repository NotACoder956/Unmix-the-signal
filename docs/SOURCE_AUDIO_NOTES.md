# Notes on the bundled demo audio

The repository includes three short 8 kHz mono WAV files so the project runs offline and the reported experiment is reproducible:

- `speech.wav`: speech synthesized from a short original sentence with eSpeak in the build environment. The generator script uses local eSpeak/eSpeak-NG when available; otherwise it produces a deterministic formant-like voiced signal.
- `music.wav`: a procedurally generated instrumental-style melody with harmonic tones and a bass line.
- `ambient_rain.wav`: procedurally generated rain-like noise with short transient droplets.

The music and ambience are synthetic sound examples, not recordings of a real instrument or weather. All three are meant to be simple, independent, controlled reference sources for algorithm development. The WAVs are included in the repository; no external download is required to run the experiment.

Regenerate them with:

```bash
python scripts/generate_demo_sources.py
```

Regenerating the speech file may produce different samples if a different eSpeak version or voice is installed. The run's included WAV files are the source of truth for the committed metrics. The Python source code license is in the top-level `LICENSE`; this note is not a claim that every installed voice engine or voice is distributed under that same license.
