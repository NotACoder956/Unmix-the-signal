# Real-world recordings from multiple devices

This adds a real-recording path to **Unmix the Signal (ICA)**. The repository can accept WAV files captured by separate phones, tablets, USB microphones or recorders. The script resamples them to a common rate, estimates their start offsets by cross-correlation, aligns the overlapping section, runs our own FastICA implementation, and saves components and diagnostic plots.

> **Physical-capture note:** genuine recordings from separate devices cannot be generated remotely in a software-only environment. This repository includes the complete import/alignment/separation/evaluation-diagnostics workflow and the protocol below. To complete the physical real-world recording bonus, capture a short session on actual devices and put the WAV files in `data/real_world/`. Do not label the bundled procedural demo as real-world data.

## Recommended quick recording protocol

You need at least as many distinct device recordings as sources you hope to recover. Three phones/microphones and three reasonably independent simultaneous sources are a good starting point for the 3-source ICA demo.

1. Choose a quiet room with minimal echo, if possible.
2. Make three sound sources active at the same time. For an accessible demo, play the project's three source WAV files (`speech.wav`, `music.wav`, `ambient_rain.wav`) through separate speakers placed at different positions. Alternatively, record real speech, music, and an environmental source, but obtain permission from people being recorded.
3. Place three recording devices in different positions so each one captures a different proportion of the sounds. All devices must record the **same sound scene at the same time**. Three recordings made at different times are not valid mixtures for this ICA experiment.
4. Set the devices to WAV/PCM if supported, use the same sample rate (16 kHz or 44.1/48 kHz is fine), disable automatic noise cancellation when possible, and avoid clipping.
5. Start all recorders, then make one sharp clap visible/audible to all devices near the beginning. If possible, clap again near the end so you can notice clock drift. Record 20–30 seconds. Keep the devices stationary during the test.
6. Transfer the files to your computer, rename them `device_1.wav`, `device_2.wav`, and `device_3.wav`, and put them in `data/real_world/`.
7. Ensure each file contains the same segment of the sound scene. The code estimates the start offsets using cross-correlation and trims all inputs to their common overlap. It does **not** automatically correct clock drift throughout a long recording, so keep the recording short. If the devices were not started together or the recordings contain little shared content, the synchronization may fail.

### Why play three tracks from separate speakers?

That creates a repeatable room experiment: three known source signals are emitted simultaneously, each microphone receives a different acoustic mixture, and the room/speakers/devices contribute real-world effects. It is stronger than adding arrays in software because the captured mixtures include actual room acoustics and device frequency responses. However, loudspeaker playback is not the same as three people or naturally occurring sources, and room reverberation means the instantaneous ICA model will only be approximate.

## Run the real-world pipeline

From the repository root, activate your virtual environment and install the package (`python -m pip install -r requirements.txt` then `python -m pip install -e .`). Then run:

```bash
python scripts/run_device_recordings.py \
  --inputs data/real_world/device_1.wav data/real_world/device_2.wav data/real_world/device_3.wav \
  --output-dir outputs/real_world
```

For two device recordings, supply two input paths. The number of components equals the number of input recordings. Use at least as many well-conditioned, synchronized mixtures as the number of sources you hope to recover. More microphones do not guarantee success if their signals are duplicate or the mixing matrix is nearly rank-deficient.

If you have already aligned the recordings yourself, use `--no-auto-align`:

```bash
python scripts/run_device_recordings.py \
  --inputs data/real_world/device_1.wav data/real_world/device_2.wav data/real_world/device_3.wav \
  --no-auto-align
```

Increase `--max-lag-seconds` if the starts differ by more than 3 seconds. Do so only when the files actually contain overlapping audio; otherwise the algorithm might align unrelated sections.

## Files generated

`outputs/real_world/` will contain:

- `component_01_raw.wav`, `component_02_raw.wav`, ...: separated but **unlabeled** ICA components. Listen to them to identify which source appears in each.
- `components_raw.npy`: exact floating-point ICA outputs before WAV peak protection.
- `aligned_device_recordings.npy`: the aligned input mixture matrix.
- `diagnostics.json`: sample rates, estimated offsets, convergence, pairwise correlations, component excess kurtosis and limitations.
- `01_device_recordings.png`: aligned input waveforms.
- `02_ica_components.png`: estimated component waveforms.
- `03_correlation_diagnostics.png`: input and output correlation matrices.
- `README.txt`: description of the output files.

The algorithm cannot know that a component is “speech” or “music” using ICA alone. ICA's order, sign and scale are ambiguous. These output files are therefore not assigned guessed semantic names.

## How to evaluate without clean ground truth

The real-world scene normally has no clean isolated copy of every source, so Pearson correlation to originals and SI-SDR are not available. Do not publish made-up SI-SDR numbers. This workflow instead reports:

- whether FastICA converged and how many iterations it used;
- estimated start offsets and retained duration;
- input and output pairwise correlation matrices (a useful diagnostic, but zero correlation alone does not prove independence);
- excess kurtosis for each estimated component;
- subjective listening comparison between the input recordings and output components.

If using the project's original source tracks as loudspeaker playback, they are known references, but room coloration, loudspeaker response, propagation delays and reverberation prevent a simple direct comparison from being a complete measure of real-world separation quality. A more rigorous score would need careful source alignment and reference-aware evaluation.

## Troubleshooting

- **Synchronization fails or produces a tiny overlap:** capture the same event on all devices, include a sharp clap at the beginning, shorten the recordings, or manually align them and use `--no-auto-align`.
- **Rank-deficient covariance error:** one device may be silent, two recordings may be nearly identical, or one source may dominate every recording. Move the devices, make sure several sources are simultaneously active, and recapture.
- **ICA doesn't converge:** inspect the plots and `diagnostics.json`; try a shorter synchronized section, more samples, less clipping, and more varied microphone locations. A non-converged run should be reported honestly.
- **Output components are not labeled, quieter or inverted:** this is normal ICA ambiguity. Compare by listening; do not assume component 1 is speech.
- **Real-world results are worse than the synthetic demo:** expected. Real rooms have reverberation, noise, device processing, clock errors, and potentially dependent sources.

## Suggested report paragraph

> We extended the controlled source-separation experiment to a real room by recording the same simultaneous sound scene using separate recording devices at different locations. The input WAV files were resampled to a common processing rate and aligned using cross-correlation, then processed by our from-scratch PCA-whitening and FastICA implementation. Because clean isolated source references were not available for the microphone recordings, we report convergence, alignment offsets, correlations, component statistics, and listening observations rather than claiming reference-based SI-SDR. The real-room setup also introduces reverberant/convolutive mixing and device-clock differences, which are limitations of the basic instantaneous ICA model.

Only use this paragraph after you have actually made and run a real-device recording. Before then, describe the recorder workflow as implemented, and clearly say the physical recording session is pending.
