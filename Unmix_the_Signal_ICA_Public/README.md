# Unmix the Signal (ICA)

**A reproducible audio source-separation project using PCA whitening and a from-scratch FastICA implementation, with an additional workflow for real recordings captured by multiple devices.**

This project addresses the VinnovateIT task titled **“Unmix the Signal (ICA)”**. It starts with three known clean demo sources, mixes them with a fixed linear mixing matrix, and then tries to recover the independent sources using only the mixtures. It includes source and mixture WAV files, separated audio, plots, metrics, automated tests, a technical report, and a Jupyter notebook. A separate real-world workflow can import WAV recordings captured simultaneously by multiple phones or microphones, align their starts, run the same ICA implementation, and report diagnostics.

> **Important scope note:** the included demo uses eSpeak-synthesized speech plus procedurally generated instrumental music and rain ambience. It is a controlled, repeatable demonstration—not a claim of production-quality separation on arbitrary recordings or real reverberant microphone recordings. The source files are included, so the main experiment does not require eSpeak or internet access.

## Results from the included demo

The saved run used three sources, three mixtures, 8 kHz mono audio, and 7 seconds of audio per source. FastICA converged in 5 iterations.

| Metric | PCA whitening only | PCA whitening + ICA |
|---|---:|---:|
| Mean absolute correlation with original sources | 0.7042 | 0.9999 |
| Mean SI-SDR | 0.09 dB | 39.84 dB |

A separate noise experiment adds white noise to the mixtures and reruns ICA:

| Input mixture SNR | Mean absolute correlation | Mean SI-SDR |
|---:|---:|---:|
| 30 dB | 0.9990 | 27.08 dB |
| 20 dB | 0.9907 | 17.24 dB |
| 10 dB | 0.9197 | 7.41 dB |
| 5 dB | 0.7991 | 2.48 dB |

These scores are **specific to the bundled synthetic demonstration**. They are measured after matching estimated components to their known original sources, correcting the arbitrary sign, and applying a least-squares scale for the output WAVs. The ICA algorithm itself sees only the mixed channels; the known originals are used for evaluation and display ordering. Results on real recordings can be significantly worse.

## Quick start

Python 3.10 or newer is recommended.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Install dependencies and this project
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

# 3. Run the full experiment
python scripts/run_experiment.py

# 4. Run automated tests
python -m pytest -q
```

The WAV sources, mixed signals, separated outputs, result metrics, and plots are already included. Running the experiment recreates the generated results. It requires NumPy, SciPy, and Matplotlib; no API key, cloud service, GPU, or download is needed.

### Open the notebook

```bash
python -m pip install jupyter
jupyter notebook notebooks/unmix_signal_ica.ipynb
```

## Real-world multi-device recording extension

The software path for processing actual recordings from multiple phones or microphones is included. It resamples input WAV files to a common rate, estimates relative start offsets by cross-correlation, keeps the common overlap, runs the same from-scratch FastICA implementation, saves unlabeled components, and produces diagnostic plots/JSON. It does not invent SI-SDR or source-correlation metrics if clean reference tracks are unavailable.

**Important:** this archive does not contain physical recordings from multiple devices, because those must be captured on real devices. The controlled three-source audio elsewhere in this repository is synthetic/procedural and must not be described as real-world device data. To complete that optional physical-capture bonus, follow [`docs/REAL_WORLD_RECORDING_GUIDE.md`](docs/REAL_WORLD_RECORDING_GUIDE.md) and place the resulting WAVs in `data/real_world/`.

After capturing at least two recordings of the **same simultaneous sound scene** (three are recommended), run from the repository root:

```bash
python scripts/run_device_recordings.py --inputs data/real_world/device_1.wav data/real_world/device_2.wav data/real_world/device_3.wav --output-dir outputs/real_world
```

Generated outputs include `component_XX_raw.wav`, `components_raw.npy`, `aligned_device_recordings.npy`, `diagnostics.json`, and three plots. Component identities are unknown by design; ICA can reorder, invert, and scale components. The script does not currently correct long-term sample-clock drift, so keep the capture short and use a synchronization clap near the start. Full details and limitations are in the recording guide.

## What's in the repository?

```text
unmix_signal_ica/
├── data/
│   ├── originals/                 # Clean demo sources
│   ├── mixed/                     # Controlled benchmark mixtures
│   └── real_world/                # Place physical multi-device recordings here
├── docs/
│   ├── REAL_WORLD_RECORDING_GUIDE.md
│   └── SOURCE_AUDIO_NOTES.md
├── notebooks/
│   └── unmix_signal_ica.ipynb
├── outputs/
│   ├── separated/                 # Controlled demo outputs
│   └── real_world/                 # Created after running device-recording workflow
├── reports/
│   ├── REPORT.md
│   ├── figures/
│   └── metrics/
├── scripts/
│   ├── generate_demo_sources.py
│   ├── run_experiment.py
│   └── run_device_recordings.py
├── src/unmix_ica/
│   ├── audio.py
│   ├── evaluate.py
│   ├── experiment.py
│   ├── ica.py
│   └── realworld.py
├── tests/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
└── pyproject.toml
```

## How the algorithm works

1. **Mix the sources.** Arrange sources as a matrix `S` shaped `(sources, samples)` and make observations using `X = A @ S`. `A` is a nonsingular mixing matrix. A single global gain prevents clipping without changing the linear mixture model.
2. **Center the channels.** Subtract each channel's mean.
3. **Whiten using PCA.** Calculate the covariance matrix, use `numpy.linalg.eigh` to get eigenvectors/eigenvalues, and multiply by the inverse square root of the eigenvalues. The whitened covariance is approximately the identity matrix.
4. **Run symmetric FastICA.** Repeatedly update component directions with the `tanh` non-linearity and then symmetrically decorrelate the rows of the weight matrix. This algorithm is implemented in `src/unmix_ica/ica.py`; it does not call an off-the-shelf ICA function.
5. **Evaluate and save.** Match components to known references by maximizing absolute correlation, account for ICA's sign/permutation/scale ambiguity, calculate Pearson correlation and SI-SDR, export WAV files, and write plots.

## Why ICA rather than PCA?

PCA identifies directions that explain variance and makes transformed dimensions uncorrelated. Uncorrelated does **not** generally mean statistically independent. ICA uses higher-order statistical structure, with non-Gaussianity as a useful clue, to look for independent sources. PCA whitening is therefore a preprocessing stage, not the final source-separation algorithm.

## Deliverables covered

- **Clean sources and mixing:** three source WAVs and three linear mixtures.
- **Centering and PCA whitening:** explicitly implemented using covariance and eigendecomposition.
- **ICA from scratch:** symmetric FastICA with a `tanh` non-linearity; no library ICA routine.
- **Playable separated audio:** three aligned demo outputs, plus the raw components.
- **Evaluation:** correlations, SI-SDR, PCA-only baseline, noise/SNR experiment, plots, CSV and JSON metrics.
- **Reproducibility:** seed-controlled algorithms, scripts, tests, report, and notebook.
- **Real-world capture software:** multi-device WAV import, cross-correlation start alignment, from-scratch ICA, component WAV exports, and diagnostic plots/JSON. Physical capture files still need to be recorded by the user.

## Limitations and honest interpretation

- Basic ICA assumes mixtures are approximately instantaneous linear combinations of independent sources and that the mixing matrix has full rank. The number of mixtures should be at least the number of sources.
- ICA is not general-purpose denoising. It can struggle with dependent sources, multiple Gaussian sources, reverberation/convolutive mixing, clipping, and substantial noise.
- Component order, sign, and scale cannot be uniquely identified from mixtures alone. The `separated_*.wav` files use known demo originals to align results for listening; `component_*_raw.wav` preserves the algorithm's order; WAV-safe peak scaling is applied only if needed. The exact numeric output before WAV scaling is saved as `components_raw.npy`.
- The bundled demo audio contains synthesized/procedural examples rather than real recordings from different devices. The real-world processing software and recording protocol are included, but a real capture session still needs to be recorded on physical devices. Do not claim the physical bonus is complete until such recordings have been processed and inspected.
- The fixed mixing matrix and included examples make the experiment reproducible, not universally representative.

## Troubleshooting

- **`ModuleNotFoundError: unmix_ica`:** activate the virtual environment and run `python -m pip install -e .` from the repository root.
- **Tests cannot find the package:** run `python -m pytest -q` from the repository root. `tests/conftest.py` adds `src/` during testing.
- **WAV output sounds quieter/different:** ICA outputs have arbitrary scale and sign. Listen to `separated_*.wav` for reference-matched playback, and use the raw `component_*.wav` files to observe unprocessed output.
- **FastICA does not converge on custom recordings:** check for silent/duplicate channels, clipping, too few mixtures, linearly dependent mixtures, and highly dependent sources. Increase `--max-iter` only after checking these conditions.

## License

The project source code is offered under the MIT License in `LICENSE`. See `docs/SOURCE_AUDIO_NOTES.md` for how bundled demo audio was generated.
