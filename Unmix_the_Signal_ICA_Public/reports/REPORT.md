# Unmix the Signal (ICA) — Technical Report

## Abstract

This project demonstrates blind audio source separation from three linearly mixed audio channels. It applies channel centering, principal-component-analysis (PCA) whitening, and a symmetric FastICA fixed-point algorithm implemented directly with NumPy. The experiment records separated audio to WAV, evaluates recovery against known clean demo sources, compares the result with PCA whitening alone, and tests sensitivity to additive white noise.

## 1. Problem definition

Let the clean independent source signals be rows of the matrix \(S \in \mathbb{R}^{N\times T}\), where \(N=3\) sources and \(T\) samples. The observed mixtures are

\[
X = A S,
\]

where \(A\in\mathbb{R}^{N\times N}\) is an unknown-to-the-separation-algorithm, full-rank mixing matrix. The source-separation code receives only \(X\); it does not receive \(S\) or \(A\). In this controlled evaluation, the originals and mixing matrix are saved so performance can be measured afterward.

The sources are 7-second mono clips at 8 kHz: eSpeak-synthesized speech, procedurally generated instrumental tones, and procedurally generated rain ambience. These are reproducible demonstration sources, not field recordings.

The experiment uses the following base mixing matrix:

\[
A = \begin{bmatrix}
1.00 & 0.45 & 0.20\\
0.25 & 1.00 & 0.50\\
0.40 & 0.20 & 1.00
\end{bmatrix}.
\]

A common gain is applied to all mixture channels to keep the WAV output below clipping. This common gain does not change the source-separation problem.

## 2. Centering

For each observed channel \(i\), its mean is removed:

\[
X_c = X - \mu, \qquad \mu_i = \frac{1}{T}\sum_{t=1}^{T} X_{i,t}.
\]

Centering removes constant offsets and simplifies the covariance calculation. It is implemented in `center_signals` in `src/unmix_ica/ica.py`.

## 3. PCA whitening

Compute the covariance matrix of the centered mixtures:

\[
C = \frac{1}{T}X_c X_c^T.
\]

Using eigendecomposition, \(C = E D E^T\), where columns of \(E\) are eigenvectors and \(D\) is the diagonal matrix of eigenvalues. The whitening transform is

\[
Z = D^{-1/2} E^T X_c.
\]

Then \(\operatorname{Cov}(Z)\approx I\). Whitening removes second-order correlation and standardizes variance. It does not generally make components statistically independent; this is why ICA is still necessary. The code implements the covariance and eigendecomposition directly and checks for rank-deficient input.

## 4. FastICA from scratch

The code uses symmetric FastICA with the nonlinear function \(g(u)=\tanh(\alpha u)\). For a current unmixing matrix \(W\), the fixed-point update is

\[
W_{new} = \mathbb{E}[g(WZ)Z^T] - \operatorname{diag}(\mathbb{E}[g'(WZ)])W.
\]

Each iteration then symmetrically decorrelates the rows:

\[
W \leftarrow (WW^T)^{-1/2}W.
\]

The convergence statistic checks whether corresponding rows of consecutive weight matrices align up to sign. A sign flip is not considered a meaningful change because ICA cannot determine a component's sign uniquely. The recovered components are \(Y=WZ\).

The implementation uses NumPy for matrix products and eigendecomposition, but it does not call a ready-made ICA routine from scikit-learn or another package. `scipy.optimize.linear_sum_assignment` is used only during evaluation to match output components to known references.

## 5. Evaluation methodology

ICA output has inherent permutation, sign, and scale ambiguities. For the controlled benchmark, the evaluation code matches estimates to original sources by maximizing absolute Pearson correlation with a Hungarian assignment. It flips the matched waveform if its correlation is negative and fits a least-squares scalar for the demo WAV output. The algorithm itself never uses this reference information during fitting.

Two metrics are reported:

- **Pearson correlation:** measures waveform similarity after source matching. The absolute value is useful when initially choosing matches because source sign can be reversed.
- **Scale-invariant signal-to-distortion ratio (SI-SDR):** projects the estimate onto the reference and compares target energy to residual error energy. Higher is better; the metric is designed to be invariant to a constant amplitude scaling.

The experiment compares the full method with PCA whitening alone. It also injects white noise at nominal input SNR levels of 30, 20, 10, and 5 dB, then reruns ICA and evaluates the estimates.

## 6. Results

### Main experiment

| Metric | PCA whitening only | PCA whitening + ICA |
|---|---:|---:|
| Mean absolute correlation | 0.7042 | 0.9999 |
| Mean SI-SDR | 0.09 dB | 39.84 dB |
| FastICA iterations | — | 5 |
| Converged | — | Yes |

Per-source ICA results after matching:

| Reference source | Correlation | SI-SDR |
|---|---:|---:|
| eSpeak-synthesized speech | 1.0000 | 41.47 dB |
| Synthetic instrumental music | 0.9999 | 38.43 dB |
| Synthetic rain ambience | 0.9999 | 39.62 dB |

### Noise-robustness experiment

| Input mixture SNR | Mean absolute correlation | Mean SI-SDR |
|---:|---:|---:|
| 30 dB | 0.9990 | 27.08 dB |
| 20 dB | 0.9907 | 17.24 dB |
| 10 dB | 0.9197 | 7.41 dB |
| 5 dB | 0.7991 | 2.48 dB |

The noise experiment shows the expected trend: quality degrades as the input mixture becomes noisier. The main separation result is unusually strong because the example uses synthetic sources, a known full-rank instantaneous mixture, and no room reverberation. It should not be interpreted as a real-world benchmark or a guarantee for arbitrary sounds.

## 7. Deliverables and reproducibility

- `data/originals/`: clean reference sources.
- `data/mixed/`: the three mixed observations used as algorithm inputs.
- `outputs/separated/separated_1.wav` through `separated_3.wav`: source-matched and scaled demo outputs for easy listening.
- `outputs/separated/component_1_raw.wav` through `component_3_raw.wav`: playable components in ICA's natural order (with WAV-safe peak scaling only if required). `components_raw.npy` stores exact numeric estimates before audio export scaling.
- `reports/metrics/`: CSV/JSON data, source assignment, mixing matrix, baseline and SNR results.
- `reports/figures/`: time-domain comparisons, correlation matrices, baseline comparison, convergence history and SNR robustness.
- `notebooks/unmix_signal_ica.ipynb`: a narrative walkthrough with runnable code.
- `tests/`: automated checks for centering, whitening, source recovery, mixing validation, WAV I/O and normalization.

Run `python scripts/run_experiment.py` to recreate outputs, then `python -m pytest -q` to verify the implementation.

## 8. Real-world multi-device workflow

The repository now includes `scripts/run_device_recordings.py` and `src/unmix_ica/realworld.py`. The workflow accepts WAV files recorded simultaneously by separate devices, converts them to a common sample rate, estimates start offsets by cross-correlation, trims to common overlap, runs the same from-scratch FastICA implementation, and exports unlabelled component WAVs, exact numeric arrays, correlation diagnostics, excess-kurtosis statistics, convergence information, and plots.

This software path is covered by automated tests using delayed multichannel fixtures. **Those test fixtures are generated for software testing; they are not real-world recordings.** No genuine multiple-device recording session has been captured in this software-only environment, so this report does not claim measured performance on physical devices. Follow `docs/REAL_WORLD_RECORDING_GUIDE.md` to make that capture and run the pipeline.

When clean isolated reference signals are unavailable, correlation-to-reference and SI-SDR should not be reported. For a real session, inspect the synchronized inputs and output component WAVs, convergence, component correlations, and subjective listening. If a known source is played through a loudspeaker, reference-aware metrics are possible only after appropriate time alignment and consideration of the room/speaker transfer functions.

## 9. Limitations and next steps

1. **Capture and inspect genuine multi-device recordings.** The end-to-end code and recording procedure are present, but the physical recording itself remains to be done by the project owner.
2. The cross-correlation alignment estimates starting offsets; it does not compensate for sample-clock drift throughout long files.
3. Evaluate reverberant/convolutive mixtures, because instantaneous ICA is not designed to undo arbitrary room impulse responses.
4. Investigate multiple initializations and alternative contrast functions if a source is difficult to separate.
5. Study the impact of the number of mixtures, source dependence, and rank deficiency.
6. Add an optional interactive player or web UI only if time remains, while keeping the core method independently testable.

## Conclusion

The submission implements centering, PCA whitening, and FastICA from scratch using standard numerical linear algebra. It produces playable output audio, evaluates the controlled benchmark against known originals, compares ICA with whitening alone, and includes a reproducible SNR robustness experiment. It also includes a tested software path for real multi-device recordings. The controlled-demo results validate the algorithm only under the demo's assumptions; physical multi-device performance remains to be measured after a real capture session.
