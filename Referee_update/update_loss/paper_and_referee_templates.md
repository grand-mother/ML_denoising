# Paper and referee-response templates

Use only the branch supported by the phase audit.

## 1. Phase reply if the audit shows a negligible boundary effect

```latex
\reply \om{We thank the referee for pointing out this distinction. The
model used in the original analysis minimized an $L_1$ difference between
the principal Fourier phases returned by the real Fourier transform; the
residual was not explicitly wrapped across the $-\pi/\pi$ boundary. We
have clarified this implementation in the Methods section.

We also evaluated the direct and wrapped phase residuals on the validation
sample. The fraction of Fourier coefficients satisfying
$|\Delta\phi_{\rm raw}|>\pi$ is \OM{insert value}. Replacing the direct
residual with the shortest circular residual changes the mean phase term
from \OM{insert value} to \OM{insert value}; after applying the selected
phase-loss weight, the change in the total validation objective is
\OM{insert value}. We therefore find that the branch-boundary convention
has \OM{insert measured practical effect} for the trained model. No model
or performance claim relies on treating the original loss as wrapped.}
```

Corresponding Methods text:

```latex
For the Fourier-domain phase term, we used the principal phases
$\phi_{\rm true}(f)=\arg\widetilde{x}_{\rm true}(f)$ and
$\phi_{\rm rec}(f)=\arg\widetilde{x}_{\rm rec}(f)$ and minimized
\begin{equation}
L_{\rm phase}
=
\frac{1}{N_f}\sum_f
\left|\phi_{\rm rec}(f)-\phi_{\rm true}(f)\right|.
\end{equation}
This implementation does not explicitly wrap the residual across the
$-\pi/\pi$ boundary. On the validation sample, \OM{insert concise audit
result}, showing that \OM{insert measured interpretation}.
```

## 2. Phase reply if one fixed-config retraining is required

```latex
\reply \om{We thank the referee for identifying this issue. The original
implementation compared the principal Fourier phases directly and did not
account for the branch boundary at $\pm\pi$. We corrected the residual to
\[
\Delta\phi(f)=\operatorname{atan2}\!\left[
\sin\!\left(\phi_{\rm rec}(f)-\phi_{\rm true}(f)\right),
\cos\!\left(\phi_{\rm rec}(f)-\phi_{\rm true}(f)\right)
\right],
\]
which returns the shortest signed angular separation. We retrained the
selected model once using the same data split, architecture, optimizer
settings, and spectral-loss weights; no new hyperparameter search was
performed. The corrected model gives \OM{insert concise comparison of the
principal validation and test metrics}. The Methods section and the
reported results have been updated accordingly.}
```

## 3. Time-only comparison

If the two models did not use the same validation split, remove the best-validation-loss comparison. Use the common-test evaluation instead:

```latex
\reply \om{We agree that the Hilbert-envelope comparison does not isolate
the contribution of the Fourier branch. We therefore evaluated the
fiducial time-plus-frequency network against a time-only autoencoder on
the same held-out traces, using identical preprocessing and metric code.
On this common test sample, the mean PSNR is \OM{time-only value} and
\OM{dual-branch value}\,dB, while the mean fractional peak-amplitude error
is \OM{time-only value} and \OM{dual-branch value}, respectively. These
results show a \OM{modest/clear, according to the measured result}
improvement from the Fourier inputs. We report this comparison in the
Appendix.}
```

## 4. Architecture and training description

The paper must describe the production model rather than changing the model to match the draft. After the production audit, correct the following statements as needed:

- replace 1,024 samples with the production trace length;
- replace gradient clipping at 5 with the production value;
- remove the Hann-taper statement unless the archived production model used it;
- describe two max-pooling operations per branch if that is the production architecture;
- list the effective fusion and decoder widths, not an overwritten nominal first decoder channel;
- fill Table I from the archived best-trial configuration.

No new calculation is required for these corrections.

## 5. Amplitude-bias caption

```latex
\caption{\textbf{Peak-amplitude bias versus input SNR.}
Binned median signed bias,
$\delta_A=A_{\rm rec}/A_{\rm true}-1$, for trigger-passing traces in the
X, Y, and Z polarization channels. Shaded bands show the central
\OM{insert exact percentage} population interval in each SNR bin. The
same selected traces and input-SNR values are used for the noisy and
denoised estimators. At low SNR, conditioning on the noisy-input maximum
biases the raw envelope estimate high, while the denoised estimate is
biased low below SNR $\sim4$, most clearly in the X and Y channels.}
```
