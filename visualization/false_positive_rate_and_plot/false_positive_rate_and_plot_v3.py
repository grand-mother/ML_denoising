#!/usr/bin/env python3
"""
false_alarm_snr_sweeps_per_method_tau.py

Key change vs previous version:
-------------------------------
For each target P_FA, we calibrate thresholds SEPARATELY for each method:
  tau_in(P_FA)  from T_noise_in   (raw noisy input)
  tau_ml(P_FA)  from T_noise_ml   (ML denoiser output)
  tau_std(P_FA) from T_noise_std  (optional baseline output)

Then we compute P_D(SNR) using each method’s own tau so that all curves correspond
to the same target false-alarm probability. This is the NP/CFAR-consistent “equal-P_FA”
comparison. :contentReference[oaicite:1]{index=1}

Workflow:
1) Run with --make-noise-only-only to create noise_only.npy.
2) Run your denoiser on noise_only.npy to create denoised_noise_only.npy.
3) Rerun this script with --sig-ml and --noise-ml to make the plots.

Outputs:
- tau_vs_pfa__<method>.pdf for each available method
- pd_vs_snr__pfa_<X>.pdf for each P_FA (single plot, multiple method curves)
- .npz dumps per P_FA with binned curves and taus for reproducibility

Dependencies: numpy, scipy, matplotlib
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.stats import beta as beta_dist


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Cfg:
    dt_ns: float = 2.0
    eps: float = 1e-12

    # Signal ROI around CLEAN peak (exclude for sigma estimate + noise harvesting)
    sig_half_width_ns: float = 150.0
    guard_half_width_ns: float = 150.0

    # Noise-only construction
    chunk_len: int = 256
    seed: int = 12345

    # Bandpass
    apply_bandpass: bool = True
    f_lo_hz: float = 50e6
    f_hi_hz: float = 200e6
    butter_order: int = 4

    # Robust sigma from waveform using MAD
    mad_scale_gaussian: float = 1.4826022185056

    # SNR binning (log space)
    snr_min: float = 0.5
    snr_max: float = 50.0
    snr_nbins: int = 12

    # Jeffreys interval for binomial P_D (Beta(0.5,0.5) prior)
    ci_alpha: float = 0.32  # 68% central interval
    # Jeffreys interval details: Beta(k+1/2, n-k+1/2). :contentReference[oaicite:2]{index=2}

    # Plot
    dpi: int = 300
    fontsize: int = 12
    lw: float = 2.0
    grid_alpha: float = 0.25


# -----------------------------
# Validation
# -----------------------------
def _validate_3d(name: str, arr: np.ndarray) -> None:
    if not isinstance(arr, np.ndarray) or arr.ndim != 3:
        raise ValueError(f"{name} must have shape (N,3,T). Got {None if arr is None else arr.shape}")
    if arr.shape[1] != 3:
        raise ValueError(f"{name} must have 3 channels. Got {arr.shape}")


def _validate_pair(a_name: str, a: np.ndarray, b_name: str, b: np.ndarray) -> None:
    _validate_3d(a_name, a)
    _validate_3d(b_name, b)
    if a.shape != b.shape:
        raise ValueError(f"{a_name} and {b_name} must match shape. Got {a.shape} vs {b.shape}")


# -----------------------------
# Noise-only construction
# -----------------------------
def _halfwidth_samples(width_ns: float, dt_ns: float) -> int:
    return max(1, int(round(width_ns / dt_ns)))


def _analytic_envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(x, axis=-1))


def _global_peak_index_from_clean(clean_3ch: np.ndarray) -> int:
    env = _analytic_envelope(clean_3ch)  # (3,T)
    env_max_t = np.max(env, axis=0)     # (T,)
    return int(np.argmax(env_max_t))


def _signal_window_indices(peak_k: int, T: int, hw: int) -> Tuple[int, int]:
    lo = max(0, peak_k - hw)
    hi = min(T, peak_k + hw + 1)
    return lo, hi


def _noise_intervals(T: int, sig_lo: int, sig_hi: int, guard_hw: int) -> List[Tuple[int, int]]:
    ex_lo = max(0, sig_lo - guard_hw)
    ex_hi = min(T, sig_hi + guard_hw)
    intervals: List[Tuple[int, int]] = []
    if ex_lo > 0:
        intervals.append((0, ex_lo))
    if ex_hi < T:
        intervals.append((ex_hi, T))
    return intervals


def build_noise_only_traces(
    clean: np.ndarray,
    noisy: np.ndarray,
    cfg: Cfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _validate_pair("clean", clean, "noisy", noisy)
    N, _, T = clean.shape

    sig_hw = _halfwidth_samples(cfg.sig_half_width_ns, cfg.dt_ns)
    guard_hw = _halfwidth_samples(cfg.guard_half_width_ns, cfg.dt_ns)

    peak_idx = np.zeros((N,), dtype=np.int64)
    sig_lo = np.zeros((N,), dtype=np.int64)
    sig_hi = np.zeros((N,), dtype=np.int64)

    pool: List[Tuple[int, int, int]] = []  # (i, a, b)
    for i in range(N):
        k = _global_peak_index_from_clean(clean[i])
        peak_idx[i] = k
        lo, hi = _signal_window_indices(k, T, sig_hw)
        sig_lo[i], sig_hi[i] = lo, hi
        for (a, b) in _noise_intervals(T, lo, hi, guard_hw):
            if (b - a) >= 1:
                pool.append((i, a, b))

    if len(pool) == 0:
        raise RuntimeError("No admissible noise intervals found. Reduce guard or check CLEAN localization.")

    rng = np.random.default_rng(cfg.seed)
    noise_only = np.empty_like(noisy)

    maxL = max(1, int(cfg.chunk_len))
    pools = {L: [(i, a, b) for (i, a, b) in pool if (b - a) >= L] for L in range(1, maxL + 1)}
    for L in range(1, maxL + 1):
        if len(pools[L]) == 0:
            raise RuntimeError(f"No intervals of length >= {L}. Reduce chunk_len or guard.")

    def sample_chunk(L: int) -> np.ndarray:
        pool_L = pools[L]
        i, a, b = pool_L[int(rng.integers(0, len(pool_L)))]
        start_max = b - L
        start = a if start_max <= a else int(rng.integers(a, start_max + 1))
        return noisy[i, :, start : start + L]

    for n in range(N):
        pos = 0
        while pos < T:
            L = min(cfg.chunk_len, T - pos)
            if L <= maxL:
                chunk = sample_chunk(L)
            else:
                valid = [(i, a, b) for (i, a, b) in pool if (b - a) >= L]
                if len(valid) == 0:
                    raise RuntimeError(f"No interval long enough for tail length {L}.")
                i, a, b = valid[int(rng.integers(0, len(valid)))]
                start_max = b - L
                start = a if start_max <= a else int(rng.integers(a, start_max + 1))
                chunk = noisy[i, :, start : start + L]
            noise_only[n, :, pos : pos + L] = chunk
            pos += L

    return noise_only, peak_idx, sig_lo, sig_hi


# -----------------------------
# Bandpass + robust sigma + scan statistic
# -----------------------------
def _bandpass_sos(cfg: Cfg, fs_hz: float) -> np.ndarray:
    nyq = 0.5 * fs_hz
    lo = cfg.f_lo_hz / nyq
    hi = cfg.f_hi_hz / nyq
    if not (0.0 < lo < hi < 1.0):
        raise ValueError(f"Invalid bandpass: lo={cfg.f_lo_hz}, hi={cfg.f_hi_hz}, fs={fs_hz}")
    return butter(cfg.butter_order, [lo, hi], btype="band", output="sos")


def bandpass_3d(x: np.ndarray, cfg: Cfg) -> np.ndarray:
    if not cfg.apply_bandpass:
        return x.astype(np.float64)
    fs_hz = 1e9 / cfg.dt_ns
    sos = _bandpass_sos(cfg, fs_hz)
    y = np.empty_like(x, dtype=np.float64)
    for ch in range(3):
        y[:, ch, :] = sosfiltfilt(sos, x[:, ch, :], axis=-1)
    return y


def mad_sigma_1d(x: np.ndarray, cfg: Cfg) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return max(cfg.mad_scale_gaussian * mad, cfg.eps)


def sigma_in_from_input(
    x_in_bp: np.ndarray,             # (N,3,T) bandpassed input (raw noisy input, or noise-only input)
    sig_lo: Optional[np.ndarray],    # if provided: exclude pulse+guard
    sig_hi: Optional[np.ndarray],
    cfg: Cfg,
) -> np.ndarray:
    _validate_3d("x_in_bp", x_in_bp)
    N, C, T = x_in_bp.shape
    sigma = np.zeros((N, C), dtype=np.float64)

    for i in range(N):
        if sig_lo is not None and sig_hi is not None:
            guard_hw = _halfwidth_samples(cfg.guard_half_width_ns, cfg.dt_ns)
            ex_lo = max(0, int(sig_lo[i]) - guard_hw)
            ex_hi = min(T, int(sig_hi[i]) + guard_hw)
            idx = np.concatenate([np.arange(0, ex_lo), np.arange(ex_hi, T)])
            if idx.size < max(32, T // 20):
                idx = np.arange(T)
        else:
            idx = np.arange(T)

        for ch in range(C):
            sigma[i, ch] = mad_sigma_1d(x_in_bp[i, ch, idx], cfg)

    return sigma


def scan_statistic(
    y_bp: np.ndarray,     # (N,3,T) bandpassed waveform to score (input, ML output, baseline output)
    sigma_in: np.ndarray, # (N,3) from INPUT (kept fixed across methods)
    cfg: Cfg,
) -> np.ndarray:
    _validate_3d("y_bp", y_bp)
    if sigma_in.shape != (y_bp.shape[0], 3):
        raise ValueError(f"sigma_in must have shape (N,3). Got {sigma_in.shape} for y_bp {y_bp.shape}")

    env = _analytic_envelope(y_bp)     # (N,3,T)
    peak = np.max(env, axis=-1)        # (N,3)
    score_ch = peak / np.maximum(sigma_in, cfg.eps)
    return np.max(score_ch, axis=1)    # conservative max over channels


def snr_from_clean(clean_bp: np.ndarray, sigma_in: np.ndarray, cfg: Cfg) -> np.ndarray:
    env = _analytic_envelope(clean_bp)     # (N,3,T)
    peak = np.max(env, axis=-1)            # (N,3)
    snr_ch = peak / np.maximum(sigma_in, cfg.eps)
    return np.max(snr_ch, axis=1)


# -----------------------------
# Binning + Jeffreys intervals
# -----------------------------
def snr_edges(cfg: Cfg) -> np.ndarray:
    lo = max(cfg.snr_min, 1e-6)
    hi = max(cfg.snr_max, lo * 1.01)
    return np.logspace(np.log10(lo), np.log10(hi), cfg.snr_nbins + 1)


def jeffreys_interval(k: int, n: int, alpha: float) -> Tuple[float, float, float]:
    # p | data ~ Beta(k+0.5, n-k+0.5) :contentReference[oaicite:3]{index=3}
    if n <= 0:
        return np.nan, np.nan, np.nan
    a = k + 0.5
    b = (n - k) + 0.5
    p_hat = k / n
    lo = float(beta_dist.ppf(alpha / 2.0, a, b))
    hi = float(beta_dist.ppf(1.0 - alpha / 2.0, a, b))
    return p_hat, lo, hi


def pd_vs_snr(snr: np.ndarray, detected: np.ndarray, edges: np.ndarray, cfg: Cfg) -> Dict[str, np.ndarray]:
    snr = np.asarray(snr, dtype=np.float64)
    detected = np.asarray(detected, dtype=bool)
    ok = np.isfinite(snr)
    snr = snr[ok]
    detected = detected[ok]

    nb = len(edges) - 1
    x = np.zeros(nb)
    pd = np.zeros(nb)
    lo = np.zeros(nb)
    hi = np.zeros(nb)
    nbin = np.zeros(nb, dtype=np.int64)

    for j in range(nb):
        m = (snr >= edges[j]) & (snr < edges[j + 1])
        n = int(np.sum(m))
        nbin[j] = n
        x[j] = np.sqrt(edges[j] * edges[j + 1])
        if n == 0:
            pd[j], lo[j], hi[j] = np.nan, np.nan, np.nan
            continue
        k = int(np.sum(detected[m]))
        pd[j], lo[j], hi[j] = jeffreys_interval(k, n, cfg.ci_alpha)

    return {"snr_center": x, "pd": pd, "pd_lo": lo, "pd_hi": hi, "n": nbin}


# -----------------------------
# Threshold calibration (per method)
# -----------------------------
def tau_from_pfa(scores_h0: np.ndarray, pfa: float) -> float:
    """
    tau(P_FA) = Quantile_{1-P_FA}(T | H0).
    This enforces the target false-alarm probability for that method’s statistic. :contentReference[oaicite:4]{index=4}
    """
    s = np.asarray(scores_h0, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 100:
        raise RuntimeError(f"Too few finite H0 scores ({s.size}) for stable quantile calibration.")
    return float(np.quantile(s, 1.0 - pfa))


# -----------------------------
# Plotting
# -----------------------------
def setup_matplotlib(cfg: Cfg) -> None:
    plt.rcParams.update({
        "font.size": cfg.fontsize,
        "axes.labelsize": cfg.fontsize,
        "axes.titlesize": cfg.fontsize,
        "legend.fontsize": cfg.fontsize - 1,
    })


def plot_tau_vs_pfa(outpath: str, pfas: np.ndarray, taus: np.ndarray, title: str, cfg: Cfg) -> None:
    fig = plt.figure(figsize=(6.0, 4.6), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(pfas, taus, linewidth=cfg.lw, marker="o", markersize=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Target false-alarm probability $P_{FA}$ (per trace)")
    ax.set_ylabel(r"Threshold $\tau$ on scan statistic $T$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=cfg.grid_alpha)
    fig.savefig(outpath, dpi=cfg.dpi, bbox_inches="tight")
    print(f"[OK] wrote {outpath}")


def plot_pd_vs_snr(outpath: str, pfa: float, curves: Dict[str, Dict[str, np.ndarray]], cfg: Cfg) -> None:
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    ax = fig.add_subplot(111)

    for label, d in curves.items():
        x = d["snr_center"]
        y = d["pd"]
        ylo = d["pd_lo"]
        yhi = d["pd_hi"]
        ax.plot(x, y, linewidth=cfg.lw, marker="o", markersize=10, label=label)
        ax.fill_between(x, ylo, yhi, alpha=0.2)

    # ax.set_xscale("log")
    fontsize = 28
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, 15)
    ax.set_xlabel("Injected SNR",fontsize=fontsize)
    ax.set_ylabel(r"Detection probability",fontsize=fontsize)
    # ax.set_title(rf"$P_D(\mathrm{{SNR}})$ at fixed target $P_{{FA}}={pfa:.0e}$ (per-method $\tau$)")
    ax.grid(True, which="both", alpha=cfg.grid_alpha)
    ax.legend(frameon=False, loc="lower right",fontsize=fontsize)

    fig.savefig(outpath, dpi=cfg.dpi, bbox_inches="tight")
    print(f"[OK] wrote {outpath}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="P_D(SNR) at fixed P_FA with per-method threshold calibration.")
    ap.add_argument("--clean", required=True, help="clean waveforms (N,3,T) .npy")
    ap.add_argument("--noisy", required=True, help="noisy waveforms (N,3,T) .npy")

    ap.add_argument("--sig-ml", default=None, help="denoiser output on signal-present noisy (N,3,T) .npy")
    ap.add_argument("--noise-ml", default=None, help="denoiser output on noise-only traces (N,3,T) .npy")

    ap.add_argument("--sig-std", default=None, help="optional baseline output on signal-present (N,3,T) .npy")
    ap.add_argument("--noise-std", default=None, help="optional baseline output on noise-only (N,3,T) .npy")

    ap.add_argument("--noise-only-out", default="noise_only.npy")
    ap.add_argument("--noise-meta-out", default="noise_only_meta.npz")
    ap.add_argument("--make-noise-only-only", action="store_true")

    ap.add_argument("--pfa", nargs="+", type=float, default=[1e-2, 1e-3, 1e-4],
                    help="list of target P_FA values (per trace)")

    # Settings
    ap.add_argument("--dt-ns", type=float, default=2.0)
    ap.add_argument("--sig-half-width-ns", type=float, default=150.0)
    ap.add_argument("--guard-half-width-ns", type=float, default=150.0)
    ap.add_argument("--chunk-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--no-bandpass", action="store_true")
    ap.add_argument("--f-lo-hz", type=float, default=50e6)
    ap.add_argument("--f-hi-hz", type=float, default=200e6)
    ap.add_argument("--butter-order", type=int, default=4)

    ap.add_argument("--snr-min", type=float, default=0.5)
    ap.add_argument("--snr-max", type=float, default=50.0)
    ap.add_argument("--snr-nbins", type=int, default=12)

    ap.add_argument("--outdir", default="fa_snr_outputs_per_method_tau")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    cfg = Cfg(
        dt_ns=float(args.dt_ns),
        sig_half_width_ns=float(args.sig_half_width_ns),
        guard_half_width_ns=float(args.guard_half_width_ns),
        chunk_len=int(args.chunk_len),
        seed=int(args.seed),
        apply_bandpass=(not args.no_bandpass),
        f_lo_hz=float(args.f_lo_hz),
        f_hi_hz=float(args.f_hi_hz),
        butter_order=int(args.butter_order),
        snr_min=float(args.snr_min),
        snr_max=float(args.snr_max),
        snr_nbins=int(args.snr_nbins),
        dpi=int(args.dpi),
    )

    os.makedirs(args.outdir, exist_ok=True)
    setup_matplotlib(cfg)

    clean = np.load(args.clean)
    noisy = np.load(args.noisy)
    _validate_pair("clean", clean, "noisy", noisy)

    # (A) Build noise-only and save
    noise_only, peak_idx, sig_lo, sig_hi = build_noise_only_traces(clean, noisy, cfg)
    np.save(args.noise_only_out, noise_only)
    np.savez(
        args.noise_meta_out,
        peak_idx=peak_idx,
        sig_lo=sig_lo,
        sig_hi=sig_hi,
        dt_ns=cfg.dt_ns,
        sig_half_width_ns=cfg.sig_half_width_ns,
        guard_half_width_ns=cfg.guard_half_width_ns,
        chunk_len=cfg.chunk_len,
        seed=cfg.seed,
    )
    print(f"[OK] wrote {args.noise_only_out} shape={noise_only.shape}")
    print(f"[OK] wrote {args.noise_meta_out}")

    if args.make_noise_only_only:
        print("[INFO] Exiting due to --make-noise-only-only. Now denoise noise_only.npy and rerun with --noise-ml.")
        return

    if args.sig_ml is None or args.noise_ml is None:
        raise ValueError(
            "Need --sig-ml and --noise-ml to compute P_D(SNR) curves.\n"
            "Workflow: run once with --make-noise-only-only, denoise noise_only.npy, then rerun."
        )

    sig_ml = np.load(args.sig_ml)
    noise_ml = np.load(args.noise_ml)
    _validate_pair("noisy", noisy, "sig_ml", sig_ml)
    _validate_pair("noise_only", noise_only, "noise_ml", noise_ml)

    sig_std = noise_std = None
    if args.sig_std is not None and args.noise_std is not None:
        sig_std = np.load(args.sig_std)
        noise_std = np.load(args.noise_std)
        _validate_pair("noisy", noisy, "sig_std", sig_std)
        _validate_pair("noise_only", noise_only, "noise_std", noise_std)

    # Bandpass everything
    clean_bp = bandpass_3d(clean, cfg)
    noisy_bp = bandpass_3d(noisy, cfg)
    sig_ml_bp = bandpass_3d(sig_ml, cfg)
    noise_only_bp = bandpass_3d(noise_only, cfg)
    noise_ml_bp = bandpass_3d(noise_ml, cfg)

    sig_std_bp = noise_std_bp = None
    if sig_std is not None and noise_std is not None:
        sig_std_bp = bandpass_3d(sig_std, cfg)
        noise_std_bp = bandpass_3d(noise_std, cfg)

    # sigma_in from INPUT (kept fixed across methods)
    sigma_sig_in = sigma_in_from_input(noisy_bp, sig_lo=sig_lo, sig_hi=sig_hi, cfg=cfg)
    sigma_noise_in = sigma_in_from_input(noise_only_bp, sig_lo=None, sig_hi=None, cfg=cfg)

    # Scores on signal-present
    T_sig_in = scan_statistic(noisy_bp, sigma_sig_in, cfg)
    T_sig_ml = scan_statistic(sig_ml_bp, sigma_sig_in, cfg)
    T_sig_std = scan_statistic(sig_std_bp, sigma_sig_in, cfg) if sig_std_bp is not None else None

    # Scores on noise-only (H0)
    T_noise_in = scan_statistic(noise_only_bp, sigma_noise_in, cfg)
    T_noise_ml = scan_statistic(noise_ml_bp, sigma_noise_in, cfg)
    T_noise_std = scan_statistic(noise_std_bp, sigma_noise_in, cfg) if noise_std_bp is not None else None

    # SNR per trace from CLEAN peak / sigma_in
    SNR = snr_from_clean(clean_bp, sigma_sig_in, cfg)
    edges = snr_edges(cfg)

    # Per-method tau(P_FA)
    pfas = np.array(sorted(set(float(p) for p in args.pfa)), dtype=np.float64)

    taus_in = np.array([tau_from_pfa(T_noise_in, p) for p in pfas], dtype=np.float64)
    taus_ml = np.array([tau_from_pfa(T_noise_ml, p) for p in pfas], dtype=np.float64)
    taus_std = None
    if T_noise_std is not None:
        taus_std = np.array([tau_from_pfa(T_noise_std, p) for p in pfas], dtype=np.float64)

    # Plot tau vs pfa per method
    plot_tau_vs_pfa(
        os.path.join(args.outdir, "tau_vs_pfa__noisy_input.pdf"),
        pfas, taus_in,
        r"Threshold calibration from noise-only (Noisy input)",
        cfg
    )
    plot_tau_vs_pfa(
        os.path.join(args.outdir, "tau_vs_pfa__ml_denoiser.pdf"),
        pfas, taus_ml,
        r"Threshold calibration from noise-only (ML denoiser)",
        cfg
    )
    if taus_std is not None:
        plot_tau_vs_pfa(
            os.path.join(args.outdir, "tau_vs_pfa__standard.pdf"),
            pfas, taus_std,
            r"Threshold calibration from noise-only (Standard)",
            cfg
        )

    # Save taus table
    cols = [pfas, taus_in, taus_ml]
    header = "PFA  tau_noisy_input  tau_ml_denoiser"
    if taus_std is not None:
        cols.append(taus_std)
        header += "  tau_standard"
    np.savetxt(
        os.path.join(args.outdir, "taus_per_method.txt"),
        np.column_stack(cols),
        header=header,
    )
    print(f"[OK] wrote {os.path.join(args.outdir, 'taus_per_method.txt')}")

    # For each P_FA, compute and plot P_D(SNR) using each method's own tau
    for i, pfa in enumerate(pfas):
        det_in = (T_sig_in >= taus_in[i])
        det_ml = (T_sig_ml >= taus_ml[i])

        curves = {
            "Noisy input": pd_vs_snr(SNR, det_in, edges, cfg),
            "ML denoiser": pd_vs_snr(SNR, det_ml, edges, cfg),
        }

        if T_sig_std is not None and taus_std is not None:
            det_std = (T_sig_std >= taus_std[i])
            curves["Standard"] = pd_vs_snr(SNR, det_std, edges, cfg)

        tag = f"pfa_{pfa:.0e}".replace("+", "")
        out_pdf = os.path.join(args.outdir, f"pd_vs_snr__{tag}.pdf")
        plot_pd_vs_snr(out_pdf, float(pfa), curves, cfg)

        out_npz = os.path.join(args.outdir, f"pd_vs_snr__{tag}.npz")
        np.savez(
            out_npz,
            pfa=float(pfa),
            tau_noisy=float(taus_in[i]),
            tau_ml=float(taus_ml[i]),
            tau_std=(float(taus_std[i]) if taus_std is not None else np.nan),
            snr_edges=edges,
            **{f"{k}__{kk}": vv for k, d in curves.items() for kk, vv in d.items()},
        )
        print(f"[OK] wrote {out_npz}")

    print("[DONE] Generated per-P_FA P_D(SNR) plots with per-method threshold calibration.")


if __name__ == "__main__":
    main()

# How to run?
#1) Generate noise-only
# python false_positive_*V3.py \
#   --clean clean_waveforms.npy \
#   --noisy noisy_waveforms.npy \
#   --make-noise-only-only \
#   --noise-only-out noise_only.npy

# 2) Run your denoiser externally
# noisy_waveforms.npy -> denoised_signal.npy
# noise_only.npy -> denoised_noise_only.npy

#3) Make plots
# python false_positive_*V3.py \
#   --clean clean_waveforms.npy \
#   --noisy noisy_waveforms.npy \
#   --sig-ml denoised_signal.npy \
#   --noise-ml denoised_noise_only.npy \
#   --pfa 1e-2 1e-3 1e-4 \
#   --outdir fa_snr_outputs_per_method_tau

# Please share the following files afterwards:
# tau_vs_pfa__noisy_input.pdf
# tau_vs_pfa__ml_denoiser.pdf
# pd_vs_snr__pfa_1e-02.pdf etc.
# taus_per_method.txt
