

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pywt  # PyWavelets – preferred over scipy.signal.morlet

# -----------------------------------------------------------------------------
# ⚙️  Configuration & constants
# -----------------------------------------------------------------------------
PAGE_TITLE = "Morlet + Step Wave Signal Playground (PyWavelets edition)"
DEFAULT_DURATION = 10  # [s]
DEFAULT_FS = 1_000      # [Hz]
DEFAULT_W = 6.0         # Morlet central frequency reference (ignored by PyWavelets)

# -----------------------------------------------------------------------------
# 🛠️  Utility functions – cached for snappy UI
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def time_vector(duration_s: float, fs: int) -> np.ndarray:
    """Return an evenly‑spaced time vector of length *duration_s × fs*."""
    return np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)


@st.cache_data(show_spinner=False)
def morlet_template(width: int, amp: float, w: float) -> np.ndarray:
    """Return a real Morlet wavelet of *width* samples and amplitude *amp*.

    Notes
    -----
    * PyWavelets' continuous wavelet `morl` does **not** expose the ω₀ parameter
      that SciPy's `signal.morlet` has.  The *w* slider is therefore treated as a
      *qualitative* knob that slightly stretches/compresses the wavelet in the
      time domain so the UI still feels responsive.
    """
    cw = pywt.ContinuousWavelet("morl")
    # `ContinuousWavelet.wavefun` returns **two** outputs: psi, x
    psi, _x = cw.wavefun(level=10)  # ≈ 4k samples

    # Normalise amplitude
    psi = psi / np.max(np.abs(psi)) * amp

    # --- Cheap way to emulate ω scaling: simple resample in time domain -------
    stretch = w / DEFAULT_W  # 1.0 ⇒ no stretch
    if stretch != 1.0:
        # Oversample / undersample by linear interpolation and then trim
        new_len = int(len(psi) / stretch)
        new_x = np.linspace(0, len(psi) - 1, new_len)
        psi = np.interp(new_x, np.arange(len(psi)), psi)

    # Finally down‑sample/upsample to the requested *width*
    idx = np.linspace(0, len(psi) - 1, width).astype(int)
    return psi[idx]


@st.cache_data(show_spinner=False)
def make_morlet(t: np.ndarray, fs: int, width: int, amp: float, interval_ms: int, w: float) -> np.ndarray:
    """Return a pulse train of Morlet bursts."""
    template = morlet_template(width, amp, w)
    pulse = np.zeros_like(t)
    step = int((interval_ms / 1000) * fs)
    for i in range(0, len(t) - width, step):
        pulse[i : i + width] += template
    return pulse


@st.cache_data(show_spinner=False)
def make_step(t: np.ndarray, bumps: int, height: float) -> np.ndarray:
    """Generate a blocky *step* train (square bumps)."""
    step_signal = np.zeros_like(t)
    if bumps == 0:
        return step_signal

    bump_w = len(t) // (bumps * 2)  # half‑duty‑cycle square wave
    for i in range(bumps):
        s = i * 2 * bump_w
        e = s + bump_w
        if e < len(t):
            step_signal[s:e] = height
    return step_signal


# -----------------------------------------------------------------------------
# 📏 SNR utility
# -----------------------------------------------------------------------------

def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Return Signal‑to‑Noise Ratio in dB."""
    p_signal = np.mean(signal ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise == 0:
        return np.inf
    return 10 * np.log10(p_signal / p_noise)


# -----------------------------------------------------------------------------
# 🚀  Main Streamlit application
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title("🧩 Morlet + Step Wave Signal Playground (PyWavelets)")

    # ---------------------------------------------------------------------
    # ⏱️  SIDEBAR – user controls
    # ---------------------------------------------------------------------
    with st.sidebar:
        st.header("🎛️ Controls")

        # ── Timebase ─────────────────────────────────────
        with st.expander("⏲️ Timebase", expanded=True):
            duration = st.slider("Duration (s)", 1, 10, DEFAULT_DURATION)
            sampling_rate = st.number_input(
                "Sampling rate (Hz)", 100, 5_000, DEFAULT_FS, step=100
            )
            nyquist = sampling_rate / 2
            st.markdown(f"Δt = **{1 / sampling_rate:.3f} s**   |   Nyquist = {nyquist:.0f} Hz")

        # ── Morlet (noise) ───────────────────────────────
        with st.expander("🌊 Morlet burst – noise"):
            wavelet_width = st.slider("Window (samples)", 20, 500, 200)
            amplitude = st.slider("Amplitude", 0.1, 10.0, 5.0)
            interval_ms = st.slider("Interval between bursts (ms)", 20, 1_000, 100)
            morlet_w = st.slider("Relative width (ω)", 1.0, 10.0, DEFAULT_W)

        # ── Step (target) ────────────────────────────────
        with st.expander("▢ Step train – target signal"):
            step_height = st.slider("Step height", 0.1, 10.0, 1.0)
            bump_count = st.slider("Number of bumps", 1, 100, 40)

    # ---------------------------------------------------------------------
    # 🔄  Generate signals
    # ---------------------------------------------------------------------
    t = time_vector(duration, sampling_rate)
    noise_sig = make_morlet(t, sampling_rate, wavelet_width, amplitude, interval_ms, morlet_w)
    step_sig = make_step(t, bump_count, step_height)
    combo_sig = noise_sig + step_sig

    # Raw SNR
    raw_snr = snr_db(step_sig, combo_sig - step_sig)

    # ---------------------------------------------------------------------
    # 🗂️  Tabs
    # ---------------------------------------------------------------------
    tab_time, tab_spec = st.tabs(["⏱ Time domain", "⚡ Spectrum & Filter"])

    # ── Time‑domain view ────────────────────────────────────────────────
    with tab_time:
        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(t, noise_sig, lw=0.8)
        axs[0].set_ylabel("Morlet")
        axs[1].plot(t, step_sig, color="orange", lw=0.8)
        axs[1].set_ylabel("Step")
        axs[2].plot(t, combo_sig, color="green", lw=0.8)
        axs[2].set_ylabel("Combined")
        for ax in axs:
            ax.grid(alpha=0.3)
        axs[-1].set_xlabel("Time (s)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Use the sidebar to tweak parameters and watch the waveforms morph in real‑time.")

    # ── Spectrum + Filter view ──────────────────────────────────────────
    with tab_spec:
        # FFT (single‑sided)
        fft_vals = np.fft.fft(combo_sig)
        freqs = np.fft.fftfreq(len(combo_sig), d=1 / sampling_rate)
        pos = freqs >= 0
        mags = np.abs(fft_vals[pos])

        low_cut, high_cut = st.slider(
            "Pass‑band (Hz)",
            min_value=0.1,
            max_value=float(nyquist),
            value=(0.1, min(10.0, nyquist)),
            step=0.1,
        )

        # Plot spectrum
        fig_f, ax_f = plt.subplots(figsize=(10, 3))
        ax_f.plot(freqs[pos], mags, lw=0.8)
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel("|X(f)|")
        ax_f.grid(alpha=0.3)
        ax_f.set_title("Magnitude Spectrum (positive freqs)")
        st.plotly_chart(fig_f, use_container_width=True)

        # Band‑pass filter in the frequency domain
        mask = (np.abs(freqs) >= low_cut) & (np.abs(freqs) <= high_cut)
        recon = np.fft.ifft(fft_vals * mask).real
        mse = np.mean((recon - step_sig) ** 2)
        filt_snr = snr_db(step_sig, recon - step_sig)

        # Reconstruction plot
        fig_r, ax_r = plt.subplots(figsize=(10, 4))
        ax_r.plot(t, step_sig, label="Target (step)", color="orange", lw=1)
        ax_r.plot(t, recon + 0.05, label="Recovered", color="blue", lw=0.9)
        ax_r.plot(t, combo_sig, label="Combined", color="gray", alpha=0.4, lw=0.8)
        ax_r.set_xlabel("Time (s)")
        ax_r.set_title(f"Band‑pass reconstruction | MSE: {mse:.3e}")
        ax_r.grid(alpha=0.3)
        ax_r.legend()
        st.plotly_chart(fig_r, use_container_width=True)

        # Metrics
        met1, met2 = st.columns(2)
        met1.metric("Raw SNR", f"{raw_snr:.2f} dB")
        met2.metric("Filtered SNR", f"{filt_snr:.2f} dB")

      

if __name__ == "__main__":
    main()
