########################################
# Morlet + Step Wave Signal Playground
# --------------------------------------
# UX‑focused rewrite — April 2025 (v2)
# • Controls live in sidebar ▶️
# • Tabs: Time domain | Spectrum & Filter combined
# • Multiselect lets you null individual dominant peaks
########################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import morlet

st.set_page_config(page_title="Morlet + Step Combiner", layout="wide")

st.title("🧩 Morlet + Step Wave Signal Playground")
st.caption("Tweak parameters in the sidebar and watch the signal evolve in real‑time.")

# -----------------------------------------------------------------------------
# ⏱️ SIDEBAR – all user controls live here
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Controls")

    # ── Timebase ─────────────────────────────────────
    with st.expander("⏲️ Timebase", expanded=True):
        duration = st.slider("Duration (s)", 1, 10, 10)
        sampling_rate = st.number_input("Sampling rate (Hz)", 100, 5000, 1000, step=100)
        nyquist = sampling_rate / 2
        st.markdown(f"Δt = **{1/sampling_rate:.3f} s**   |   Nyquist = {nyquist:.0f} Hz")

    # ── Morlet (noise) ───────────────────────────────
    with st.expander("🌊 Morlet burst – noise"):
        wavelet_width = st.slider("Window (samples)", 20, 500, 200)
        amplitude = st.slider("Amplitude", 0.1, 10.0, 5.0)
        interval_ms = st.slider("Interval between bursts (ms)", 20, 1000, 100)
        morlet_w = st.slider("Central frequency ω", 1.0, 10.0, 6.0)

    # ── Step (target) ────────────────────────────────
    with st.expander("▢ Step train – target signal"):
        step_height = st.slider("Step height", 0.1, 10.0, 1.0)
        bump_count = st.slider("Number of bumps", 1, 100, 40)

    # ── Filter ───────────────────────────────────────
    with st.expander("🎚️ Band‑pass filter"):
        low_cut, high_cut = st.slider(
            "Pass‑band (Hz)",
            min_value=0.1,
            max_value=float(nyquist),
            value=(0.1, min(10.0, nyquist)),
            step=0.1,
        )

# -----------------------------------------------------------------------------
# 🛠️ Helper functions – cached for snappy UI
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def time_vector(duration_s: float, fs: int) -> np.ndarray:
    """Return an evenly‑spaced time vector."""
    return np.linspace(0, duration_s, int(duration_s * fs))

@st.cache_data(show_spinner=False)
def make_morlet(t, fs, width, amp, interval_ms, w):
    template = morlet(M=width, w=w).real * amp
    pulse = np.zeros_like(t)
    step = int((interval_ms / 1000) * fs)
    for i in range(0, len(t) - width, step):
        pulse[i:i + width] += template
    return pulse

@st.cache_data(show_spinner=False)
def make_step(t, bumps, height):
    step_signal = np.zeros_like(t)
    if bumps == 0:
        return step_signal
    bump_w = len(t) // (bumps * 2)
    for i in range(bumps):
        s, e = i * 2 * bump_w, i * 2 * bump_w + bump_w
        if e < len(t):
            step_signal[s:e] = height
    return step_signal

# -----------------------------------------------------------------------------
# 🔄 Generate signals based on controls
# -----------------------------------------------------------------------------
T = time_vector(duration, sampling_rate)
noise_sig = make_morlet(T, sampling_rate, wavelet_width, amplitude, interval_ms, morlet_w)
step_sig = make_step(T, bump_count, step_height)
combo_sig = noise_sig + step_sig

# -----------------------------------------------------------------------------
# 🗂️ TABS – separate visual concerns
# -----------------------------------------------------------------------------
_time_tab, _spec_tab = st.tabs(["⏱ Time domain", "⚡ Spectrum & Filter"])

# ── Time‑domain view ──────────────────────────────────────────────────────────
with _time_tab:
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(T, noise_sig, lw=.8); axs[0].set_ylabel("Morlet")
    axs[1].plot(T, step_sig, color="orange", lw=.8); axs[1].set_ylabel("Step")
    axs[2].plot(T, combo_sig, color="green", lw=.8); axs[2].set_ylabel("Combined")
    for ax in axs: ax.grid(alpha=.3)
    axs[-1].set_xlabel("Time (s)")
    st.pyplot(fig)
    st.info("Use the sidebar to tweak parameters and watch the waveforms morph in real‑time.")

# ── Spectrum + Filter view ────────────────────────────────────────────────────
with _spec_tab:
    # FFT once per user change
    fft_vals = np.fft.fft(combo_sig)
    freqs = np.fft.fftfreq(len(combo_sig), d=1 / sampling_rate)
    pos = freqs >= 0
    mags = np.abs(fft_vals[pos])

    # Top‑5 peaks (excluding DC)
    peak_idx = mags.argsort()[-6:][::-1]
    dom_freqs = [round(freqs[pos][i], 2) for i in peak_idx if freqs[pos][i] > 0][:5]

    # ----- Layout: spectrum plot on the left, controls on right -----
    spec_col, ctrl_col = st.columns([3, 1])

    with spec_col:
        fig_f, ax_f = plt.subplots(figsize=(10, 3))
        ax_f.plot(freqs[pos], mags, lw=.8)
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel("|X(f)|")
        ax_f.grid(alpha=.3)
        ax_f.set_title("Magnitude Spectrum (positive freqs)")
        st.pyplot(fig_f)

    with ctrl_col:
        st.markdown("### 🎚 Filter settings")
        st.markdown("#### 🔎 Dominant peaks")
        if dom_freqs:
            suppress = st.multiselect(
                "Frequencies to suppress (Hz)", dom_freqs, default=[]
            )
        else:
            suppress = []
            st.write("No peaks found")
        st.markdown("---")
        st.markdown(f"**Pass‑band:** {low_cut} – {high_cut} Hz")

    # Build mask: band‑pass + suppress specific peaks
    mask = np.zeros_like(fft_vals, dtype=complex)
    band = (np.abs(freqs) >= low_cut) & (np.abs(freqs) <= high_cut)
    mask[band] = 1  # start with band‑pass

    # Suppress selected peaks (and their negative counterparts)
    if suppress:
        bin_res = sampling_rate / len(combo_sig)
        for f0 in suppress:
            kill = np.abs(freqs - f0) <= (bin_res / 2)
            mask[kill] = 0
    
    recon = np.fft.ifft(fft_vals * mask).real
    mse = np.mean((recon - step_sig) ** 2)

    # ------- Plot reconstruction vs target & original --------
    fig_r, ax_r = plt.subplots(figsize=(10, 4))
    ax_r.plot(T, step_sig, label="Target (step)", color="orange", lw=1)
    ax_r.plot(T, recon + 0.5, label="Recovered", color="blue", lw=.9)
    ax_r.plot(T, combo_sig, label="Combined", color="gray", alpha=.4, lw=.8)
    ax_r.set_xlabel("Time (s)")
    ax_r.set_title(f"Band‑pass reconstruction | MSE: {mse:.3e}")
    ax_r.grid(alpha=.3)
    ax_r.legend()
    st.pyplot(fig_r)

    # Feedback box
    st.success(
        f"Filtering {low_cut:.1f} – {high_cut:.1f} Hz | Suppressed peaks: {suppress if suppress else 'None'} | MSE ≈ {mse:.3e}"
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.caption("Made with ❤️ + Streamlit • explore, learn, and have fun!")
