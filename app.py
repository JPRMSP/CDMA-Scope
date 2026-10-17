
import itertools

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="CDMA-Scope | FI1924",
    page_icon="📡",
    layout="wide",
)

st.title("📡 CDMA-Scope")
st.subheader("Real-Time CDMA Signal Detection & Multiuser Analysis")
st.caption(
    "Anna University | FI1924 | Faculty of Information and "
    "Communication Engineering | Regulation 2021"
)

st.markdown(
    """
    Explore spreading codes, multiuser interference, matched filtering,
    near-far resistance and optimum detection. Every experiment generates
    new signals mathematically; no datasets or pretrained models are used.
    """
)


def hadamard_matrix(n):
    """Generate an orthogonal Walsh-Hadamard matrix."""
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def generate_codes(kind, users, sf, rng):
    """Return normalized spreading signatures."""
    if kind == "Walsh orthogonal":
        codes = hadamard_matrix(sf)[:users]
    else:
        codes = rng.choice([-1.0, 1.0], size=(users, sf))
    return codes.T / np.sqrt(sf)


def run_simulation(
    users, sf, nbits, snr_db, near_far_db,
    code_kind, channel_kind, seed
):
    rng = np.random.default_rng(seed)

    S = generate_codes(code_kind, users, sf, rng)

    # Each column represents one user's independently generated bits.
    bits = rng.choice([-1.0, 1.0], size=(users, nbits))

    # User 0 is the desired user. Other users can be much stronger.
    powers = np.ones(users)
    if users > 1:
        powers[1:] = 10.0 ** (near_far_db / 20.0)

    # Known channel gains are assumed at the receiver for coherent detection.
    if channel_kind == "Rayleigh fading":
        fading = np.sqrt(
            rng.normal(size=users) ** 2
            + rng.normal(size=users) ** 2
        ) / np.sqrt(2.0)
    else:
        fading = np.ones(users)

    gains = powers * fading
    H = S * gains[np.newaxis, :]

    # Chip-level received samples for each symbol interval.
    clean = H @ bits

    # Noise variance referenced to average received bit energy.
    eb = np.mean(np.sum(H**2, axis=0))
    sigma = np.sqrt(max(eb, 1e-12) / (2 * 10 ** (snr_db / 10)))
    received = clean + rng.normal(
        0.0, sigma, size=clean.shape
    )

    # Conventional coherent matched-filter detector.
    matched_scores = H.T @ received
    matched_bits = np.where(matched_scores >= 0, 1.0, -1.0)

    # Decorrelating detector: invert the signature Gram matrix.
    gram = H.T @ H
    decor_scores = np.linalg.pinv(gram) @ (H.T @ received)
    decor_bits = np.where(decor_scores >= 0, 1.0, -1.0)

    # Exhaustive maximum-likelihood detector.
    # Practical for a small number of simultaneously active users.
    candidates = np.array(
        list(itertools.product([-1.0, 1.0], repeat=users))
    ).T
    hypotheses = H @ candidates
    ml_bits = np.empty_like(bits)

    if users <= 8:
        for t in range(nbits):
            distances = np.sum(
                (hypotheses - received[:, t:t + 1]) ** 2,
                axis=0,
            )
            ml_bits[:, t] = candidates[:, np.argmin(distances)]
    else:
        ml_bits = np.full_like(bits, np.nan)

    # Actual measured bit error rates.
    def ber(estimate):
        return np.mean(estimate != bits)

    results = {
        "Matched filter": (matched_bits, ber(matched_bits)),
        "Decorrelating": (decor_bits, ber(decor_bits)),
    }
    if users <= 8:
        results["Maximum likelihood"] = (ml_bits, ber(ml_bits))

    return {
        "codes": S,
        "bits": bits,
        "received": received,
        "clean": clean,
        "results": results,
        "sigma": sigma,
        "gains": gains,
        "powers": powers,
        "fading": fading,
        "snr": snr_db,
    }


with st.sidebar:
    st.header("⚙️ Simulation controls")

    users = st.slider("Active CDMA users", 2, 8, 4)
    sf = st.select_slider(
        "Spreading factor",
        options=[8, 16, 32, 64],
        value=16,
    )
    nbits = st.slider("Bits per user", 100, 2000, 500, step=100)
    snr_db = st.slider("Eb/N0 (dB)", -5, 25, 8)
    near_far_db = st.slider(
        "Interferer power above target (dB)", 0, 30, 15
    )
    code_kind = st.selectbox(
        "Spreading code",
        ["Walsh orthogonal", "PN random"],
    )
    channel_kind = st.selectbox(
        "Channel",
        ["AWGN only", "Rayleigh fading"],
    )
    seed = st.number_input(
        "Signal generation seed",
        min_value=0,
        max_value=999999,
        value=42,
    )

    run = st.button(
        "🚀 Run new experiment",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "Adjust any control and rerun to generate a new experiment."
    )

if "simulation" not in st.session_state:
    st.session_state.simulation = None

if run or st.session_state.simulation is None:
    st.session_state.simulation = run_simulation(
        users, sf, nbits, snr_db, near_far_db,
        code_kind, channel_kind, int(seed)
    )

# Rerun if parameters change, even without pressing the button.
current_config = (
    users, sf, nbits, snr_db, near_far_db,
    code_kind, channel_kind, int(seed)
)
if st.session_state.get("last_config") != current_config:
    st.session_state.simulation = run_simulation(
        *current_config
    )
    st.session_state.last_config = current_config

sim = st.session_state.simulation
results = sim["results"]

st.success(
    f"Experiment ready: {users} users · SF={sf} · "
    f"{nbits} bits/user · {channel_kind}"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Active users", users)
m2.metric("Spreading factor", sf)
m3.metric("Eb/N0", f"{snr_db} dB")
m4.metric("Noise standard deviation", f"{sim['sigma']:.4f}")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Detector comparison",
        "📡 Spreading codes",
        "🌊 Received signal",
        "🧪 Theory & observations",
    ]
)

with tab1:
    st.subheader("Measured bit error rate")

    cols = st.columns(len(results))
    for col, (name, (_, error_rate)) in zip(cols, results.items()):
        col.metric(name, f"{error_rate:.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(results.keys())
    values = [results[name][1] for name in names]
    bars = ax.bar(names, values, color=["#2878B5", "#E09F3E", "#2A9D8F"][:len(names)])
    ax.set_ylabel("Bit error rate (BER)")
    ax.set_ylim(0, max(0.05, max(values, default=0) * 1.25))
    ax.set_title("Detector performance on the same received signal")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Desired user: transmitted vs detected bits")
    detector_name = st.selectbox(
        "Choose detector",
        list(results.keys()),
    )
    detected = results[detector_name][0][0]
    transmitted = sim["bits"][0]
    show_n = min(60, nbits)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    x = np.arange(show_n)
    ax.step(x, transmitted[:show_n], where="mid",
            label="Transmitted", linewidth=2)
    ax.step(x, detected[:show_n] * 0.85, where="mid",
            label="Detected (scaled for display)", alpha=0.8)
    ax.set_xlabel("Bit index")
    ax.set_ylabel("BPSK symbol")
    ax.set_yticks([-1, 0, 1])
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    errors = int(np.sum(transmitted != detected))
    st.write(
        f"Desired-user errors: **{errors} / {nbits}** "
        f"using {detector_name}."
    )

with tab2:
    st.subheader("Spreading signatures")

    fig, ax = plt.subplots(figsize=(10, 4))
    for k in range(users):
        ax.step(
            np.arange(sf),
            sim["codes"][:, k],
            where="mid",
            label=f"User {k + 1}",
        )
    ax.set_xlabel("Chip index")
    ax.set_ylabel("Normalized chip value")
    ax.set_title(f"{code_kind}: spreading signatures")
    ax.legend(ncol=4)
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    corr = sim["codes"].T @ sim["codes"]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Normalized code cross-correlation matrix")
    ax.set_xlabel("User")
    ax.set_ylabel("User")
    ax.set_xticks(range(users), range(1, users + 1))
    ax.set_yticks(range(users), range(1, users + 1))
    for i in range(users):
        for j in range(users):
            ax.text(j, i, f"{corr[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Correlation")
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Walsh codes have zero cross-correlation for distinct users "
        "when chip timing is aligned and the full code is observed. "
        "Random PN signatures generally have nonzero cross-correlation."
    )

with tab3:
    st.subheader("Received chip waveform")
    symbol_index = st.slider(
        "Select symbol interval",
        0, nbits - 1, 0,
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        sim["clean"][:, symbol_index],
        marker="o",
        label="Combined signal (without noise)",
    )
    ax.plot(
        sim["received"][:, symbol_index],
        marker="x",
        label="Received signal (with noise)",
    )
    ax.set_xlabel("Chip index")
    ax.set_ylabel("Amplitude")
    ax.set_title("Multiuser signal at the receiver")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Received channel gains")
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(
        [f"User {i + 1}" for i in range(users)],
        sim["gains"],
        color="#2878B5",
    )
    ax.set_ylabel("Effective amplitude")
    ax.set_title("User power and channel gain")
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    st.subheader("Signal detection theory")

    st.markdown(
        r"""
        **BPSK modulation**

        Each bit is represented by \(b_k \in \{-1,+1\}\).

        **Received synchronous CDMA model**

        \[
        \mathbf{y}=\mathbf{H}\mathbf{b}+\mathbf{n}
        \]

        Here, \(\mathbf{H}\) contains spreading signatures and
        effective channel amplitudes, while \(\mathbf{n}\) is
        additive white Gaussian noise.

        **Matched-filter decision**

        \[
        \hat b_k=\operatorname{sign}(\mathbf{h}_k^T\mathbf{y})
        \]

        **Decorrelating detector**

        \[
        \hat{\mathbf b}=
        \operatorname{sign}\left[
        (\mathbf H^T\mathbf H)^\dagger\mathbf H^T\mathbf y
        \right]
        \]

        **Maximum-likelihood detector**

        \[
        \hat{\mathbf b}_{ML}=
        \arg\min_{\mathbf b\in\{-1,+1\}^K}
        \|\mathbf y-\mathbf H\mathbf b\|^2
        \]

        **Measured bit error rate**

        \[
        BER=\frac{\text{number of incorrect detected bits}}
        {\text{total transmitted bits}}
        \]
        """
    )

    st.subheader("Interpret the experiment")
    st.markdown(
        """
        - Increase interferer power to investigate the near-far problem.
        - Compare Walsh and PN codes to examine cross-correlation.
        - Lower Eb/N0 to study the impact of receiver noise.
        - Select Rayleigh fading to introduce random channel amplitudes.
        - Compare all detectors using identical transmitted bits and noise.
        """
    )

    st.warning(
        "Model assumptions: synchronous users, BPSK, one fixed channel "
        "gain per user per experiment, and coherent detection with "
        "known channel gains. The decorrelator uses a pseudoinverse. "
        "The exhaustive ML detector is computationally expensive. "
        "This is not a complete asynchronous CDMA, IS-95, or WCDMA "
        "implementation."
    )

st.divider()
st.caption(
    "CDMA-Scope | Educational signal-processing laboratory | "
    "Synthetic signals only | No datasets or pretrained models"
)
