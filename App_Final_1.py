import streamlit as st
import numpy as np
import librosa
import pickle
import os
import socket
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import tempfile
import gdown

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Song Recognition",
    page_icon="🎵",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .result-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #00d4ff;
        border-radius: 16px;
        padding: 28px 36px;
        text-align: center;
        margin: 20px 0;
    }
    .song-title {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 8px 0 4px 0;
    }
    .score-label {
        font-size: 1rem;
        color: #aaaaaa;
        margin: 0;
    }
    .score-value {
        font-size: 1.4rem;
        color: #ffffff;
        font-weight: 600;
    }
    .no-match {
        font-size: 1.5rem;
        color: #ff4b4b;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #0077ff);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px 36px;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)


# ── Download + load database (cached) ────────────────────────
GDRIVE_FILE_ID = "1qbTuP8quE8hmVJtURhj2NuqZaJTSRp9n"
DB_PATH        = "song_database.pkl"

@st.cache_resource
def load_database():
    if not os.path.exists(DB_PATH):
        with st.spinner("⬇️ Downloading database from Google Drive (first run only)…"):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, DB_PATH, quiet=False)

    if not os.path.exists(DB_PATH):
        return None          # download failed

    with open(DB_PATH, "rb") as f:
        return pickle.load(f)


# ── Core DSP helpers ──────────────────────────────────────────
def compute_spectrogram(x, fs, window_size=2048, hop_size=1024):
    num_windows = (len(x) - window_size) // hop_size
    spectrogram = np.zeros((window_size // 2, num_windows))
    for i in range(num_windows):
        start = i * hop_size
        segment = x[start:start + window_size]
        X = np.fft.fft(segment)
        spectrogram[:, i] = np.abs(X[:window_size // 2])
    freqs = np.linspace(0, fs / 2, spectrogram.shape[0])
    times = np.arange(spectrogram.shape[1]) * hop_size / fs
    return spectrogram, freqs, times


def find_peaks(spectrogram_db, freqs, freq_neighbourhood=10, time_neighbourhood=10):
    max_db = np.max(spectrogram_db)
    max_freq_bin = np.where(freqs <= 8000)[0][-1]
    peaks = []
    for i in range(freq_neighbourhood, max_freq_bin - freq_neighbourhood):
        freq = freqs[i]
        threshold = max_db - 25 if freq < 4000 else max_db - 30
        for j in range(time_neighbourhood, spectrogram_db.shape[1] - time_neighbourhood):
            current = spectrogram_db[i, j]
            if current > threshold:
                neighbourhood = spectrogram_db[
                    i - freq_neighbourhood:i + freq_neighbourhood + 1,
                    j - time_neighbourhood:j + time_neighbourhood + 1
                ]
                if current == np.max(neighbourhood):
                    peaks.append((i, j))
    return peaks


def build_fingerprints(peaks, freqs, times, fan_value=10, min_dt=0, max_dt=5):
    peak_points = sorted([(times[p[1]], int(freqs[p[0]])) for p in peaks])
    fingerprints = []
    for i in range(len(peak_points)):
        t1, f1 = peak_points[i]
        for j in range(1, fan_value + 1):
            if i + j >= len(peak_points):
                break
            t2, f2 = peak_points[i + j]
            delta_t = t2 - t1
            if min_dt <= delta_t <= max_dt:
                fingerprints.append((int(f1), int(f2), round(delta_t, 3), round(t1, 3)))
    return fingerprints, peak_points


def match_fingerprints(query_fingerprints, database, confidence_thresh=5):
    offset_histogram = {}
    for qf1, qf2, qdt, qt in query_fingerprints:
        key = (qf1, qf2, qdt)
        if key in database:
            for song_name, st in database[key]:
                pair = (song_name, round(st - qt, 2))
                offset_histogram[pair] = offset_histogram.get(pair, 0) + 1

    best_song, best_count = None, 0
    for (song_name, _), count in offset_histogram.items():
        if count > best_count:
            best_count = count
            best_song = song_name

    if best_song is None or best_count < confidence_thresh:
        return None, best_count, offset_histogram
    return best_song, best_count, offset_histogram


# ── Plot helpers ──────────────────────────────────────────────
def plot_spectrogram(spectrogram, freqs, times):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    img = ax.imshow(
        20 * np.log10(spectrogram + 1),
        aspect="auto", origin="lower", cmap="inferno",
        extent=[times[0], times[-1], freqs[0], freqs[-1]]
    )
    ax.set_ylim(0, 10000)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Frequency (Hz)", color="white")
    ax.set_title("Spectrogram", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    cb = fig.colorbar(img, ax=ax)
    cb.set_label("Magnitude (dB)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    fig.tight_layout()
    return fig


def plot_constellation(spectrogram, freqs, times, peaks):
    peak_freq_points = [freqs[p[0]] for p in peaks]
    peak_time_points = [times[p[1]] for p in peaks]
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    img = ax.imshow(
        20 * np.log10(spectrogram + 1),
        aspect="auto", origin="lower", cmap="inferno",
        extent=[times[0], times[-1], freqs[0], freqs[-1]]
    )
    ax.set_ylim(0, 10000)
    ax.scatter(peak_time_points, peak_freq_points, color="cyan", s=8, zorder=5)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Frequency (Hz)", color="white")
    ax.set_title("Constellation Map", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    cb = fig.colorbar(img, ax=ax)
    cb.set_label("Magnitude (dB)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    fig.tight_layout()
    return fig


def plot_offset_histogram(offset_histogram, best_song):
    offsets = [offset for (song, offset), _ in offset_histogram.items() if song == best_song]
    counts  = [count  for (song, _),    count in offset_histogram.items() if song == best_song]
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.bar(offsets, counts, width=0.5, color="cyan", edgecolor="none")
    ax.set_xlabel("Time Offset (s)", color="white")
    ax.set_ylabel("Matching Fingerprints", color="white")
    ax.set_title(f"Offset Histogram — {best_song}", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    fig.tight_layout()
    return fig


# ── Network info helper ───────────────────────────────────────
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
st.title("🎵 Song Recognition App")
st.markdown("Upload a short audio clip and the app will identify the song from the database.")

# Sidebar ── network info
with st.sidebar:
    st.header("🌐 Access Links")
    local_ip = get_local_ip()
    st.markdown(f"""
| Type | URL |
|------|-----|
| 🖥️ Local | `http://localhost:8501` |
| 📡 Network | `http://{local_ip}:8501` |
""")
    st.caption("Share the Network link with other devices on the same Wi-Fi.")
    st.divider()
    st.markdown("**How to run:**")
    st.code("streamlit run app.py --server.address=0.0.0.0 --server.port=8501", language="bash")

# Load DB
database = load_database()
if database is None:
    st.error("❌ Failed to download `song_database.pkl` from Google Drive. Check your internet connection and that the file is shared publicly, then restart the app.")
    st.stop()

st.success(f"✅ Database loaded — {len(database):,} unique fingerprint hashes")

# Upload
uploaded = st.file_uploader("Upload an audio clip (MP3 / WAV)", type=["mp3", "wav"])

if uploaded:
    st.audio(uploaded, format="audio/mp3")

    if st.button("🔍 Identify Song"):
        with st.spinner("Analysing audio…"):
            # Save to temp file (librosa needs a path)
            suffix = ".mp3" if uploaded.name.endswith(".mp3") else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                x, fs = librosa.load(tmp_path, sr=22050)
                if len(x.shape) == 2:
                    x = x[:, 0]

                spectrogram, freqs, times = compute_spectrogram(x, fs)
                spectrogram_db = 20 * np.log10(spectrogram + 1)
                peaks = find_peaks(spectrogram_db, freqs)
                fingerprints, _ = build_fingerprints(peaks, freqs, times)
                best_song, best_count, offset_histogram = match_fingerprints(fingerprints, database)

            finally:
                os.unlink(tmp_path)

        # ── Result ────────────────────────────────────────────
        st.markdown("---")
        if best_song:
            song_display = os.path.splitext(best_song)[0]
            st.markdown(f"""
<div class="result-box">
    <p class="score-label">🎶 Matched Song</p>
    <p class="song-title">{song_display}</p>
    <p class="score-label">Aligned Fingerprints</p>
    <p class="score-value">{best_count}</p>
    <p class="score-label">Query Fingerprints</p>
    <p class="score-value">{len(fingerprints)}</p>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="result-box">
    <p class="no-match">❌ No match found</p>
    <p class="score-label">Try a longer or cleaner audio clip.</p>
</div>
""", unsafe_allow_html=True)

        # ── Plots ──────────────────────────────────────────────
        st.markdown("### 📊 Visualisations")
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(plot_spectrogram(spectrogram, freqs, times))

        with col2:
            st.pyplot(plot_constellation(spectrogram, freqs, times, peaks))

        if best_song and offset_histogram:
            st.pyplot(plot_offset_histogram(offset_histogram, best_song))
