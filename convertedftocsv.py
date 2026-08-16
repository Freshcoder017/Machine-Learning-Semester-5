import os
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# =========================
# SETTINGS
# =========================

DATA_FOLDER = "."              # Folder containing h01.edf ... h20.edf
# Frequency bands
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45)
}


# =========================
# FEATURE EXTRACTION
# =========================

def extract_channel_features(signal, sfreq):

    features = {}

    # -------------------------
    # Time-domain features
    # -------------------------

    features["mean"] = np.mean(signal)
    features["std"] = np.std(signal)
    features["variance"] = np.var(signal)
    features["rms"] = np.sqrt(np.mean(signal ** 2))
    features["skewness"] = skew(signal)
    features["kurtosis"] = kurtosis(signal)

    # -------------------------
    # Frequency-domain features
    # -------------------------

    freqs, psd = welch(
        signal,
        fs=sfreq,
        nperseg=min(4096, len(signal))
    )

    # Total power from 0.5–45 Hz
    total_mask = (freqs >= 0.5) & (freqs <= 45)

    total_power = np.trapezoid(
        psd[total_mask],
        freqs[total_mask]
    )

    for band_name, (low, high) in BANDS.items():

        mask = (freqs >= low) & (freqs < high)

        band_power = np.trapezoid(
            psd[mask],
            freqs[mask]
        )

        # Relative power
        if total_power != 0:
            relative_power = band_power / total_power
        else:
            relative_power = 0

        features[f"{band_name}_relative_power"] = relative_power

    # -------------------------
    # Dominant frequency
    # -------------------------

    dominant_mask = (freqs >= 0.5) & (freqs <= 45)

    dominant_freq = freqs[dominant_mask][
        np.argmax(psd[dominant_mask])
    ]

    features["dominant_frequency"] = dominant_freq

    return features


# =========================
# PROCESS ONE EDF
# =========================

def process_edf(filepath):

    print(f"Processing: {filepath}")

    raw = mne.io.read_raw_edf(
        filepath,
        preload=True,
        verbose=False
    )

    sfreq = raw.info["sfreq"]

    data = raw.get_data()

    channel_names = raw.ch_names

    row = {}

    # Subject ID
    filename = os.path.basename(filepath)
    subject_id = os.path.splitext(filename)[0]

    row["subject_id"] = subject_id

    # -------------------------
    # Process every channel
    # -------------------------

    for i, channel in enumerate(channel_names):

        signal = data[i]

        channel_features = extract_channel_features(
            signal,
            sfreq
        )

        # Add channel name to feature name
        for feature_name, value in channel_features.items():

            column_name = f"{channel}_{feature_name}"

            row[column_name] = value

    return row


# =========================
# MAIN
# =========================

all_rows = []

for i in range(1, 21):

    filename = f"h{i:02d}.edf"

    filepath = os.path.join(
        DATA_FOLDER,
        filename
    )

    if not os.path.exists(filepath):

        print(f"WARNING: {filename} not found")
        continue

    try:

        row = process_edf(filepath)

        all_rows.append(row)

    except Exception as e:

        print(f"ERROR processing {filename}: {e}")


# =========================
# CREATE DATAFRAME
# =========================

df = pd.DataFrame(all_rows)

# Put subject_id first
columns = ["subject_id"] + [
    col for col in df.columns
    if col != "subject_id"
]

df = df[columns]

# Save
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==============================")
print("DONE!")
print("==============================")
print(f"Subjects processed: {len(df)}")
print(f"Number of features: {len(df.columns) - 1}")
print(f"Saved to: {OUTPUT_FILE}")