"""Script to rewrite the FFT notebook."""
import json

NB_PATH = r'g:\Master\Thesis\FLT\Code\ECG-to-stress\notebooks\06_FFT_analysis.ipynb'

# Keep the existing metadata from the original notebook
with open(NB_PATH, 'r', encoding='utf-8') as f:
    original = json.load(f)

cells = [

    # Cell 1 — Imports (unchanged)
    {
        "cell_type": "code",
        "execution_count": 4,
        "id": "9fbdce19",
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import os\n",
            "import warnings\n",
            "import sys\n",
            "sys.path.append('..')\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Import your classes\n",
            "from src.data import Data\n",
            "from src.features import Features\n",
            "from src.visualization import Visualization\n",
            "from src.correlation import Correlation\n",
            "\n",
            "# Initialize classes\n",
            "data_loader = Data(fs=700)\n",
            "feature_extractor = Features(fs=700)\n",
            "viz = Visualization()\n",
            "corr = Correlation()"
        ]
    },

    # Cell 2 — Load data, helper functions, extract segments
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "load_data_segments",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================================\n",
            "# 1. Load ECG data from subject S2\n",
            "# ============================================================\n",
            "file_path = '../data/WESAD/data/S2.pkl'\n",
            "ecg, label = data_loader.read_subject(file_path)\n",
            "fs = data_loader.fs\n",
            "\n",
            "print(f'Loaded ECG: shape={ecg.shape}, fs={fs} Hz')\n",
            "print(f'Unique labels: {np.unique(label)}')\n",
            "\n",
            "# WESAD label convention:\n",
            "#   1 = Baseline (non-stress),  2 = Stress,\n",
            "#   3 = Amusement (non-stress), 4 = Meditation\n",
            "label_names = {1: 'Baseline', 2: 'Stress', 3: 'Amusement', 4: 'Meditation'}\n",
            "\n",
            "# Focus on Baseline (1) as 'Non-stress' and Stress (2) as 'Stress'\n",
            "NON_STRESS_LABEL = 1\n",
            "STRESS_LABEL    = 2\n",
            "\n",
            "\n",
            "# ============================================================\n",
            "# 2. Find first continuous segment of a given label\n",
            "# ============================================================\n",
            "def find_continuous_segment(label_arr, target_label, duration_sec, fs):\n",
            "    needed = int(duration_sec * fs)\n",
            "    positions = np.where(label_arr == target_label)[0]\n",
            "    if len(positions) == 0:\n",
            "        return None, None\n",
            "    run_length = 1\n",
            "    for i in range(1, len(positions)):\n",
            "        if positions[i] == positions[i-1] + 1:\n",
            "            run_length += 1\n",
            "            if run_length >= needed:\n",
            "                start_idx = positions[i] - needed + 1\n",
            "                end_idx = positions[i] + 1\n",
            "                return start_idx, end_idx\n",
            "        else:\n",
            "            run_length = 1\n",
            "    return None, None\n",
            "\n",
            "\n",
            "# ============================================================\n",
            "# 3. Compute FFT magnitude spectrum\n",
            "# ============================================================\n",
            "def compute_fft(signal, fs, max_freq=10):\n",
            "    n = len(signal)\n",
            "    # Hanning window to reduce spectral leakage\n",
            "    windowed_signal = signal * np.hanning(n)\n",
            "    fft_vals = np.fft.rfft(windowed_signal)\n",
            "    freqs = np.fft.rfftfreq(n, d=1/fs)\n",
            "    magnitude = np.abs(fft_vals) / n\n",
            "    idx = freqs <= max_freq\n",
            "    return freqs[idx], magnitude[idx]\n",
            "\n",
            "\n",
            "# ============================================================\n",
            "# 4. Extract segments of different durations\n",
            "# ============================================================\n",
            "durations = [30, 120, 300]\n",
            "segments = {}  # {label: {duration_sec: (signal, start_idx)}}\n",
            "\n",
            "for lbl_name, lbl_val in [('Non-stress', NON_STRESS_LABEL),\n",
            "                            ('Stress', STRESS_LABEL)]:\n",
            "    segments[lbl_name] = {}\n",
            "    for dur in durations:\n",
            "        start, end = find_continuous_segment(label, lbl_val, dur, fs)\n",
            "        if start is not None:\n",
            "            seg = ecg[start:end]\n",
            "            segments[lbl_name][dur] = (seg, start)\n",
            "            print(f'{lbl_name} ({dur}s): samples [{start}:{end}], '\n",
            "                  f'time = {start/fs:.1f}s - {end/fs:.1f}s')\n",
            "        else:\n",
            "            print(f'{lbl_name} ({dur}s): NOT FOUND')\n",
            "\n",
            "print('\\nAll segments extracted.')"
        ]
    },
]
    # Cell 3 — FFT comparison plot (30s vs 120s vs 300s)
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "fft_multi_duration",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================================\n",
            "# 5. FFT comparison: 30s vs 120s vs 300s\n",
            "#    for both Non-stress (Baseline) and Stress\n",
            "# ============================================================\n",
            "max_freq_plot = 10  # Focus on 0-10 Hz for ECG\n",
            "colors = {'Non-stress': '#2d7a3e', 'Stress': '#c41e3a'}\n",
            "\n",
            "fig, axes = plt.subplots(2, 3, figsize=(18, 8))\n",
            "\n",
            "for row, lbl_name in enumerate(['Non-stress', 'Stress']):\n",
            "    for col, dur in enumerate(durations):\n",
            "        ax = axes[row, col]\n",
            "        if dur in segments[lbl_name]:\n",
            "            sig, start_idx = segments[lbl_name][dur]\n",
            "            freqs, mag = compute_fft(sig, fs, max_freq=max_freq_plot)\n",
            "            ax.plot(freqs, mag, color=colors[lbl_name], linewidth=1.0)\n",
            "            ax.set_xlabel('Frequency (Hz)', fontsize=10)\n",
            "            ax.set_title(f'{lbl_name} - {dur}s', fontsize=11, fontweight='bold')\n",
            "            ax.grid(True, alpha=0.3)\n",
            "            df = 1.0 / dur\n",
            "            ax.annotate(f'df = {df:.3f} Hz', xy=(0.98, 0.95),\n",
            "                        ha='right', va='top', fontsize=8,\n",
            "                        transform=ax.transAxes,\n",
            "                        bbox=dict(boxstyle='round,pad=0.3',\n",
            "                                  facecolor='lightyellow', alpha=0.8))\n",
            "        else:\n",
            "            ax.text(0.5, 0.5, 'No data', ha='center', va='center',\n",
            "                    transform=ax.transAxes, fontsize=14, color='gray')\n",
            "        if col == 0:\n",
            "            ax.set_ylabel('Magnitude', fontsize=11)\n",
            "\n",
            "plt.suptitle('FFT Magnitude Spectrum - ECG at Different Window Durations',\n",
            "             fontsize=14, fontweight='bold', y=1.02)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },

    # Cell 4 — 30s Stress vs Non-stress side-by-side
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "fft_stress_vs_nonstress",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================================\n",
            "# 6. 30-second FFT: Stress vs Non-stress side-by-side\n",
            "# ============================================================\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n",
            "\n",
            "for idx, (lbl_name, display_name, color) in enumerate([\n",
            "    ('Non-stress', 'Non-stress (Baseline)', '#2d7a3e'),\n",
            "    ('Stress',     'Stress',                 '#c41e3a'),\n",
            "]):\n",
            "    ax = axes[idx]\n",
            "    if 30 in segments[lbl_name]:\n",
            "        sig, start_idx = segments[lbl_name][30]\n",
            "        freqs, mag = compute_fft(sig, fs, max_freq=max_freq_plot)\n",
            "        ax.plot(freqs, mag, color=color, linewidth=1.2)\n",
            "        ax.set_xlabel('Frequency (Hz)', fontsize=11)\n",
            "        ax.set_ylabel('Magnitude', fontsize=11)\n",
            "        ax.set_title(f'{display_name} - 30s FFT', fontsize=12, fontweight='bold')\n",
            "        ax.grid(True, alpha=0.3)\n",
            "        # Highlight LF and HF bands\n",
            "        ax.axvspan(0.04, 0.15, alpha=0.12, color='blue', label='LF band')\n",
            "        ax.axvspan(0.15, 0.4,  alpha=0.12, color='red',  label='HF band')\n",
            "        ax.legend(fontsize=8, loc='upper right')\n",
            "    else:\n",
            "        ax.text(0.5, 0.5, 'No data', ha='center', va='center',\n",
            "                transform=ax.transAxes, fontsize=14, color='gray')\n",
            "\n",
            "plt.suptitle('30-Second FFT Comparison: Stress vs Non-stress',\n",
            "             fontsize=14, fontweight='bold', y=1.05)\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print('='*60)\n",
            "print('FFT Analysis Complete!')\n",
            "print(f'  - Sampling frequency: {fs} Hz')\n",
            "print(f'  - 30 sec = {30*fs} samples')\n",
            "print(f'  - 120 sec = {120*fs} samples')\n",
            "print(f'  - 300 sec = {300*fs} samples')\n",
            "print(f'  - Freq. resolution: 30s: {1/30:.3f} Hz, '\n",
            "      f'120s: {1/120:.4f} Hz, 300s: {1/300:.4f} Hz')\n",
            "print('='*60)"
        ]
    }
]

new_nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, indent=1, ensure_ascii=False)

print('Notebook rewritten successfully!')

