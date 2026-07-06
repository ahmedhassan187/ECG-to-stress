"""Modify main.py to refactor it"""

import re

path = r'g:\Master\Thesis\FLT\Code\ECG-to-stress\src\main.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace docstring with cleaner version
old_doc = '"""\nECG-to-Stress Analysis CLI\nCommand-line interface for WESAD dataset analysis, feature extraction, correlation analysis,\nmachine learning model training, and FFT frequency analysis.\n\nLabel mapping (WESAD):\n    1 \u2192 Baseline   \u2192 Non-Stress (binary 0)\n    2 \u2192 Stress     \u2192 Stress     (binary 1)\n    3 \u2192 Amusement  \u2192 Non-Stress (binary 0)\n    4 \u2192 Meditation \u2192 Stress     (binary 1)\n\nUsage Examples:\n    py src/main.py --help                           # Show help message\n    py src/main.py -i data/WESAD -c                 # Run correlation analysis with custom dataset path\n    py src/main.py --input /path/to/data -c         # Same with long flag\n    py src/main.py -c                               # Run correlation analysis (default dataset path)\n    py src/main.py --corr --features feature1 feature2  # Run correlation on specific features\n    py src/main.py -f                               # Plot full ECG signals (default 5000 points)\n    py src/main.py --full -p 10000                  # Plot with 10000 points per chunk\n    py src/main.py -m                               # Train all models on all datasets\n    py src/main.py --ml -d 30 120                   # Train models on 30s and 120s datasets\n    py src/main.py -m -mo knn svm random_forest     # Train specific models on all datasets\n    py src/main.py --fft                            # Run FFT analysis (30s/120s/300s chunks)\n    py src/main.py --fft -d 30 120                  # FFT on specific durations only\n    py src/main.py --fft --fft-max-pairs 1000       # More cosine-similarity pairs\n    \n    # PREDICTION MODE\n    py src/main.py -p predict -d 30                 # Predict on test data using 30s models\n    py src/main.py --predict -i data/test_data      # Predict on custom test data\n    py src/main.py -p predict -mo knn random_forest # Use specific models for prediction\n"""'
new_doc = '"""\nECG-to-Stress Analysis CLI\n===========================\nCommand-line interface for WESAD dataset analysis, feature extraction,\ncorrelation analysis, machine learning model training, and FFT analysis.\n\nLabel mapping (WESAD):\n    1 -> Baseline   -> Non-Stress (binary 0)\n    2 -> Stress     -> Stress     (binary 1)\n    3 -> Amusement  -> Non-Stress (binary 0)\n    4 -> Meditation -> Stress     (binary 1)\n\nUsage Examples:\n    py src/main.py --help                           # Show help\n    py src/main.py -c                               # Correlation analysis\n    py src/main.py -f                               # Full signal visualization\n    py src/main.py -m                               # Train all ML models\n    py src/main.py --fft                            # FFT frequency analysis\n    py src/main.py --predict -d 30                  # Predict on test data\n"""'

if old_doc in content:
    content = content.replace(old_doc, new_doc, 1)
    print("Docstring replaced")
else:
    print("Docstring NOT FOUND")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
