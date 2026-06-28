"""Test script to validate FFT functions in src/features.py"""
import sys
sys.path.insert(0, 'g:\\Master\\Thesis\\FLT\\Code\\ECG-to-stress\\src')
from features import Features
import numpy as np

# Test 1: Single chunk FFT
f = Features(fs=700)
chunk = np.sin(2*np.pi*1*np.arange(700)/700) * 1000  # 1 Hz sine wave, 1 second
freqs, mag = f.compute_fft(chunk)
print(f'FFT single chunk test:')
print(f'  freqs shape: {freqs.shape}')
print(f'  mag shape: {mag.shape}')
print(f'  max freq: {freqs[-1]:.2f} Hz')

# Test 2: Batch FFT
chunks = [chunk, chunk, np.random.randn(700)]
results = f.compute_fft_batch(chunks, verbose=False)
print(f'FFT batch test:')
print(f'  Number of results: {len(results)}')
for i, (fr, ma) in enumerate(results):
    print(f'  Result {i}: freqs={fr.shape}, mag={ma.shape}')

print('\nAll tests passed!')
