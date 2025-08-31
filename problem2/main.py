import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import time

# AES S-box
SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

def hamming_weight(x):
    """Calculate Hamming weight of a byte."""
    return bin(x).count('1')

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
    """Call in a loop to create terminal progress bar"""
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = round(100.00 * (iteration / float(total)), decimals)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def load_data(file_path):
    """Load and parse power trace data."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, header=None)
    
    # Parse plaintexts and ciphertexts (assuming 32-char hex strings)
    plaintexts = []
    for p in df.iloc[:, 0]:
        # Convert hex string to integer, then to 16-byte array
        p_int = int(p, 16)
        p_bytes = [(p_int >> (8 * i)) & 0xFF for i in range(15, -1, -1)]
        plaintexts.append(p_bytes)
    
    ciphertexts = []
    for c in df.iloc[:, 1]:
        c_int = int(c, 16)
        c_bytes = [(c_int >> (8 * i)) & 0xFF for i in range(15, -1, -1)]
        ciphertexts.append(c_bytes)
    
    # Power traces (remaining columns)
    traces = df.iloc[:, 2:].values
    
    # Check for NaN or infinite values
    if np.any(np.isnan(traces)) or np.any(np.isinf(traces)):
        print("Warning: Traces contain NaN or infinite values. Replacing with zeros.")
        traces = np.nan_to_num(traces)
    
    print(f"Data shape: {df.shape}")
    print(f"Sample plaintext: {plaintexts[0]}")
    print(f"Sample ciphertext: {ciphertexts[0]}")
    print(f"Sample trace (first 10 points): {traces[0, :10]}")
    
    return np.array(plaintexts), np.array(ciphertexts), traces

def align_traces_dynamic(traces, reference_idx=0, max_shift=50):
    """Align traces using cross-correlation with limited shift."""
    print("Aligning traces using cross-correlation with limited shift...")
    reference = traces[reference_idx]
    aligned_traces = np.zeros_like(traces)
    
    for i in range(traces.shape[0]):
        print_progress(i, traces.shape[0], prefix='Aligning:', suffix='Complete')
        
        # Compute cross-correlation with limited shift range
        correlation = np.zeros(max_shift * 2 + 1)
        for shift in range(-max_shift, max_shift + 1):
            shifted_trace = np.roll(traces[i], shift)
            # Use only overlapping part
            if shift >= 0:
                corr = np.corrcoef(reference[shift:], shifted_trace[shift:])[0, 1]
            else:
                corr = np.corrcoef(reference[:shift], shifted_trace[:shift])[0, 1]
            correlation[shift + max_shift] = corr if not np.isnan(corr) else 0
        
        # Find shift that maximizes correlation
        best_shift = np.argmax(correlation) - max_shift
        # Apply shift
        aligned_traces[i] = np.roll(traces[i], -best_shift)
    
    return aligned_traces

def preprocess_traces(traces, cutoff_freq=0.15, order=4):
    """Apply preprocessing steps to reduce noise."""
    print("Preprocessing traces...")
    
    # 1. Normalization
    print("  - Normalizing traces...")
    mean_trace = np.mean(traces, axis=0)
    std_trace = np.std(traces, axis=0)
    # Avoid division by zero
    std_trace[std_trace == 0] = 1
    normalized_traces = (traces - mean_trace) / std_trace
    
    # 2. Low-pass filtering
    print("  - Applying low-pass filter...")
    b, a = signal.butter(order, cutoff_freq)
    filtered_traces = np.zeros_like(normalized_traces)
    
    for i in range(normalized_traces.shape[0]):
        print_progress(i, normalized_traces.shape[0], prefix='Filtering:', suffix='Complete')
        filtered_traces[i] = signal.filtfilt(b, a, normalized_traces[i])
    
    return filtered_traces

def find_leakage_points(traces, plaintexts, byte_pos=0, num_test_keys=10):
    """Identify time points with highest leakage for a specific byte."""
    print(f"Finding leakage points for byte {byte_pos}...")
    num_traces = traces.shape[0]
    num_samples = traces.shape[1]
    
    # Test multiple key guesses to find the best leakage points
    test_keys = np.random.choice(256, num_test_keys, replace=False)
    max_correlations = np.zeros(num_samples)
    
    for k_idx, k in enumerate(test_keys):
        print_progress(k_idx, len(test_keys), prefix='Testing keys:', suffix='Complete')
        
        # Compute hypothetical power
        hypothetical = np.zeros(num_traces)
        for i in range(num_traces):
            p_byte = plaintexts[i][byte_pos]
            intermediate = SBOX[p_byte ^ k]
            hypothetical[i] = hamming_weight(intermediate)
        
        # Calculate correlation at each time point
        for t in range(num_samples):
            try:
                corr, _ = pearsonr(hypothetical, traces[:, t])
                if np.isnan(corr):
                    corr = 0
                if abs(corr) > abs(max_correlations[t]):
                    max_correlations[t] = corr
            except:
                pass
    
    # Find peaks in the correlation trace
    peaks, _ = signal.find_peaks(max_correlations, height=np.percentile(max_correlations, 95))
    
    # Sort peaks by correlation value
    peak_values = max_correlations[peaks]
    sorted_indices = np.argsort(peak_values)[::-1]
    top_peaks = peaks[sorted_indices[:5]]  # Take top 5 peaks
    
    print(f"Top leakage points for byte {byte_pos}: {top_peaks}")
    
    # Plot correlation trace with peaks
    plt.figure(figsize=(12, 4))
    plt.plot(max_correlations)
    plt.plot(peaks, max_correlations[peaks], "x")
    plt.title(f"Max Correlation Trace for Byte {byte_pos}")
    plt.xlabel("Time Sample")
    plt.ylabel("Correlation")
    plt.savefig(f"leakage_points_byte_{byte_pos}.png")
    plt.close()
    
    return top_peaks

def perform_cpa(traces, plaintexts, byte_pos, leakage_points):
    """Perform CPA for a specific key byte using identified leakage points."""
    print(f"Performing CPA for byte position {byte_pos}...")
    num_traces = traces.shape[0]
    
    # Calculate hypothetical power for all key guesses
    hypothetical_matrix = np.zeros((256, num_traces))
    for k in range(256):
        for i in range(num_traces):
            p_byte = plaintexts[i][byte_pos]
            intermediate = SBOX[p_byte ^ k]
            hypothetical_matrix[k, i] = hamming_weight(intermediate)
    
    # Calculate correlations for all key guesses at leakage points
    correlations = np.zeros(256)
    for k in range(256):
        print_progress(k, 256, prefix=f'Byte {byte_pos}:', suffix='Complete')
        max_corr = 0
        for t in leakage_points:
            try:
                corr, _ = pearsonr(hypothetical_matrix[k], traces[:, t])
                if np.isnan(corr):
                    corr = 0
                if abs(corr) > abs(max_corr):
                    max_corr = corr
            except:
                pass
        correlations[k] = max_corr
    
    # Rank key candidates by absolute correlation
    ranked = np.argsort(np.abs(correlations))[::-1]
    
    # Print top 5 candidates for debugging
    print(f"Top 5 candidates for byte {byte_pos}:")
    for i in range(min(5, len(ranked))):
        print(f"  {ranked[i]}: {correlations[ranked[i]]:.4f}")
    
    # Plot correlation distribution
    plt.figure(figsize=(10, 4))
    plt.bar(range(256), np.abs(correlations))
    plt.title(f"Correlation Distribution for Byte {byte_pos}")
    plt.xlabel("Key Guess")
    plt.ylabel("Absolute Correlation")
    plt.savefig(f"correlation_dist_byte_{byte_pos}.png")
    plt.close()
    
    return ranked, correlations

def main():
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Load data
    file_path = "real_power_trace.csv"  # Update with actual file path
    plaintexts, ciphertexts, traces = load_data(file_path)
    
    # Preprocess traces
    traces = align_traces_dynamic(traces, max_shift=50)
    traces = preprocess_traces(traces, cutoff_freq=0.15)
    
    # Calculate average trace for visualization
    avg_trace = np.mean(traces, axis=0)
    
    # Plot average trace
    plt.figure(figsize=(12, 4))
    plt.plot(avg_trace)
    plt.title("Average Power Trace")
    plt.xlabel("Time Sample")
    plt.ylabel("Power")
    plt.savefig("results/average_trace.png")
    plt.close()
    
    # Find leakage points for each byte
    leakage_points_all = []
    for byte_pos in range(16):
        leakage_points = find_leakage_points(traces, plaintexts, byte_pos=byte_pos, num_test_keys=20)
        leakage_points_all.append(leakage_points)
    
    # Perform CPA for each key byte
    key_guess = np.zeros(16, dtype=int)
    for byte_pos in range(16):
        ranked, correlations = perform_cpa(traces, plaintexts, byte_pos, leakage_points_all[byte_pos])
        key_guess[byte_pos] = ranked[0]
        
        # Save ranking for this byte
        with open(f"results/byte_{byte_pos:02d}.txt", "w") as f:
            for candidate in ranked:
                f.write(f"{candidate}\n")
    
    # Save full key guess
    with open("results/secret_key.txt", "w") as f:
        for byte in key_guess:
            f.write(f"{byte:02x}")
    
    print(f"\nRecovered key: {''.join(f'{byte:02x}' for byte in key_guess)}")
    print("\nResults saved to 'results' directory")

if _name_ == "_main_":
    main()