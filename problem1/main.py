import csv

# AES S-box
S_box = [
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

def main():
    # Read the CSV file
    with open('simulated_power_trace.csv', 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Extract ciphertexts and power traces
    ciphertexts = [row[0] for row in rows]
    power_traces = [list(map(float, row[1:])) for row in rows]
    
    T = len(ciphertexts)
    D = len(power_traces[0]) if T > 0 else 0
    
    # Precompute mean and standard deviation for each time point
    mean_power = [0.0] * D
    std_power = [0.0] * D
    for t in range(D):
        col = [power_traces[i][t] for i in range(T)]
        mean_power[t] = sum(col) / T
        variance = sum((x - mean_power[t])**2 for x in col) / T
        std_power[t] = variance ** 0.5
    
    # Precompute centered power traces
    P_centered = []
    for t in range(D):
        centered_t = [power_traces[i][t] - mean_power[t] for i in range(T)]
        P_centered.append(centered_t)
    
    # Process each byte of the key
    full_key_guesses = [None] * 16
    for j in range(16):
        guesses = []  # List of (key_byte, score)
        for k in range(256):
            HW_input = []
            HW_output = []
            for i in range(T):
                ct_hex = ciphertexts[i]
                ct_bytes = bytes.fromhex(ct_hex)
                ct_j = ct_bytes[j]
                h_input = ct_j ^ k
                h_output = S_box[h_input]
                HW_input.append(bin(h_input).count("1"))
                HW_output.append(bin(h_output).count("1"))
            
            # Compute mean and std for HW_input
            mean_hw_in = sum(HW_input) / T
            if T > 1:
                std_hw_in = (sum((x - mean_hw_in)**2 for x in HW_input) / T) ** 0.5
            else:
                std_hw_in = 0.0
            HW_centered_in = [x - mean_hw_in for x in HW_input]
            
            # Compute mean and std for HW_output
            mean_hw_out = sum(HW_output) / T
            if T > 1:
                std_hw_out = (sum((x - mean_hw_out)**2 for x in HW_output) / T) ** 0.5
            else:
                std_hw_out = 0.0
            HW_centered_out = [x - mean_hw_out for x in HW_output]
            
            # Compute covariance for each time point
            cov_in = [0.0] * D
            cov_out = [0.0] * D
            for t in range(D):
                s_in = 0.0
                s_out = 0.0
                for i in range(T):
                    s_in += HW_centered_in[i] * P_centered[t][i]
                    s_out += HW_centered_out[i] * P_centered[t][i]
                cov_in[t] = s_in / T
                cov_out[t] = s_out / T
            
            # Compute correlation for each time point
            corr_in = [0.0] * D
            corr_out = [0.0] * D
            for t in range(D):
                if std_hw_in * std_power[t] != 0:
                    corr_in[t] = cov_in[t] / (std_hw_in * std_power[t])
                if std_hw_out * std_power[t] != 0:
                    corr_out[t] = cov_out[t] / (std_hw_out * std_power[t])
            
            max_corr_in = max(abs(x) for x in corr_in)
            max_corr_out = max(abs(x) for x in corr_out)
            score = max(max_corr_in, max_corr_out)
            guesses.append((k, score))
        
        # Sort guesses by score descending
        guesses.sort(key=lambda x: x[1], reverse=True)
        full_key_guesses[j] = guesses
        
        # Write to byte_jj.txt
        filename = f'byte_{j:02d}.txt'
        with open(filename, 'w') as f:
            for k, score in guesses:
                f.write(f'{k}\n')
    
    # Extract the last round key (top guess for each byte)
    last_round_key = [full_key_guesses[j][0][0] for j in range(16)]
    
    # Reverse the key schedule to get the original key
    RCON = [0, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]
    w = [None] * 44
    for i in range(4):
        start = i * 4
        end = start + 4
        w[40 + i] = last_round_key[start:end]
    
    for i in range(43, 3, -1):
        if i % 4 == 0:
            word = w[i-1]
            rotated = word[1:] + [word[0]]
            sboxed = [S_box[b] for b in rotated]
            rcon_val = RCON[i // 4]
            g_word = [sboxed[0] ^ rcon_val] + sboxed[1:]
            w[i-4] = [w[i][j] ^ g_word[j] for j in range(4)]
        else:
            w[i-4] = [w[i][j] ^ w[i-1][j] for j in range(4)]
    
    original_key = []
    for i in range(4):
        original_key.extend(w[i])
    
    # Write the original key to key.txt
    original_key_hex = ''.join(f'{b:02x}' for b in original_key)
    with open('key.txt', 'w') as f:
        f.write(original_key_hex)

if _name_ == '_main_':
    main()
