import os
import random
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def list_signal_types_in_dir(dir_path):
    signal_types = dict()
    for filename in os.listdir(dir_path):
        modType = filename.split('_')[0]
        modTypeWithSyn = modType + '_syn' if 'chunk' not in filename else modType
        signal_types[modTypeWithSyn] = signal_types.get(modTypeWithSyn, 0) + 1
    print("Signal Types, Counts, Percentage:")
    total_files = sum(signal_types.values())
    for modType, count in signal_types.items():
        percentage = (count / total_files) * 100
        print(f"{modType}: {count} files ({percentage:.2f}%)")
    print("Total files:", sum(signal_types.values()))

def sample_signal_data(source_dir, target_dir, sample_size=1000):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    seen = set()
    source_dir_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    for i in range(sample_size):
        random_file = random.choice(source_dir_files)
        # resample if the file already exists in the target directory 
        while random_file in os.listdir(target_dir): 
            random_file = random.choice(source_dir_files)
        source_file_path = os.path.join(source_dir, random_file)
        target_file_path = os.path.join(target_dir, random_file)
        shutil.copy(source_file_path, target_file_path)
        seen.add(random_file)
    return

def train_test_splitter(source_dir, train_dir, test_dir, train_ratio=0.8, file_filter=None, move_files=False):
    """
    Randomly split files in source_dir into train_dir and test_dir.

    Args:
        source_dir (str): Path to source folder.
        train_dir (str): Path to training folder.
        test_dir (str): Path to testing folder.
        train_ratio (float): Proportion of files to use for training (default 0.8).
        file_filter (callable, optional): Function to filter files (e.g., lambda f: f.endswith('.csv')).
        move_files (bool): If True, move files instead of copying.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    if file_filter:
        all_files = [f for f in all_files if file_filter(f)]

    random.shuffle(all_files)
    split_idx = int(train_ratio * len(all_files))
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    for f in train_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(train_dir, f)
        shutil.move(src, dst) if move_files else shutil.copy(src, dst)
    for f in test_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(test_dir, f)
        shutil.move(src, dst) if move_files else shutil.copy(src, dst)

    print(f"Copied {len(train_files)} files to {train_dir}")
    print(f"Copied {len(test_files)} files to {test_dir}")

# Example usage:
# train_test_splitter(SOURCE_DIR, TRAIN_DIR, TEST_DIR, train_ratio=0.8, file_filter=

    
    print("Sampling complete.")

def delete_iq_header(dir_path):
    """
    Checks the 11th row to see if it is a header
    """
    total_files = len(os.listdir(dir_path))
    # print(dir_path)
    for idx, filename in enumerate(os.listdir(dir_path)):
        # print(filename)
        if filename.endswith('.csv'):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            if len(lines) > 10 and 'I' in lines[10]:
                print(f"Deleting header from {filename}")
                with open(file_path, 'w') as f:
                    f.writelines(lines[:10])  # Keep the first 10 lines
                    f.writelines(lines[11:])  # Skip the 11th line
            else:
                pass
            print(f"Processed {idx + 1}/{total_files} files in {dir_path}")

def visualize_fft(folder_path, Fs):
    """
    Visualizes the FFT of CSV files in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing CSV files.
        Fs (float): Sampling frequency in Hz.
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # === Initialize a set to track seen files ===
    seen = set()
    
    # === Loop through all CSV files ===
    for filename in os.listdir(folder_path):
        if (filename.endswith('.csv') or filename.endswith('.CSV')):
            filepath = os.path.join(folder_path, filename)
            seen.add(filename.split("_")[0])
            # === Load CSV: assume I in column 0, Q in column 1 ===
            try:
                df = pd.read_csv(filepath, skiprows=10, header=None)  # Skip metadata rows if needed
            except Exception as e:
                print(f"Could not load {filename}: {e}")
                continue

            if df.shape[1] < 2:
                print(f"File {filename} does not have at least two columns.")
                continue

            I = df.iloc[:, 0].values
            Q = df.iloc[:, 1].values
            x = I + 1j * Q  # Form complex baseband signal

            # === Compute FFT ===
            N = len(x)
            X = np.fft.fftshift(np.fft.fft(x, n=N))
            freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/Fs))
            print(len(freqs), len(X))
            # breakpoint()
            magnitude = 20 * np.log10(np.abs(X) + 1e-12)  # dB scale, avoid log(0)

            # === Plot ===
            plt.figure(figsize=(10, 10))
            plt.plot(freqs / 1e3, magnitude)
            plt.plot(freqs[160000:165000] / 1e3, magnitude[160000:165000])
            plt.title(f'Frequency Spectrum of {filename}')
            plt.xlabel('Frequency (kHz)')
            plt.ylabel('Magnitude (dB)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def extract_real_data():
    source_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/synthetic/synthetic_set1'
    comp_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/synthetic/synthetic_set3'
    target_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/synthetic/real_data'
    os.makedirs(target_path, exist_ok=True)
    # Loop through source path, if file is not in comp_path, copy it to target_path
    for filename in os.listdir(source_path):
        if filename not in os.listdir(comp_path):
            source_file_path = os.path.join(source_path, filename)
            target_file_path = os.path.join(target_path, filename)
            shutil.copy(source_file_path, target_file_path)
            print(f"Copied {filename} to {target_path}")
        else:
            print(f"{filename} already exists in {comp_path}, skipping.")

def rename_noise_files():
    source_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/real_data_labelled'
    for filename in os.listdir(source_path): 
        modType = filename.split('_')[0]
        freq = filename.split('_')[1] 
        if filename.startswith('8pcsk'):
            new_filename = '8cpsk_' + '_'.join(filename.split('_')[1:])
            # breakpoint()
            os.rename(os.path.join(source_path, filename), os.path.join(source_path, new_filename))
        # if not ((modType == '8pcsk' and freq == 'f70M') or 
        #         (modType == '16qam' and freq == 'f270M') or
        #         (modType == 'fm' and freq == 'f70M')):
        #     new_filename = 'unknown_' + filename
        #     os.rename(os.path.join(source_path, filename), os.path.join(source_path, new_filename))

def visualize_constellation():
    folder_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/original_testing'
    # file_name = 'fm_f70M_fs56M_s10k_snr10_fadingRician_x99.csv'
    file_name = '16qam_f270M_200k_55.csv'
    # file_name = '8cpsk_f70M_s200k_x93.csv'
    # file_name = 'fm_f70M_s200k_sig1_45.csv'
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, skiprows=10, header=None)  # Adjust skiprows as needed
    I = df.iloc[:, 0].values
    Q = df.iloc[:, 1].values
    plt.figure(figsize=(8, 8))
    plt.scatter(I, Q, s=1, alpha=0.5)
    plt.title('Constellation Diagram')
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def sample_display_signal_data(source_dir, file_sample_size=16, signal_display_length=1000):
    """
    Samples a specified number of files from source_dir to target_dir.
    
    Args:
        source_dir (str): Path to the source directory containing signal files.
        target_dir (str): Path to the target directory where sampled files will be saved.
        sample_size (int): Number of files to sample from the source directory.
    """
    
    all_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    sampled_files = random.sample(all_files, min(file_sample_size, len(all_files)))
    plt.figure(figsize=(20, 20))
    for index, file in enumerate(sampled_files):
        file_path = os.path.join(source_dir, file)
        df = pd.read_csv(file_path, skiprows=10, header=None)  #
        I = df.iloc[:, 0].values
        Q = df.iloc[:, 1].values
        if signal_display_length is not None:
            I = I[:signal_display_length]
            Q = Q[:signal_display_length]
        plt.subplot(4, 4, index + 1)
        plt.plot(I, label='In-phase (I)', color='blue', alpha=0.7)
        plt.plot(Q, label='Quadrature (Q)', color='orange', alpha=0.7)
        modType = file.split('_')[0]
        plt.title(f'{modType} - {index + 1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dir_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/synthetic/synthetic_set1'
    # list_signal_types_in_dir(dir_path)
    source_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/test_datasets/test_jul22_combined'
    target_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/test_datasets/test_mixed_2'
    # sample_signal_data(source_path, target_path, sample_size=3000)
    # print("Sampled 3000 files from source to target directory.")
# 
    SOURCE_DIR = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/test_jul22_syn'
    TRAIN_DIR = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/train_datasets/train_jul30_ood'
    TEST_DIR = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/test_real_data_labelled'
    # train_test_splitter(SOURCE_DIR, TRAIN_DIR, TEST_DIR, train_ratio=0.8)
    list_signal_types_in_dir(TRAIN_DIR)
    # list_signal_types_in_dir(TEST_DIR)
    # visualize_fft(TRAIN_DIR, Fs=56e6)  # Adjust Fs as needed
    # delete_iq_header(TRAIN_DIR)
    # extract_real_data()
    # rename_noise_files()
    # visualize_constellation()
    # sample_display_signal_data(SOURCE_DIR, file_sample_size=16)
import os
import random
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def list_signal_types_in_dir(dir_path):
    signal_types = dict()
    for filename in os.listdir(dir_path):
        modType = filename.split('_')[0]
        modTypeWithSyn = modType + '_syn' if 'chunk' not in filename else modType
        signal_types[modTypeWithSyn] = signal_types.get(modTypeWithSyn, 0) + 1
    print("Signal Types, Counts, Percentage:")
    total_files = sum(signal_types.values())
    for modType, count in signal_types.items():
        percentage = (count / total_files) * 100
        print(f"{modType}: {count} files ({percentage:.2f}%)")
    print("Total files:", sum(signal_types.values()))

def sample_signal_data(source_dir, target_dir, sample_size=1000):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    seen = set()
    source_dir_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    for i in range(sample_size):
        random_file = random.choice(source_dir_files)
        # resample if the file already exists in the target directory 
        while random_file in os.listdir(target_dir): 
            random_file = random.choice(source_dir_files)
        source_file_path = os.path.join(source_dir, random_file)
        target_file_path = os.path.join(target_dir, random_file)
        shutil.copy(source_file_path, target_file_path)
        seen.add(random_file)
    return

def train_test_splitter(source_dir, train_dir, test_dir, train_ratio=0.8, file_filter=None, move_files=False):
    """
    Randomly split files in source_dir into train_dir and test_dir.

    Args:
        source_dir (str): Path to source folder.
        train_dir (str): Path to training folder.
        test_dir (str): Path to testing folder.
        train_ratio (float): Proportion of files to use for training (default 0.8).
        file_filter (callable, optional): Function to filter files (e.g., lambda f: f.endswith('.csv')).
        move_files (bool): If True, move files instead of copying.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    if file_filter:
        all_files = [f for f in all_files if file_filter(f)]

    random.shuffle(all_files)
    split_idx = int(train_ratio * len(all_files))
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    for f in train_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(train_dir, f)
        shutil.move(src, dst) if move_files else shutil.copy(src, dst)
    for f in test_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(test_dir, f)
        shutil.move(src, dst) if move_files else shutil.copy(src, dst)

    print(f"Copied {len(train_files)} files to {train_dir}")
    print(f"Copied {len(test_files)} files to {test_dir}")

# Example usage:
# train_test_splitter(SOURCE_DIR, TRAIN_DIR, TEST_DIR, train_ratio=0.8, file_filter=

    
    print("Sampling complete.")

def delete_iq_header(dir_path):
    """
    Checks the 11th row to see if it is a header
    """
    total_files = len(os.listdir(dir_path))
    # print(dir_path)
    for idx, filename in enumerate(os.listdir(dir_path)):
        # print(filename)
        if filename.endswith('.csv'):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            if len(lines) > 10 and 'I' in lines[10]:
                print(f"Deleting header from {filename}")
                with open(file_path, 'w') as f:
                    f.writelines(lines[:10])  # Keep the first 10 lines
                    f.writelines(lines[11:])  # Skip the 11th line
            else:
                pass
            print(f"Processed {idx + 1}/{total_files} files in {dir_path}")

def visualize_fft(folder_path, Fs):
    """
    Visualizes the FFT of CSV files in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing CSV files.
        Fs (float): Sampling frequency in Hz.
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # === Initialize a set to track seen files ===
    seen = set()
    
    # === Loop through all CSV files ===
    for filename in os.listdir(folder_path):
        if (filename.endswith('.csv') or filename.endswith('.CSV')):
            filepath = os.path.join(folder_path, filename)
            seen.add(filename.split("_")[0])
            # === Load CSV: assume I in column 0, Q in column 1 ===
            try:
                df = pd.read_csv(filepath, skiprows=10, header=None)  # Skip metadata rows if needed
            except Exception as e:
                print(f"Could not load {filename}: {e}")
                continue

            if df.shape[1] < 2:
                print(f"File {filename} does not have at least two columns.")
                continue

            I = df.iloc[:, 0].values
            Q = df.iloc[:, 1].values
            x = I + 1j * Q  # Form complex baseband signal

            # === Compute FFT ===
            N = len(x)
            X = np.fft.fftshift(np.fft.fft(x, n=N))
            freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/Fs))
            print(len(freqs), len(X))
            # breakpoint()
            magnitude = 20 * np.log10(np.abs(X) + 1e-12)  # dB scale, avoid log(0)

            # === Plot ===
            plt.figure(figsize=(10, 10))
            plt.plot(freqs / 1e3, magnitude)
            plt.plot(freqs[160000:165000] / 1e3, magnitude[160000:165000])
            plt.title(f'Frequency Spectrum of {filename}')
            plt.xlabel('Frequency (kHz)')
            plt.ylabel('Magnitude (dB)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def extract_real_data():
    source_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/synthetic/synthetic_set1'
    comp_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/synthetic/synthetic_set3'
    target_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/synthetic/real_data'
    os.makedirs(target_path, exist_ok=True)
    # Loop through source path, if file is not in comp_path, copy it to target_path
    for filename in os.listdir(source_path):
        if filename not in os.listdir(comp_path):
            source_file_path = os.path.join(source_path, filename)
            target_file_path = os.path.join(target_path, filename)
            shutil.copy(source_file_path, target_file_path)
            print(f"Copied {filename} to {target_path}")
        else:
            print(f"{filename} already exists in {comp_path}, skipping.")

def rename_noise_files():
    source_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/real_data_labelled'
    for filename in os.listdir(source_path): 
        modType = filename.split('_')[0]
        freq = filename.split('_')[1] 
        if filename.startswith('8pcsk'):
            new_filename = '8cpsk_' + '_'.join(filename.split('_')[1:])
            # breakpoint()
            os.rename(os.path.join(source_path, filename), os.path.join(source_path, new_filename))
        # if not ((modType == '8pcsk' and freq == 'f70M') or 
        #         (modType == '16qam' and freq == 'f270M') or
        #         (modType == 'fm' and freq == 'f70M')):
        #     new_filename = 'unknown_' + filename
        #     os.rename(os.path.join(source_path, filename), os.path.join(source_path, new_filename))

def visualize_constellation():
    folder_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/original_testing'
    # file_name = 'fm_f70M_fs56M_s10k_snr10_fadingRician_x99.csv'
    file_name = '16qam_f270M_200k_55.csv'
    # file_name = '8cpsk_f70M_s200k_x93.csv'
    # file_name = 'fm_f70M_s200k_sig1_45.csv'
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, skiprows=10, header=None)  # Adjust skiprows as needed
    I = df.iloc[:, 0].values
    Q = df.iloc[:, 1].values
    plt.figure(figsize=(8, 8))
    plt.scatter(I, Q, s=1, alpha=0.5)
    plt.title('Constellation Diagram')
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def sample_display_signal_data(source_dir, file_sample_size=16, signal_display_length=1000):
    """
    Samples a specified number of files from source_dir to target_dir.
    
    Args:
        source_dir (str): Path to the source directory containing signal files.
        target_dir (str): Path to the target directory where sampled files will be saved.
        sample_size (int): Number of files to sample from the source directory.
    """
    
    all_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    sampled_files = random.sample(all_files, min(file_sample_size, len(all_files)))
    plt.figure(figsize=(20, 20))
    for index, file in enumerate(sampled_files):
        file_path = os.path.join(source_dir, file)
        df = pd.read_csv(file_path, skiprows=10, header=None)  #
        I = df.iloc[:, 0].values
        Q = df.iloc[:, 1].values
        if signal_display_length is not None:
            I = I[:signal_display_length]
            Q = Q[:signal_display_length]
        plt.subplot(4, 4, index + 1)
        plt.plot(I, label='In-phase (I)', color='blue', alpha=0.7)
        plt.plot(Q, label='Quadrature (Q)', color='orange', alpha=0.7)
        modType = file.split('_')[0]
        plt.title(f'{modType} - {index + 1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dir_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/synthetic/synthetic_set1'
    # list_signal_types_in_dir(dir_path)
    source_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/test_datasets/test_jul22_combined'
    target_path = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/test_datasets/test_mixed_2'
    # sample_signal_data(source_path, target_path, sample_size=3000)
    # print("Sampled 3000 files from source to target directory.")
# 
    SOURCE_DIR = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/test_jul22_syn'
    TRAIN_DIR = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/train_datasets/train_jul30_ood'
    TEST_DIR = 'C:/Users/UserAdmin/Desktop/Jason - Signal Classification/AI Models/Data/test_real_data_labelled'
    # train_test_splitter(SOURCE_DIR, TRAIN_DIR, TEST_DIR, train_ratio=0.8)
    list_signal_types_in_dir(TRAIN_DIR)
    # list_signal_types_in_dir(TEST_DIR)
    # visualize_fft(TRAIN_DIR, Fs=56e6)  # Adjust Fs as needed
    # delete_iq_header(TRAIN_DIR)
    # extract_real_data()
    # rename_noise_files()
    # visualize_constellation()
    # sample_display_signal_data(SOURCE_DIR, file_sample_size=16)s