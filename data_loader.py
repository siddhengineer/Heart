# data_loader.py
import os
import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def extract_physionet_data(base_path, folders):
    data_list = []
    label_list = []

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            continue
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

        for wav_file in tqdm(wav_files, desc=f"Processing {folder}"):
            wav_path = os.path.join(folder_path, wav_file)
            hea_file = wav_file.replace('.wav', '.hea')
            hea_path = os.path.join(folder_path, hea_file)

            if not os.path.exists(hea_path):
                continue

            with open(hea_path, 'r') as f:
                lines = f.readlines()
                label = None
                for line in lines:
                    if line.startswith('#'):
                        if 'Abnormal' in line or 'abnormal' in line:
                            label = 'Abnormal'
                            break
                        elif 'Normal' in line or 'normal' in line:
                            label = 'Normal'
                            break
                if label:
                    data_list.append((wav_path, label))
                    label_list.append(label)

    print("\nPhysioNet label distribution:")
    counts = Counter(label_list)
    for k, v in counts.items():
        print(f"{k}: {v}")
    return data_list

def extract_id_from_filename(filename):
    name = filename.replace('.wav', '')
    match = re.search(r'(\d{10,})(?:_([A-Z]))?', name)
    if match:
        return match.group(1), match.group(2) if match.group(2) else ''
    return None, ''

def load_kaggle_labels(csv_path):
    df = pd.read_csv(csv_path)
    label_dict = {}
    for _, row in df.iterrows():
        fname = str(row['fname']).strip()
        id_, suffix = extract_id_from_filename(fname)
        if id_ is None:
            continue
        label_raw = row['label']
        if pd.isna(label_raw) or label_raw == '':
            continue
        label = str(label_raw).lower().strip()
        binary_label = 'Normal' if label == 'normal' else 'Abnormal'
        key = f"{id_}_{suffix}" if suffix else id_
        label_dict[key] = binary_label
    return label_dict

def extract_kaggle_data(base_path):
    all_labels = {
        **load_kaggle_labels(os.path.join(base_path, 'set_a.csv')),
        **load_kaggle_labels(os.path.join(base_path, 'set_b.csv')),
    }
    audio_dirs = ['set_a', 'set_b']
    data_list, label_list = [], []

    for audio_dir in audio_dirs:
        dir_path = os.path.join(base_path, audio_dir)
        if not os.path.exists(dir_path):
            continue
        wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]

        for wav_file in tqdm(wav_files, desc=f"Processing {audio_dir}"):
            id_, suffix = extract_id_from_filename(wav_file)
            if id_ is None:
                continue
            file_key = f"{id_}_{suffix}" if suffix else id_
            if file_key in all_labels:
                label = all_labels[file_key]
                wav_path = os.path.join(dir_path, wav_file)
                data_list.append((wav_path, label))
                label_list.append(label)

    print("\nKaggle label distribution:")
    counts = Counter(label_list)
    for k, v in counts.items():
        print(f"{k}: {v}")
    return data_list

def load_and_validate_datasets(physionet_path, physionet_folders, kaggle_path, save_path=None):
    print("\nLoading PhysioNet dataset...")
    physionet_data = extract_physionet_data(physionet_path, physionet_folders)

    print("\nLoading Kaggle dataset...")
    kaggle_data = extract_kaggle_data(kaggle_path)

    all_data = physionet_data + kaggle_data
    combined_labels = [label for _, label in all_data]
    label_counts = Counter(combined_labels)

    print("\nCombined label distribution:")
    for k, v in label_counts.items():
        print(f"{k}: {v}")

    # Visualization
    plt.figure(figsize=(10, 3))
    pd.Series(label_counts).plot(kind='bar')
    plt.title("Combined Label Distribution")
    plt.xticks(rotation=0)
    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/label_distribution.png")
    plt.close()

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"\n✅ Saved combined dataset to {save_path}")

    return all_data
