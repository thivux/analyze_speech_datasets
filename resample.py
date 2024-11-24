import random
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import os
import subprocess
import librosa
import soundfile as sf


def resample_audio(source_path, target_path):
    # Command to resample, trim silence, and normalize volume using sox
    cmd = (f'sox "{source_path}" -r 8000 "{target_path}"')
    #    silence 1 1 0.2 reverse silence 1 1 0.2 reverse')
    subprocess.run(cmd, shell=True, check=True)


def process_files(source_paths, target_dir):
    # trim & norm vol of inference results
    for path in tqdm(source_paths):
        filename = os.path.basename(path)
        target_path = os.path.join(target_dir, filename)
        # normalize vol & resample
        resample_audio(path, target_path)


if __name__ == "__main__":
    # sachnoi
    # rootdir = '/lustre/scratch/client/vinai/users/thivt1/code/oneshot'
    # data = pd.read_csv(os.path.join(rootdir, 'artifacts/step19_augmented_train.csv'), sep='|', names=['path', 'text', 'speaker', 'duration'])
    # data = data.sample(1000, random_state=42)
    # source_paths = data['path'].tolist()
    # source_paths = [os.path.join(rootdir, path) for path in source_paths]

    # target_folder = "sachnoi-8khz"
    # os.makedirs(target_folder, exist_ok=True)
    # process_files(source_paths, target_folder)

    # vivoice 
    repo = "capleaf/viVoice"                                                                                                                  
    dataset = load_dataset(repo, use_auth_token='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')                                                          
    print(dataset)
    dataset = dataset['train']
    random.seed(42)

    # Randomly sample 1000 samples
    sampled_dataset = dataset.shuffle(seed=42).select(range(1000))
    # save the sampled dataset to a folder 
    target_folder = "vivoice-16khz"
    os.makedirs(target_folder, exist_ok=True)
    for sample in tqdm(sampled_dataset):
        audio = sample['audio']
        wav = audio['array']
        sr = audio['sampling_rate']
        path = sample['path']
        target_path = os.path.join(target_folder, path)
        librosa.save(audio, target_path, sr=16000)

