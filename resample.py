import glob
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
    for i, path in tqdm(enumerate(source_paths), total=len(source_paths)):
        # filename = os.path.basename(path)
        filename = f'{i}.wav'
        target_path = os.path.join(target_dir, filename)
        # normalize vol & resample
        resample_audio(path, target_path)


def get_wav_files(folder_path):
    wav_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    return wav_files


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
    # repo = "capleaf/viVoice"                                                                                                                  
    # dataset = load_dataset(repo, use_auth_token='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')                                                          
    # print(dataset)
    # dataset = dataset['train']
    # random.seed(42)

    # # Randomly sample 1000 samples
    # sampled_dataset = dataset.shuffle(seed=42).select(range(1000))
    # # save the sampled dataset to a folder 
    # target_folder = "vivoice-16khz"
    # os.makedirs(target_folder, exist_ok=True)
    # for sample in tqdm(sampled_dataset):
    #     audio = sample['audio']
    #     wav = audio['array']
    #     sr = audio['sampling_rate']
    #     path = sample['path']
    #     target_path = os.path.join(target_folder, path)
    #     librosa.save(audio, target_path, sr=16000)

    # vlsp
    # with open('metadata/wav_file_list.txt', 'r') as f:
    #     wav_files = [file.strip() for file in f.readlines()]

    # # sample 1000 files
    # random.seed(42)
    # sampled_paths = random.sample(wav_files, 1000)
    # target_folder = "vlsp-8khz"
    # os.makedirs(target_folder, exist_ok=True)
    # process_files(sampled_paths, target_folder)

    # bud500 
    # same as vivoice, get 1000 samples, save to dir and then resample
    # repo = "linhtran92/viet_bud500"
    # dataset = load_dataset(repo)
    # dataset = dataset['train']
    # random.seed(42)
    # sampled_dataset = dataset.shuffle(seed=42).select(range(1000))
    # target_folder = "bud500-16khz"
    # os.makedirs(target_folder, exist_ok=True)
    # for i, sample in tqdm(enumerate(sampled_dataset), total=1000):
    #     audio = sample['audio']
    #     wav = audio['array']
    #     sr = audio['sampling_rate']
    #     path = f'{i}.wav'
    #     target_path = os.path.join(target_folder, path)
    #     sf.write(target_path, wav, sr, format='wav')

    # source_paths = glob.glob('bud500-16khz/*.wav')
    # target_folder = "bud500-8khz"
    # os.makedirs(target_folder, exist_ok=True)
    # process_files(source_paths, target_folder)    

    # vnceleb
    wav_files = get_wav_files("./vietnameceleb/")   
    # sample 1000 files 
    random.seed(42)
    sampled_paths = random.sample(wav_files, 1000)
    breakpoint()
    target_folder = "vnceleb-8khz"
    os.makedirs(target_folder, exist_ok=True)
    process_files(sampled_paths, target_folder)
    
    # vinbigdata
    # wav_files = get_wav_files("./vinbigdata/")
    # # sample 1000 files 
    # random.seed(42)
    # sampled_paths = random.sample(wav_files, 1000)
    # target_folder = "vinbigdata-8khz"
    # os.makedirs(target_folder, exist_ok=True)
    # process_files(sampled_paths, target_folder)
    