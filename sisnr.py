'''
create csv files that includes metadata (duration & speaker name) of different datasets
'''

import numpy as np
import os
from tqdm import tqdm
from datasets import load_dataset
from itertools import islice
import pandas as pd 
import json
from tqdm.contrib.concurrent import thread_map
import subprocess
import librosa
import glob

import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator
from speechbrain.inference.metrics import SNREstimator as snrest


model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr', run_opts={"device":"cuda"}) 
snr_est_model = snrest.from_hparams(source="speechbrain/REAL-M-sisnr-estimator",savedir='pretrained_models/REAL-M-sisnr-estimator', run_opts={"device":"cuda"})


def divide_list_by_idx(lst_length, x, idx):
    # Calculate the size of each sublist
    chunk_size = lst_length // x
    remainder = lst_length % x  # Handle the case where the list can't be evenly divided
    # Calculate start and end index for the idx-th chunk
    start_index = chunk_size * idx + min(idx, remainder)
    end_index = start_index + chunk_size + (1 if idx < remainder else 0)
    return (start_index, end_index)


def cal_sisnr(path):
    est_source = model.separate_file(path=path, savedir='speechbrain-sisnr')
    mix, _ = torchaudio.load(path)
    mix = mix.to("cuda")
    snrhat = snr_est_model.estimate_batch(mix, est_source)
    return snrhat


def sachnoi():
    paths = glob.glob('sachnoi-8khz/*.wav')
    los = []
    his = []
    for path in tqdm(paths): 
        lo, hi = cal_sisnr(path)
        los.append(lo.item())
        his.append(hi.item())

    metadata = [[path, lo, hi] for path, lo, hi in zip(paths, los, his)]
    df = pd.DataFrame(metadata, columns=['path', 'sisnr-low', 'sisnr-high'])
    df.to_csv(f'sisnr/sachnoi.csv', index=False)
    print('done processing si-snr for sachnoi dataset')


def vin27():
    metadata = []

    with open("/lustre/scratch/client/vinai/users/thivt1/code/TTS/recipes/ljspeech/xtts_v2/VIN27/full_metadata.csv", 'r') as f: 
        data = json.load(f)

    for sample in tqdm(data): 
        metadata.append([sample['path'], sample['speaker'], sample['duration']])

    print(f'there are {len(metadata)} samles in vin27 dataset')

    df = pd.DataFrame(metadata, columns=['path', 'speaker', 'duration'])
    df.to_csv('metadata/vin27.csv', index=False)
    print('done processing metadata for vin27 dataset')


def vivoice(idx):
    # dataset = load_dataset(
    #     "parquet",
    #     data_files="./viVoice/data/*.parquet",
    #     split="train"
    # )

    # Load the dataset                                                                                                                            
    repo = "capleaf/viVoice"                                                                                                                  
    dataset = load_dataset(repo, use_auth_token='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')                                                          
    print(dataset)
    dataset = dataset['train']
    print(f'there are {len(dataset)} samples in vivoice dataset')
    
    n = len(dataset)
    stop_idx = n // 2
    # start, end = divide_list_by_idx(n, 20, idx)
    # print(f'start: {start}, end: {end}')
    # subset = dataset[start:end]
    # print(f'processing {len(subset)} samples in this subset')
    metadata = []
    for i, sample in tqdm(enumerate(dataset), total=n): 
        if i < stop_idx:
            continue 
        audio = sample['audio']
        wav = audio['array']
        snr = wada_snr(wav)
        path = audio['path']
        metadata.append([path, snr])

    df = pd.DataFrame(metadata, columns=['path', 'snr'])
    df.to_csv(f'snr/vivoice_{idx}.csv', index=False)
    print('done processing snr for vivoice dataset')


def bud500(): 
    dataset = load_dataset("linhtran92/viet_bud500")
    print(dataset)
    metadata = []

    for split in ['train', 'validation', 'test']:
        for i, sample in enumerate(tqdm(dataset[split])): 
            audio = sample['audio']
            # duration = len(audio['array']) / audio['sampling_rate']
            snr = wada_snr(audio['array'])
            path = f'{split}_{i}'
            metadata.append([path, snr])

    print(f'there are {len(metadata)} samples in bud500')  

    df = pd.DataFrame(metadata, columns=['path', 'snr'])
    df.to_csv('snr/bud500.csv', index=False)
    print('done processing snr for bud500 dataset')


def get_wav_files(folder_path):
    wav_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    return wav_files


def vnceleb(): 
    wav_files = get_wav_files("./vietnameceleb/") 
    print(f'there are {len(wav_files)} files in vnceleb dataset')
    speakers = [file.split("/")[-2] for file in wav_files]
    durations = thread_map(wada_snr, wav_files, max_workers=4)

    metadata = [[file, speaker, duration] for file, speaker, duration in zip(wav_files, speakers, durations)]
    df = pd.DataFrame(metadata, columns=['path', 'speaker', 'snr'])
    df.to_csv('snr/vnceleb.csv', index=False)
    print('done processing snr for vnceleb dataset')


def vinbigdata(): 
    wav_files = get_wav_files("./vinbigdata/") 
    print(f'there are {len(wav_files)} files in vinbigdata dataset')
    print(wav_files[:3])
    snrs = thread_map(wada_snr, wav_files, max_workers=4)

    metadata = [[file, duration] for file, duration in zip(wav_files, snrs)]
    df = pd.DataFrame(metadata, columns=['path', 'snr'])
    df.to_csv('snr/vinbigdata.csv', index=False)
    print('done processing snr for vinbigdata dataset')


def find_wav_files(directory):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files


def vlsp():
    directory = "/lustre/scratch/client/vinai/users/thivt1/data/VLSP_data"

    # Get the list of WAV files
    # wav_files = find_wav_files(directory)

    # Optionally, save the list to a file
    # with open('metadata/wav_file_list.txt', 'w') as f:
    #     for file in wav_files:
    #         f.write(f"{file}\n")
    with open('metadata/wav_file_list.txt', 'r') as f:
        wav_files = [file.strip() for file in f.readlines()]

    print(f"\nTotal WAV files found: {len(wav_files)}")
    print("List saved to metadata/wav_file_list.txt")

    durations = thread_map(wada_snr, wav_files, max_workers=4)

    metadata = [[file, duration] for file, duration in zip(wav_files, durations)]
    df = pd.DataFrame(metadata, columns=['path', 'snr'])
    df.to_csv('snr/vlsp.csv', index=False)
    print('done processing snr for vlsp dataset')


if __name__ == '__main__': 
    sachnoi()
    # sachnoi(2)
    # vin27()
    # vivoice(1)
    # bud500()
    # vnceleb()
    # vinbigdata()
    # vivoice()
    # vlsp()
