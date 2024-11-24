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


def vivoice():
    paths = glob.glob('vivoice-8khz/*.wav')
    los = []
    his = []
    for path in tqdm(paths): 
        lo, hi = cal_sisnr(path)
        los.append(lo.item())
        his.append(hi.item())

    metadata = [[path, lo, hi] for path, lo, hi in zip(paths, los, his)]
    df = pd.DataFrame(metadata, columns=['path', 'sisnr-low', 'sisnr-high'])
    df.to_csv(f'sisnr/vivoice.csv', index=False)
    print('done processing si-snr for vivoice dataset')


def bud500(): 
    paths = glob.glob('bud500-8khz/*.wav')
    los = []
    his = []
    for path in tqdm(paths): 
        lo, hi = cal_sisnr(path)
        los.append(lo.item())
        his.append(hi.item())

    metadata = [[path, lo, hi] for path, lo, hi in zip(paths, los, his)]
    df = pd.DataFrame(metadata, columns=['path', 'sisnr-low', 'sisnr-high'])
    df.to_csv(f'sisnr/bud500.csv', index=False)
    print('done processing si-snr for bud500 dataset')


def get_wav_files(folder_path):
    wav_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    return wav_files


def vnceleb(): 
    paths = glob.glob('vnceleb-8khz/*.wav')
    los = []
    his = []
    for path in tqdm(paths): 
        lo, hi = cal_sisnr(path)
        los.append(lo.item())
        his.append(hi.item())

    metadata = [[path, lo, hi] for path, lo, hi in zip(paths, los, his)]
    df = pd.DataFrame(metadata, columns=['path', 'sisnr-low', 'sisnr-high'])
    df.to_csv(f'sisnr/vnceleb.csv', index=False)
    print('done processing si-snr for vnceleb dataset')


def vinbigdata(): 
    paths = glob.glob('vinbigdata-8khz/*.wav')
    los = []
    his = []
    for path in tqdm(paths): 
        lo, hi = cal_sisnr(path)
        los.append(lo.item())
        his.append(hi.item())

    metadata = [[path, lo, hi] for path, lo, hi in zip(paths, los, his)]
    df = pd.DataFrame(metadata, columns=['path', 'sisnr-low', 'sisnr-high'])
    df.to_csv(f'sisnr/vinbigdata.csv', index=False)
    print('done processing si-snr for vinbigdata dataset')


def find_wav_files(directory):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files


def vlsp():
    paths = glob.glob('vlsp-8khz/*.wav')
    los = []
    his = []
    for path in tqdm(paths): 
        lo, hi = cal_sisnr(path)
        los.append(lo.item())
        his.append(hi.item())

    metadata = [[path, lo, hi] for path, lo, hi in zip(paths, los, his)]
    df = pd.DataFrame(metadata, columns=['path', 'sisnr-low', 'sisnr-high'])
    df.to_csv(f'sisnr/vlsp.csv', index=False)
    print('done processing si-snr for vlsp dataset')


if __name__ == '__main__': 
    # sachnoi()
    # sachnoi(2)
    # vin27()
    # vivoice(1)
    # bud500()
    vnceleb()
    # vinbigdata()
    # vivoice()
    # vlsp()
