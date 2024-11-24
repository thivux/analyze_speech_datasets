'''
calculate the speaking rate (wpm) of different datasets
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
import string


def normalize_transcript(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))


def sachnoi():
    # load data
    with open("/lustre/scratch/client/vinai/users/thivt1/code/VoiceCraft/data/sach_noi_train.json", 'r') as f:
        train_data = json.load(f)

    with open("/lustre/scratch/client/vinai/users/thivt1/code/VoiceCraft/data/sach_noi_test.json", 'r') as f:
        test_data = json.load(f)

    data = train_data + test_data 

    # get wpm
    metadata = []
    for sample in data: 
        transcript = sample['transcript']
        transcript = normalize_transcript(transcript)
        duration = float(sample['duration'])
        wps = len(transcript.split()) / duration
        wpm = wps * 60
        metadata.append([sample['path'], transcript, sample['speaker'], duration, wpm])

    df = pd.DataFrame(metadata, columns=['path', 'transcript', 'speaker', 'duration', 'wpm'])
    df.to_csv('wpm/sachnoi.csv', index=False)
    print('done processing speaking rate for sachnoi dataset')

        
def vivoice():
    repo = "capleaf/viVoice"                                                                                                                  
    dataset = load_dataset(repo, use_auth_token='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')                                                          
    print(dataset)
    dataset = dataset['train']
    print(f'there are {len(dataset)} samples in vivoice dataset')

    metadata = []
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)): 
        audio = sample['audio']
        duration = len(audio['array']) / audio['sampling_rate']
        path = audio['path']
        transcript = sample['text']
        transcript = normalize_transcript(transcript)
        wps = len(transcript.split()) / duration
        wpm = wps * 60
        metadata.append([path, transcript, duration, wpm])
    
    df = pd.DataFrame(metadata, columns=['path', 'transcript', 'duration', 'wpm'])
    df.to_csv('wpm/vivoice.csv', index=False)
    print('done processing speaking rate for sachnoi dataset')

    
def bud500(): 
    dataset = load_dataset("linhtran92/viet_bud500")
    print(dataset)
    metadata = []

    for split in ['train', 'validation', 'test']:
        for i, sample in enumerate(tqdm(dataset[split])): 
            audio = sample['audio']
            transcript = sample['transcription']
            transcript = normalize_transcript(transcript)
            duration = len(audio['array']) / audio['sampling_rate']
            wps = len(transcript.split()) / duration
            wpm = wps * 60
            path = f'{split}_{i}'
            metadata.append([path, transcript, duration, wpm])

    df = pd.DataFrame(metadata, columns=['path', 'transcript', 'duration', 'wpm'])
    df.to_csv('wpm/bud500.csv', index=False)
    print('done processing speaking rate for bud500 dataset')

    
def get_wav_files(folder_path):
    wav_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    return wav_files


def vinbigdata():
    data = pd.read_csv('metadata/vinbigdata.csv')

    wpms = []
    for i, row in tqdm(data.iterrows()): 
        wav_path, duration = row['path'], row['duration']
        txt_path = wav_path.replace('.wav', '.txt')
        with open(txt_path, 'r') as f:
            transcript = f.read()
        transcript = normalize_transcript(transcript)
        wps = len(transcript.split()) / float(duration)
        wpm = wps * 60
        wpms.append(wpm)
    
    data['wpm'] = wpms
    data.to_csv('wpm/vinbigdata.csv', index=False)
    print('done processing speaking rate for vinbigdata dataset')

    
def vlsp(): 
    # duration mapping 
    data = pd.read_csv('metadata/vlsp.csv')
    duration_map = {}
    for i, row in data.iterrows():
        duration_map[row['path']] = row['duration']

    # ==============================================
    rootdir = '/home/thivt1/data/VLSP_data'
    dev2021 = os.path.join(rootdir, 'VLSP-ASR-2021-labeled_devset')

    # read the transcript file 
    transcript_file = os.path.join(dev2021, 'transcript.txt')
    dev2021_map = {}
    with open(transcript_file, 'r') as f:
        for line in f: 
            name, transcript = line.strip().split('\t')
            dev2021_map[name + ".wav"] = normalize_transcript(transcript)
    
    # list file in the wav folder
    wav_files = get_wav_files(os.path.join(dev2021, 'wav'))
    wav_files = [file.replace("/home", '/lustre/scratch/client/vinai/users') for file in wav_files]
    metadata1 = []
    for file in tqdm(wav_files): 
        name = os.path.basename(file)
        duration = duration_map[file]
        transcript = dev2021_map[name]
        wps = len(transcript.split()) / float(duration)
        wpm = wps * 60
        metadata1.append([name, transcript, duration, wpm])

    print(len(metadata1))

    # ==============================================
    train2021 = os.path.join(rootdir, 'VLSP-ASR-2021-labeled_devset')

    # read the transcript file 
    transcript_file = os.path.join(train2021, 'transcript.txt')
    train2021_map = {}
    with open(transcript_file, 'r') as f:
        for line in f: 
            name, transcript = line.strip().split('\t')
            train2021_map[name + ".wav"] = normalize_transcript(transcript)
    
    # list file in the wav folder
    wav_files = get_wav_files(os.path.join(train2021, 'wav'))
    wav_files = [file.replace("/home", '/lustre/scratch/client/vinai/users') for file in wav_files]
    metadata2 = []
    for file in tqdm(wav_files): 
        name = os.path.basename(file)
        duration = duration_map[file]
        transcript = train2021_map[name]
        wps = len(transcript.split()) / float(duration)
        wpm = wps * 60
        metadata2.append([name, transcript, duration, wpm])

    print(len(metadata2))

    # ==============================================
    wav_files1 = get_wav_files(os.path.join(rootdir, "vlsp2020_train_set_02"))
    wav_files2 = get_wav_files(os.path.join(rootdir, "zalo"))
    wav_files = wav_files1 + wav_files2
    wav_files = [file.replace("/home", '/lustre/scratch/client/vinai/users') for file in wav_files]
    metadata3 = []
    for file in tqdm(wav_files): 
        name = os.path.basename(file)
        duration = duration_map[file]
        txt_file = file.replace('.wav', '.txt')
        with open(txt_file, 'r') as f:
            transcript = f.read()
        transcript = normalize_transcript(transcript)
        wps = len(transcript.split()) / float(duration)
        wpm = wps * 60
        metadata3.append([name, transcript, duration, wpm])
    
    print(len(metadata3))


if __name__ == '__main__': 
    # sachnoi()
    # vin27()
    # vivoice()
    # bud500()
    # vinbigdata()
    # vivoice()
    vlsp()