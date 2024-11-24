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

    
if __name__ == '__main__': 
    # sachnoi()
    # vin27()
    # vivoice()
    bud500()
    # vnceleb()
    # vinbigdata()
    # vivoice()
    # vlsp()