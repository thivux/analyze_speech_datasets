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
    print('done processing snr for sachnoi dataset')

        
if __name__ == '__main__': 
    sachnoi()
    # vin27()
    # vivoice(1)
    # bud500()
    # vnceleb()
    # vinbigdata()
    # vivoice()
    # vlsp()