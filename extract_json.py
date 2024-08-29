'''
create csv files that includes metadata (duration & speaker name) of different datasets
'''

import os
from tqdm import tqdm
from datasets import load_dataset
from itertools import islice
import pandas as pd 
import json
from tqdm.contrib.concurrent import thread_map
import subprocess


def get_duration(filepath):
    """Get the duration of an audio file using ffprobe."""
    cmd = ["ffprobe", "-i", filepath, "-show_entries",
           "format=duration", "-v", "quiet", "-of", "csv=p=0"]
    try:
        dur = subprocess.check_output(
            cmd, stderr=subprocess.PIPE).decode('utf-8').strip()
        return float(dur)
    except subprocess.CalledProcessError as e:
        print(f"Error processing file {filepath}: {e.stderr.decode('utf-8')}")
    except ValueError:
        print(f"Error converting duration to float for file {filepath}, got '{dur}'")
    return 0


def sachnoi():
    metadata = []
    # with open("/lustre/scratch/client/vinai/users/thivt1/code/oneshot/artifacts/step14_tone_norm_transcript_no_multispeaker.txt", 'r') as f: 
    #     for sample in f:
    #         path, transcript, speaker, duration = sample.strip().split("|")
    #         metadata.append([path, speaker, duration])

    with open("/lustre/scratch/client/vinai/users/thivt1/code/VoiceCraft/data/sach_noi_train.json", 'r') as f:
        train_data = json.load(f)

    for sample in train_data: 
        path = sample['path']
        speaker = sample['speaker']
        duration = sample['duration']
        metadata.append([path, speaker, duration])

    with open("/lustre/scratch/client/vinai/users/thivt1/code/VoiceCraft/data/sach_noi_test.json", 'r') as f:
        test_data = json.load(f)

    for sample in test_data: 
        path = sample['path']
        speaker = sample['speaker']
        duration = sample['duration']
        metadata.append([path, speaker, duration])

    print(f'there are {len(metadata)} samples in sachnoi dataset')
    df = pd.DataFrame(metadata, columns=['path', 'speaker', 'duration'])
    df.to_csv('metadata/sachnoi.csv', index=False)
    print('done processing metadata for sachnoi dataset')


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
    # dataset = load_dataset(
    #     "parquet",
    #     data_files="./viVoice/data/*.parquet",
    #     split="train"
    # )

    dataset = load_dataset("./viVoice")
    print(dataset)

    # print(f'there are {len(dataset)} samples in vivoice dataset')
    #
    # metadata = []
    # for sample in tqdm(dataset): 
    #     audio = sample['audio']
    #     duration = len(audio['array']) / audio['sampling_rate']
    #     path = audio['path']
    #     metadata.append([path, duration])
    #
    # df = pd.DataFrame(metadata, columns=['path', 'duration'])
    # df.to_csv('metadata/vivoice.csv', index=False)
    # print('done processing metadata for vivoice dataset')


def bud500(): 
    dataset = load_dataset("linhtran92/viet_bud500")
    print(dataset)
    metadata = []

    for split in ['train', 'validation', 'test']:
        for i, sample in enumerate(tqdm(dataset[split])): 
            audio = sample['audio']
            duration = len(audio['array']) / audio['sampling_rate']
            path = f'{split}_{i}'
            metadata.append([path, duration])

    print(f'there are {len(metadata)} samples in bud500')  

    df = pd.DataFrame(metadata, columns=['path', 'duration'])
    df.to_csv('metadata/bud500.csv', index=False)
    print('done processing metadata for bud500 dataset')


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
    durations = thread_map(get_duration, wav_files, max_workers=4)

    metadata = [[file, speaker, duration] for file, speaker, duration in zip(wav_files, speakers, durations)]
    df = pd.DataFrame(metadata, columns=['path', 'speaker', 'duration'])
    df.to_csv('metadata/vnceleb.csv', index=False)
    print('done processing metadata for vnceleb dataset')


def vinbigdata(): 
    wav_files = get_wav_files("./vinbigdata/") 
    print(f'there are {len(wav_files)} files in vinbigdata dataset')
    print(wav_files[:3])
    durations = thread_map(get_duration, wav_files, max_workers=4)

    metadata = [[file, duration] for file, duration in zip(wav_files, durations)]
    df = pd.DataFrame(metadata, columns=['path', 'duration'])
    df.to_csv('metadata/vinbigdata.csv', index=False)
    print('done processing metadata for vinbigdata dataset')


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
    wav_files = find_wav_files(directory)

    # Optionally, save the list to a file
    with open('metadata/wav_file_list.txt', 'w') as f:
        for file in wav_files:
            f.write(f"{file}\n")

    print(f"\nTotal WAV files found: {len(wav_files)}")
    print("List saved to metadata/wav_file_list.txt")

    durations = thread_map(get_duration, wav_files, max_workers=4)

    metadata = [[file, duration] for file, duration in zip(wav_files, durations)]
    df = pd.DataFrame(metadata, columns=['path', 'duration'])
    df.to_csv('metadata/vlsp.csv', index=False)
    print('done processing metadata for vlsp dataset')


if __name__ == '__main__': 
    # sachnoi()
    # vin27()
    # vivoice()
    # bud500()
    # vnceleb()
    # vinbigdata()
    # vivoice()
    vlsp()
    

