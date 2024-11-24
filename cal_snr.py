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


def wada_snr(wav):
    # Direct blind estimation of the SNR of a speech signal.
    #
    # Paper on WADA SNR:
    #   http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf
    #
    # This function was adapted from this matlab code:
    #   https://labrosa.ee.columbia.edu/projects/snreval/#9

    # read input file 
    # wav, _ = librosa.load(filepath, sr=None)

    # init
    eps = 1e-10
    # next 2 lines define a fancy curve derived from a gamma distribution -- see paper
    db_vals = np.arange(-20, 101)
    g_vals = np.array([0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006, 0.41027138, 0.41052627, 0.41101024, 0.41143264, 0.41231718, 0.41337272, 0.41526426, 0.4178192 , 0.42077252, 0.42452799, 0.42918886, 0.43510373, 0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236, 0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496, 0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148, 0.81240985, 0.84219775, 0.87166406, 0.90030504, 0.92880418, 0.95655449, 0.9835349 , 1.01047155, 1.0362095 , 1.06136425, 1.08579312, 1.1094819 , 1.13277995, 1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313, 1.27103891, 1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935, 1.3605727 , 1.37345513, 1.38577122, 1.39733504, 1.40856397, 1.41959619, 1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938, 1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915, 1.5229097 , 1.528578  , 1.53389835, 1.5391211 , 1.5439065 , 1.54858517, 1.55310776, 1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767, 1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477, 1.5941969 , 1.59693155, 1.599446  , 1.60185011, 1.60408668, 1.60627134, 1.60826199, 1.61004547, 1.61192472, 1.61369656, 1.61534074, 1.61688905, 1.61838916, 1.61985374, 1.62135878, 1.62268119, 1.62390423, 1.62513143, 1.62632463, 1.6274027 , 1.62842767, 1.62945532, 1.6303307 , 1.63128026, 1.63204102])

    # peak normalize, get magnitude, clip lower bound
    wav = np.array(wav)
    wav = wav / abs(wav).max()
    abs_wav = abs(wav)
    abs_wav[abs_wav < eps] = eps

    # calcuate statistics
    # E[|z|]
    v1 = max(eps, abs_wav.mean())
    # E[log|z|]
    v2 = np.log(abs_wav).mean()
    # log(E[|z|]) - E[log(|z|)]
    v3 = np.log(v1) - v2

    # table interpolation
    wav_snr_idx = None
    if any(g_vals < v3):
        wav_snr_idx = np.where(g_vals < v3)[0].max()
    # handle edge cases or interpolate
    if wav_snr_idx is None:
        wav_snr = db_vals[0]
    elif wav_snr_idx == len(db_vals) - 1:
        wav_snr = db_vals[-1]
    else:
        wav_snr = db_vals[wav_snr_idx] + \
            (v3-g_vals[wav_snr_idx]) / (g_vals[wav_snr_idx+1] - \
            g_vals[wav_snr_idx]) * (db_vals[wav_snr_idx+1] - db_vals[wav_snr_idx])

    # Calculate SNR
    dEng = sum(wav**2)
    dFactor = 10**(wav_snr / 10)
    dNoiseEng = dEng / (1 + dFactor) # Noise energy
    dSigEng = dEng * dFactor / (1 + dFactor) # Signal energy
    snr = 10 * np.log10(dSigEng / dNoiseEng)

    return snr


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

    for sample in tqdm(train_data): 
        path = sample['path']
        speaker = sample['speaker']
        # duration = sample['duration']
        snr = wada_snr(path)
        metadata.append([path, speaker, snr])

    with open("/lustre/scratch/client/vinai/users/thivt1/code/VoiceCraft/data/sach_noi_test.json", 'r') as f:
        test_data = json.load(f)

    for sample in tqdm(test_data): 
        path = sample['path']
        speaker = sample['speaker']
        # duration = sample['duration']
        snr = wada_snr(path)
        metadata.append([path, speaker, snr])

    print(f'there are {len(metadata)} samples in sachnoi dataset')
    df = pd.DataFrame(metadata, columns=['path', 'speaker', 'snr'])
    df.to_csv('snr/sachnoi.csv', index=False)
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


def divide_list_by_idx(lst_length, x, idx):
    # Calculate the size of each sublist
    chunk_size = lst_length // x
    remainder = lst_length % x  # Handle the case where the list can't be evenly divided
    
    # Calculate start and end index for the idx-th chunk
    start_index = chunk_size * idx + min(idx, remainder)
    end_index = start_index + chunk_size + (1 if idx < remainder else 0)
    
    return (start_index, end_index)


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
    # sachnoi()
    # vin27()
    vivoice(1)
    # bud500()
    # vnceleb()
    # vinbigdata()
    # vivoice()
    # vlsp()
    

