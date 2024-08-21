from datasets import load_dataset


def download_dataset(dataset_name):
    if dataset_name == 'bud500': 
        repo = "linhtran92/viet_bud500"
    elif dataset_name == 'vivoice': 
        repo = "capleaf/viVoice"

    # Load the dataset
    dataset = load_dataset(repo, use_auth_token='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')

    # Print a sample from the train dataset
    print(dataset) 
    print(dataset['train'][0])


if __name__ == '__main__': 
    # download_dataset('bud500') 
    download_dataset('vivoice') 

