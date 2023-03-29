from huggingface_hub import snapshot_download


if __name__=='__main__':
    # Download all models/tokenizers from HuggingFace
    path = snapshot_download(repo_id='mbartolo/roberta-large-synqa-ext', # download all files ...
                             local_dir='models/roberta-large-synqa-ext', # ... to local system
                             local_dir_use_symlinks=False) # no caching or symlinks; get ALL files!
    print(path)