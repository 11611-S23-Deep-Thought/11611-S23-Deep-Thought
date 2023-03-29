from huggingface_hub import snapshot_download
import os

# Online repo paths
SYNQA_ONLINE    = 'mbartolo/roberta-large-synqa-ext'
PASSAGES_ONLINE = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'

# local directory paths
SYNQA_LOCAL     = 'models/roberta-large-synqa-ext'
PASSAGES_LOCAL  = 'models/ms-marco-TinyBERT-L-2-v2'

def download_model(repo_id, local_dir, redownload=False):
    if not redownload and os.path.isdir(local_dir):
        print(f'{repo_id} is already saved (pass redownload=True to download it again)')
        return

    # local_dir_use_symlinks=False: no caching or symlinks; put ALL files here!
    path = snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f'Model saved: {repo_id} -> {path}')

if __name__=='__main__':
    # Download all models/tokenizers from HuggingFace to local system (in /models)
    download_model(SYNQA_ONLINE, SYNQA_LOCAL)
    download_model(PASSAGES_ONLINE, PASSAGES_LOCAL)