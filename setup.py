from huggingface_hub import snapshot_download
import os
from boolean_question import BoolQ

# Online repo paths
SYNQA_ONLINE    = 'mbartolo/roberta-large-synqa-ext'
POLAR_ONLINE    = 'andi611/distilbert-base-uncased-qa-boolq'
PASSAGES_ONLINE = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'

def download_model(repo_id, redownload=False):

    # this is saved in your cache! delete the cache after you're done.
    path = snapshot_download(repo_id=repo_id)
    print(f'Model saved: {repo_id} -> {path}')

if __name__=='__main__':
    # Install all pip requirements
    os.system('pip3 install -r requirements.txt')

    # Download all models/tokenizers from HuggingFace to local system (in /models)
    download_model(SYNQA_ONLINE)
    download_model(PASSAGES_ONLINE)

    # BACKUP: in case POLAR_ONLINE doesn't work well, use 'from boolean_question import BoolQ'
    # code is in https://github.com/Saadmairaj/boolean-question/blob/master/boolean_question/boolq.py
    BoolQ()