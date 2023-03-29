#!/usr/bin/env python3

""" BEGIN: import all necessary NLP modules here! """
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline
)

""" END """

import argparse
import os

# Define all model hyperparameters here (makes it easier to use)
config = {
    'model': 'models/roberta-large-synqa-ext', # SynQA EXT
    "max_length": 400, # max length of input sequence (otherwise, use model's default)
    "truncation": 'only_second', # never truncate the question, only the context
    "padding": 'max_length', # how to pad the input sequence
    "return_overflowing_tokens": True, # return overflowing tokens when after truncation (input sequence > max_length)
    "return_offsets_mapping": True, # return (char_start, char_end) for each token in the input sequence
    "stride": 128, # number of tokens b/w truncated and overflowing sequences
    "n_best_size": 20, # top N answers to pick
    "max_answer_length": 120, # max length of each answer
    "batch_size": 128, # batch size
}

def main(args):

    article = args.article
    questions = args.questions

    # Check that text files are valid
    if not os.path.exists(article):
        raise BaseException(f"{article} does not exist")
    if not os.path.exists(questions):
        raise BaseException(f"{questions} does not exist")

    # Read the article
    with open(article, 'r') as f:
        article = f.read().replace('\n', ' ')
    print('Article:\n', article[:300])

    # Read the questions
    with open(questions, 'r') as f:
        questions = f.read().splitlines()
    print('Questions:\n', questions)

    # Use GPU if available (it's fine if not)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print(f'using {DEVICE}')

    # Build the QA/QG model
    model = AutoModelForQuestionAnswering.from_pretrained(config['model'])
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    QA = pipeline('question-answering',
                  model=model,
                  tokenizer=tokenizer,
                  device=DEVICE,
                  framework='pt') # use PyTorch instead of TensorFlow

    # Loop over the questions
    for q in questions:
        print('Q:', q)
        answer = QA(question=q, context=article, top_k=3)
        print('A:', answer)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('article')   # define first argument
    parser.add_argument('questions') # define second argument

    args = parser.parse_args()
    main(args)