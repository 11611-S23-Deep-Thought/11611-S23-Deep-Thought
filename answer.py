#!/usr/bin/env python3

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline
)
from sentence_transformers import CrossEncoder

from setup import *
import logging
import argparse
import os


# Define all model configs and hyperparameters
config = {
    'qa_model': SYNQA_LOCAL,
    'qa_ranker': PASSAGES_LOCAL,
    'max_length': 400,                      # max length of input sequence
    'truncation': 'only_second',            # never truncate the question, only the context
    'padding': 'max_length',                # how to pad the input sequence
    'return_overflowing_tokens': True,      # return overflowing tokens after truncation (when input sequence > max_length)
    'return_offsets_mapping': True,         # return (char_start, char_end) for each token in the input sequence
    'stride': 128,                          # number of tokens b/w truncated and overflowing sequences
    'top_c': 2,                             # top C contexts/passages to look at
    'top_a': 3,                             # top K answers to pick
    'max_answer_length': 120,               # max length of each answer
    'batch_size': 128,                      # batch size
}

# Read the text files
def read_files(article, questions):
    logging.debug('reading files...')

    # Check that text files are valid
    if not os.path.exists(article):
        raise BaseException(f"{article} does not exist")
    if not os.path.exists(questions):
        raise BaseException(f"{questions} does not exist")
    
    # Read the article (and split into useful passages)
    passages = []
    with open(article, 'r') as f:
        for a in f.read().split('\n'):
            a = a.strip()
            if len(a)==0 or a[-1] not in '.!?': # must end in real sentence
                continue
            passages.append(a)

    # Read the questions
    with open(questions, 'r') as f:
        questions = f.read().splitlines()

    # Debug: print all articles and questions
    for i, p in enumerate(passages):
        logging.debug(f'Passage {i}: {p}')
    for i, q in enumerate(questions):
        logging.debug(f'Question {i}: {q}')

    return passages, questions

# Find the most relevant passages in the article, then merge into 1 context
def reduce_article(ranker, q, passages):
    scores = ranker.predict([(q, passage) for passage in passages])
    best_idx = np.argsort(scores)[-config['top_c']:]
    context = ' '.join([passages[i] for i in best_idx])
    logging.debug(f'Relevant Context: {context}')
    return context

def main(args):

    article = args.article
    questions = args.questions
    passages, questions = read_files(article, questions)

    # Use GPU if available (it's fine if not)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    logging.debug(f'using {DEVICE}')

    # Build the QA/QG model
    logging.debug('building QA pipeline...')
    model = AutoModelForQuestionAnswering.from_pretrained(config['qa_model'])
    tokenizer = AutoTokenizer.from_pretrained(config['qa_model'])
    QA = pipeline('question-answering',
                  model=model,
                  tokenizer=tokenizer,
                  device=DEVICE,
                  framework='pt') # use PyTorch instead of TensorFlow

    # Build the passage ranker
    logging.debug('building ranker...')
    ranker = CrossEncoder(config['qa_ranker'])


    logging.debug('\n\n' + '-'*50 + '\n\n')


    for q in questions:
        # Get the list of answers (+confidences) for each question
        context = reduce_article(ranker, q, passages)
        answers = QA(question=q, context=context, top_k=config['top_a'])
        for a in answers:
            logging.debug('Candidate: %s (%.3f)' % (a['answer'], a['score']))
        
        # Print the best answer
        a = answers[0]['answer']
        logging.debug(f'Q: {q}')
        logging.debug(f'A: {a}')
        logging.debug('- '*10)
        print(a)

if __name__=='__main__':

    # Define logging level
    DEBUG = True # print debug messages?
    level = logging.DEBUG if DEBUG else logging.WARNING
    logging.basicConfig(level=level)

    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('article')   # define first argument
    parser.add_argument('questions') # define second argument

    args = parser.parse_args()
    main(args)