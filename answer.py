#!/usr/bin/env python3

import numpy as np
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    pipeline
)
from sentence_transformers import CrossEncoder
from boolean_question import BoolQ

from setup import *
import logging
import argparse
import sys
import os


# Define all model configs and hyperparameters
config = {
    'qa_model1': SYNQA_ONLINE,               # model for regular questions
    'qa_model2': POLAR_ONLINE,               # model for yes/no questions
    'qa_ranker': PASSAGES_ONLINE,
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
            if len(a)==0 or a[-1] not in '.!?"\'': # must end in real sentence or quote
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

# Build QA pipeline
def build_pipeline(model_name, DEVICE):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    QA = pipeline('question-answering',
                  model=model,
                  tokenizer=tokenizer,
                  device=DEVICE,
                  framework='pt') # use PyTorch instead of TensorFlow
    return QA

def build_pipeline2(model_name, DEVICE):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    QA = pipeline('text-classification',
                  model=model,
                  tokenizer=tokenizer,
                  device=DEVICE,
                  framework='pt'
                  ) # use PyTorch instead of TensorFlow
    return QA

def build_pipeline3():
    transformers.logging.set_verbosity_error() # silence logging warnings
    return BoolQ()

# Find the most relevant passages in the article, then merge into 1 context
def reduce_article(ranker, q, passages):
    scores = ranker.predict([(q, passage) for passage in passages])
    best_idx = np.argsort(scores)[-config['top_c']:]
    context = ' '.join([passages[i] for i in best_idx])
    logging.debug(f'Relevant Context: {context}')
    return context

# Determine is question is polar or not
def is_polar(q):
    wh_questions = ['what', 'when', 'where', 'who', 'whom', 'which', 'whose', 'why', 'how']
    first_word = q.split()[0].lower()
    return first_word not in wh_questions
    

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

    # Build the QA models
    logging.debug('building first QA pipeline...') # for regular wh- questions
    QA1 = build_pipeline(config['qa_model1'], DEVICE)
    logging.debug('building second QA pipeline...') # for polar questions
    # QA2 = build_pipeline2(config['qa_model2'], DEVICE)
    b = build_pipeline3()

    # Build the passage ranker
    logging.debug('building ranker...')
    ranker = CrossEncoder(config['qa_ranker'])


    logging.debug('\n\n' + '-'*50 + '\n\n')


    for q in questions:

        # Get the most relevant passages
        context = reduce_article(ranker, q, passages)

        # First identify the question type
        # Then get the list of answers (+confidences) for each question
        if not is_polar(q):
            logging.debug('Not polar: using QA1')
            answers = QA1(question=q, context=context, top_k=config['top_a'])
            
            for a in answers:
                logging.debug('Candidate: %s (%.3f)' % (a['answer'], a['score']))
            a = answers[0]['answer']
        else:
            logging.debug('Polar: using QA2')
            a = 'Yes' if b.predict(context, q) else 'No'
            data = b.prediction_details()
            logging.debug('Candidate: %s (True: %.3f, False: %.3f)' %
                          (data['answer'], data['true confidence'], data['false confidence']))
        
        
        # Print the best answer
        logging.debug(f'Q: {q}')
        logging.debug(f'A: {a}')
        logging.debug('- '*10)
        print(a)

if __name__=='__main__':

    # Define logging level
    DEBUG = False # print debug messages?
    level = logging.DEBUG if DEBUG else logging.WARNING
    logging.basicConfig(level=level)

    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('article')   # define first argument
    parser.add_argument('questions') # define second argument

    args = parser.parse_args()
    main(args)