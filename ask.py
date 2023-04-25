#!/usr/bin/env python3

from torch.utils.data import DataLoader
import spacy
from spacy import displacy
import nltk
import sys
from nltk import tokenize
import transformers

def format_inputs(context: str, answer: str):
    return f"{answer} \\n {context}"

nltk.download('punkt', quiet=True)
transformers.logging.set_verbosity_error()

from transformers import pipeline
from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig

model = pipeline("text2text-generation", model='Salesforce/mixqg-base', tokenizer='Salesforce/mixqg-base')
model_1 = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased-squad-qg')
tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased-squad-qg')

file_name = sys.argv[1]
total_questions = int(sys.argv[2])
num_factoid = total_questions//2

num_polar = total_questions - num_factoid

factoid_set = set()
polar_set = set()

polar = 0
factoid = 0

NER = spacy.load("en_core_web_sm")
with open(file_name) as f :
    text = f.read()  
fulldict = {}
words = text.split(" ")
for word in words :
    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in word:
        if ele in punc:
            word = word.replace(ele, "")
            
    fulldict[word.lower()] = word   


def modify(ques) :
    if ques[-1] == '?' :
        ques = ques[:-1]
    qwords = ques.split()
    result = ""
    result += qwords[0].title() + " "
    for qword in qwords[1:] :
        if qword in fulldict :
            result += fulldict[qword]
        else :
            result += qword
        result += " "
    return result[:-1] + "?"
            
            
    
all_sentences = tokenize.sent_tokenize(text)
polar_answers = ["Yes", "No"]    
    
for sentence in all_sentences :

    if polar >= num_polar and factoid >= num_factoid :
        break;

    nes = NER(sentence) #named entities
    
    if len(nes.ents) == 0 :
        continue
    
    for word in tuple(nes.ents[0]) + (polar_answers[0],) :
        if polar >= num_polar and factoid >= num_factoid :
          break;
        FACT_TO_GENERATE_QUESTION_FROM = ""
        if(isinstance(word,str) and polar < num_polar) :
          ne = word
          FACT_TO_GENERATE_QUESTION_FROM = format_inputs(sentence,ne)
          ques = model(FACT_TO_GENERATE_QUESTION_FROM)
          ques = ques[0]['generated_text']
          ques = modify(ques)
          if ques not in polar_set :
            print(ques)
            polar_set.add(ques)
            polar += 1
        else :
          if (factoid < num_factoid and not isinstance(word,str)) :
            ne = word.text
            FACT_TO_GENERATE_QUESTION_FROM = ne + " [SEP] " + sentence
            inputs = tokenizer([FACT_TO_GENERATE_QUESTION_FROM], return_tensors='pt')
            question_ids = model_1.generate(inputs['input_ids'], num_beams=5, early_stopping=True, max_length = 100)
            ques = tokenizer.batch_decode(question_ids, skip_special_tokens=True)[0]
            ques = modify(ques)      
            if ques not in factoid_set :
              print(ques)
              factoid_set.add(ques)
              factoid += 1