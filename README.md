# 11611-S23-Deep-Thought
## Main repository for 11411/11611: Natural Language Processing Final Project (QA/QG)

Note:
- DO NOT upload the actual models to GitHub! Delete models/ before commiting code.

How to run and compile:
1. ```$ pip install -r requirements.txt```
2. ```$ python setup.py```
    - load all the models locally to models/ before running programs!
3. add ```#!/usr/bin/env python3``` to the top of scripts (answer.py and ask.py)
4. in main() of each program, set ```DEBUG=True``` or ```DEBUG=False```
5. ```$ chmod +x answer.py ask.py```
    - this makes both scripts executable
6. to run QA: ```$ python answer.py article.txt questions.txt```
7. to run QG: ```$ python ask.py article.txt N```