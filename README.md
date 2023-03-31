# 11611-S23-Deep-Thought
## Main repository for 11411/11611: Natural Language Processing Final Project (QA/QG)

Note:
- DO NOT upload the actual models to GitHub! Delete models/ before commiting code.

How to run and compile:
1. ```$ python3 setup.py```
    - this also runs ```$ pip3 install -r requirements.txt```
    - load all the models before running programs!
2. add ```#!/usr/bin/env python3``` to the top of scripts (answer.py and ask.py)
3. in main() of each program, set ```DEBUG=True``` or ```DEBUG=False```
4. ```$ chmod +x answer.py ask.py```
    - this makes both scripts executable
5. to run QA: ```$ python answer.py article.txt questions.txt```
6. to run QG: ```$ python ask.py article.txt N```