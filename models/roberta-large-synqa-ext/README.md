---
language:
- en
tags:
- question-answering
license: apache-2.0
datasets:
- adversarial_qa
- mbartolo/synQA
- squad
metrics:
- exact_match
- f1
model-index:
- name: mbartolo/roberta-large-synqa-ext
  results:
  - task:
      type: question-answering
      name: Question Answering
    dataset:
      name: adversarial_qa
      type: adversarial_qa
      config: adversarialQA
      split: validation
    metrics:
    - name: Exact Match
      type: exact_match
      value: 53.2
      verified: true
    - name: F1
      type: f1
      value: 64.6266
      verified: true
---

# Model Overview
This is a RoBERTa-Large QA Model trained from https://huggingface.co/roberta-large in two stages. First, it is trained on synthetic adversarial data generated using a BART-Large question generator on Wikipedia passages from SQuAD as well as Wikipedia passages external to SQuAD, and then it is trained on SQuAD and AdversarialQA (https://arxiv.org/abs/2002.00293) in a second stage of fine-tuning.

# Data
Training data: SQuAD + AdversarialQA
Evaluation data: SQuAD + AdversarialQA

# Training Process
Approx. 1 training epoch on the synthetic data and 2 training epochs on the manually-curated data.

# Additional Information
Please refer to https://arxiv.org/abs/2104.08678 for full details.