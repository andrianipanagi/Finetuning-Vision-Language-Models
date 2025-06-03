# Finetuning-Vision-Language-Models

This repository contains code and instructions for fine-tuning vision-language models on visual question answering (VQA) tasks using custom VQA datasets.

## Features
 - Fine-tunes pre-trained vision-language models (e.g., Florence2, SmolVLM) on VQA datasets
 - Support for custom image-question-answer datasets
 - Training, and evaluation scripts

## Model Support
 - Microsoft/Florence2
 - HuggingFace/SmolVLM

## Dataset Format
The dataset should consist of entries with:
  - image: file path or image object
  - question: text string
  - answer: target answer (string or list of acceptable answers)
    
```
Example (JSON format):
{
  "image": "images/0001.jpg",
  "question": "How many people are sitting?",
  "answer": "The number of people sitting in the bus is 23."
}
```

