# LLM4KGC

This repository contains scripts and code for a research project exploring Text-to-Knowledge Graph (T2KG) construction using Large Language Models (LLMs). The project investigates three key methods: Zero-Shot Prompting (ZSP), Few-Shot Prompting (FSP), and Fine-Tuning (FT).

## Overview
The goal of this project is to evaluate the efficacy of various approaches in constructing knowledge graphs from textual data. We focus on LLMs such as Llama2, Mistral, and Starling, comparing their performance across different methods and datasets.

## Dependencies
#### Install necessary dependencies (run this once per machine)
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q -U datasets scipy ipywidgets matplotlib

## Key Features

- **Methods:** ZSP, FSP, FT
- **LLMs:** Llama2, Mistral, Starling
- **Datasets:** WebNLG+2020 (testing), training dataset of WebNLG+2020 (fine-tuning)

## fine-tuning

You can fine-tune LLMs using the following :

 - bash fine_tune.sh

## Inference

We also provide the inference scripts to directly acquire the generation results on the test set

- bash gen_with_llms.sh

## Evaluation

- eval.sh

## Structure

- `Generate_KGs_with_LLMs/`: Code for generate KGs from text with Zero-shot and 7-shots
- `webnlg_data/`: Contains trainging, validation and test datasets of webnlg+2020
- `fine-tune/`: Contains fine-tuning LLMs script
- `Evaluate_graphs/`: contains the graph evaluation metrics


## Results


## Future Work
    - Enhance evaluation metrics for synonyms
    - Investigate LLMs for data augmentation
    - Evaluate fine-tuned models on diverse datasets

## License
