# Examples for using SiLLM

Make sure to install the additional requirements for the examples:
``` sh
pip install -r requirements-examples.txt
```

## HelpSteer LoRA
LoRA training [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) with the Nvidia [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) dataset.

Run `train.sh` in the `helpsteer` directory to download the dataset & model from HuggingFace and start the LoRA training. You can customize the training configuration by editing `config.yml`.

## DPO-Mix-7K
DPO training [Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat) with the [DPO Mix 7K](https://huggingface.co/datasets/argilla/dpo-mix-7k) dataset. The training consists of a supervised fine tuning (SFT) followed by direct preference optimization (DPO).

Run `train.sh` in the `dpo-mix-7k` directory to download the dataset & model from HuggingFace and start the training. You can customize the training configuration by editing the config files `sft.yml` and `dpo.yml`.

## MMLU Benchmark
Implementation of the "Massive Multitask Language Understanding" benchmark using the [MMLU](https://huggingface.co/datasets/cais/mmlu) dataset.

Run `mmlu.py` with the model you would like to evaluate.

## Perplexity
Calculating perplexity scores for a sample [dataset](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3) of entry paragraphs from Wikipedia articles.

Run `perplexity.py` with the model you would like to evaluate. Add quantization options to evaluate perplexity with quantized models.