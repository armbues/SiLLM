![sillm](https://github.com/armbues/SiLLM/assets/4117144/859002e9-d209-480b-adb2-7276cd360cbe)

# SiLLM - Silicon LLM Training & Inference Toolkit
SiLLM simplifies the process of training and running Large Language Models (LLMs) on Apple Silicon by leveraging the [MLX](https://github.com/ml-explore/mlx/) framework. Building upon the foundation provided by [MLX Examples](https://github.com/ml-explore/mlx-examples), this project introduces additional features specifically designed to enhance LLM operations with MLX in a streamlined package.

- **LLM Loading**: load LLMs for inference and training in different formats (Huggingface, Torch, GGUF, MLX)
- **LoRA Training**: train LLMs using *Low-rank Adaptation*
- **DPO Training**: train LLMs with *Direct Preference Optimization*

## Features

- Model architectures: Llama, Mistral, Mixtral, Phi-2, Gemma, Qwen2, Starcoder2
- Conversation templates: llama-2, chatml, alpaca, vicuna, gemma, phi, openchat
- API server with OpenAI compatible chat endpoints
- Loss functions for DPO: sigmoid, hinge, IPO, DPOP
- Training loss plots using matplotlib
- Perplexity calculation

## Installation

Using pip:
``` sh
pip install sillm
```

## Usage

### Examples

#### [LoRA Fine-tuning](examples/helpsteer/)
LoRA training [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) with the Nvidia [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) dataset.

#### [DPO Fine-tuning](examples/dpo-mix-7k/)
DPO training [Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat) with the [DPO Mix 7K](https://huggingface.co/datasets/argilla/dpo-mix-7k) dataset. The training consists of a supervised fine tuning (SFT) followed by direct preference optimization (DPO).

#### [MMLU Benchmark](examples/mmlu/)
Implementation of the "Massive Multitask Language Understanding" benchmark using the [MMLU](https://huggingface.co/datasets/cais/mmlu) dataset.

#### [Perplexity](examples/perplexity/)
Calculating perplexity scores for a sample [dataset](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3) of entry paragraphs from Wikipedia articles.

### Command-line interface (CLI) scripts

#### Chat:
``` sh
python -m sillm.chat /path/to/model
```
Running sillm.chat in the terminal with Gemma-2B-it on a MacBook Air M2 with 16GB memory:

https://github.com/armbues/SiLLM/assets/4117144/42e2d0f8-3bd8-44ca-9f78-8c4a885b8939

#### Server:
``` sh
python -m sillm.server /path/to/model --port 8000
```

#### LoRA Fine-tuning:
``` sh
python -m sillm.lora /path/to/model -d /path/to/dataset
```
#### DPO Fine-tuning:
``` sh
python -m sillm.dpo /path/to/model -d /path/to/dataset
```
Run the CLI scripts with the argument -h to see a print-out of all available arguments.

### Python

Minimal example of loading a model with SiLLM and generating a text completion.
``` python
import sillm

model = sillm.load("/path/to/model")
for s, _ in model.generate("On a beautiful Sunday morning,"):
    print(s, flush=True, end="")
```

## Model Support
SiLLM generally supports loading LLMs of the following model architectures/families: *Llama 2*, *Mistral*, *Mixtral*, *Gemma*, *Phi*, *Qwen 2*, *StarCoder2*.

Here is a list of models that were successfully tested with SiLLM:

| Model Family | Models/Sizes (HF) | Models/Sizes (GGUF) | Models/Sizes (MLX) |
| --- | --- | --- | --- |
| Llama-2 | | [7b-chat.Q8_0](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF), [13b-chat.Q8_0](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF) | [7b](https://huggingface.co/mlx-community/Llama-2-7b-mlx), [7b-chat](https://huggingface.co/mlx-community/Llama-2-7b-chat-mlx) |
| Mistral v0.2 | [7b-instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | [7b-instruct-v0.2.Q8_0](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) | |
| Mixtral v0.1 | | | [8x7B-Instruct](https://huggingface.co/mlx-community/Mixtral-8x7B-Instruct-v0.1) |
| Gemma | [2b](https://huggingface.co/google/gemma-2b), [2b-it](https://huggingface.co/google/gemma-7b-it), [7b](https://huggingface.co/google/gemma-7b), [7b-it](https://huggingface.co/google/gemma-7b-it) | |
| Phi-2 | [2.7b](https://huggingface.co/microsoft/phi-2) | |
| Qwen 1.5 | [7b-chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat), [14b-chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat) | |
| StarCoder2 | [3b](https://huggingface.co/bigcode/starcoder2-3b), [7b](https://huggingface.co/bigcode/starcoder2-7b), [15b](https://huggingface.co/bigcode/starcoder2-15b) | |
| CodeLlama | | [70b-instruct.Q4_0](https://huggingface.co/TheBloke/CodeLlama-70B-Instruct-GGUF), [Phind-34b-v2.Q4_0](https://huggingface.co/TheBloke/Phind-CodeLlama-34B-v2-GGUF) | |

## Roadmap

- Saving model to transformers format
- Repetition penalty for inference
- Learning rate schedulers for training
- Merging models

## License

## Acknowledgments
