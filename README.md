![sillm](https://github.com/armbues/SiLLM/assets/4117144/859002e9-d209-480b-adb2-7276cd360cbe)

# SiLLM - Silicon LLM Training & Inference Toolkit
SiLLM simplifies the process of training and running Large Language Models (LLMs) on Apple Silicon by leveraging the [MLX](https://github.com/ml-explore/mlx/) framework. Building upon the foundation provided by [MLX Examples](https://github.com/ml-explore/mlx-examples), this project introduces additional features specifically designed to enhance LLM operations with MLX in a streamlined package.

- **LLM Loading**: load LLMs for inference and training in different formats (Huggingface, Torch, GGUF, MLX)
- **LoRA Training**: train LLMs using *Low-rank Adaptation*
- **DPO Training**: train LLMs with *Direct Preference Optimization* using different loss functions (sigmoid, hinge, IPO, DPOP)

### Features

### Roadmap

- API server (OpenAI compatible)
- Training loss plots
- Saving model to transformers format
- Repetition penalty for inference
- Learning rate schedulers for training
- Merging models

## Installation

Using pip:
```
pip install sillm
```

## Usage

### Command-line interface (CLI) scripts

Chat:
```
python -m sillm.chat /path/to/model
```
LoRA fine-tuning:
```
python -m sillm.lora /path/to/model -d /path/to/dataset
```
DPO fine-tuning:
```
python -m sillm.dpo /path/to/model -d /path/to/dataset
```
Run the CLI scripts with the argument -h to see a print-out of all available arguments.

### Python

~~~ python
import sillm

model = sillm.load("/path/to/model")
for s in model.generate("On a beautiful Sunday morning,"):
    print(s, flush=True, end="")
~~~

## Model Support

SiLLM generally supports loading LLMs of the following types: *Llama 2*, *Mistral*, *Mixtral*, *Gemma*, *Phi*, *Qwen 2*, *Starcoder 2*.

Here is a list of models that were successfully tested with SiLLM:
[Gemma 7B Instruction](https://huggingface.co/google/gemma-7b-it)

## License

## Acknowledgments