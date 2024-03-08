# Examples for using SiLLM

Make sure to install the additional requirements for the examples:
``` sh
pip install -r requirements-examples.txt
```

## HelpSteer LoRA
LoRA Training [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) with the Nvidia [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) dataset.

Run `train.sh` in the `helpsteer` directory to download the HelpSteer dataset/model from HuggingFace and start the LoRA training. You can customize the training configuration by editing `config.yaml`.