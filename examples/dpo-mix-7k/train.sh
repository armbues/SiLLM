#!/bin/sh

python download.py
python -m sillm.lora model/ -c sft.yml -vv
python -m sillm.dpo model/ -c dpo.yml -a adapters/sft/ckpt-final.safetensors -vv