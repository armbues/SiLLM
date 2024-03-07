#!/bin/sh

python download.py
python -m sillm.lora model/ -c config.yaml -vv