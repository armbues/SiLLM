import argparse

import datasets

import sillm

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Test models with the MMLU benchmark.")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("-t", "--temp", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    # Load model
    model = sillm.load(args.model)

    mmlu = datasets.load_dataset("cais/mmlu", split=['all'])