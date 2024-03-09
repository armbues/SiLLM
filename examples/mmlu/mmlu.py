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

    mmlu = datasets.load_dataset("cais/mmlu", "all", split="test")

    for entry in mmlu:
        prompt = f"Answer the multiple choice question below in the subject {entry['subject'].replace('_', ' ')}.\n\n"
        prompt += entry["question"] + "\n\n"
        prompt += "Choices:\n"
        for i, choice in enumerate(entry["choices"]):
            prompt += f"{i+1}. {choice}\n"
        prompt += "\n Answer: "