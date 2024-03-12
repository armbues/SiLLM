import argparse

import datasets

import mlx.core as mx

import sillm
import sillm.utils as utils

########
# MMLU benchmark implementation using MLX via SiLLM.
# References:
# https://github.com/hendrycks/test
# https://huggingface.co/datasets/cais/mmlu
########
if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Test models with the MMLU benchmark.")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()

    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)

    # Load model
    model = sillm.load(args.model)

    # Load MMLU dataset
    mmlu = datasets.load_dataset("cais/mmlu", "all", split="test")

    # Initialize counters
    num_total = { "all": 0 }
    num_correct = { "all": 0 }
    accuracy = { "all": 0.0 }

    choices = ["A", "B", "C", "D"]
    choices_ids = [model.tokenizer.encode(choice, bos=False)[0] for choice in choices]

    for entry in mmlu.shuffle():
        subject = entry["subject"].replace('_', ' ')
        answer = entry["answer"]

        # Assemble prompt
        prompt = f"Answer the multiple choice question below in the subject {subject}. Only respond with the letter corresponding to the correct answer.\n\n"
        prompt += entry["question"].lstrip() + "\n\n"
        prompt += "Choices:\n"
        for i, choice in enumerate(entry["choices"]):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nAnswer:\n"

        # Predict answer
        y = mx.array(model.tokenizer.encode(prompt))
        logits, _ = model.model(y[None])
        logits = logits[:, -1, :]
        choices_logits = mx.take(logits, mx.array(choices_ids))
        choices_prob = mx.softmax(choices_logits)
        pred = mx.argmax(choices_prob).item()

        # Print prompt and prediction
        print(f"==== {subject} ====")
        print(prompt, end="")
        print(choices[pred])

        # Update counters
        num_total["all"] += 1
        num_total.setdefault(subject, 0)
        num_correct.setdefault(subject, 0)
        num_total[subject] += 1
        
        accuracy.setdefault(subject, 0.0)
        pred_acc = choices_prob[answer].item()
        accuracy["all"] += pred_acc
        accuracy[subject] += pred_acc
    
        if pred == answer:
            num_correct["all"] += 1
            num_correct[subject] += 1

            print(f"\n => CORRECT!  \t{choices[answer]} ({pred_acc:.2f})\ttotal: {num_correct['all']}/{num_total['all']}")
        else:
            print(f"\n => INCORRECT!\t{choices[answer]} ({pred_acc:.2f})\ttotal: {num_correct['all']}/{num_total['all']}")