import argparse
import pathlib

import huggingface_hub

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Downloader for Hugging Face models.")
    parser.add_argument("model", type=str, help="The model identifier (e.g. microsoft/Phi-3-medium-4k-instruct)")
    parser.add_argument("output", type=str, help="Output directory for model files")
    parser.add_argument("--token", default=None, type=str, help="Hugging Face API token")
    parser.add_argument("--proxy", default=None, type=str, help="Proxy URL")
    args = parser.parse_args()

    if args.token:
        huggingface_hub.login(args.token)

    model_path = pathlib.Path(args.output)
    model_path.mkdir(parents=True, exist_ok=True)

    allow_patterns = ["*.json", "*.safetensors", "*.model", "pytorch_model*.bin"]

    if args.proxy:
        proxies = {
            "https": args.proxy
        }
    else:
        proxies = None

    huggingface_hub.snapshot_download(repo_id=args.model, allow_patterns=allow_patterns, proxies=proxies, local_dir=model_path)