import argparse
import pathlib
import json

def load_chat_template(fpath: str) -> str:
    with open(fpath) as f:
        config = json.loads(f.read())
        
        if "chat_template" in config:
            return config["chat_template"].replace("\n", "")
    
    return None

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Extractor for Jinja2 templates from tokenizer config files.")
    parser.add_argument("config", type=str, help="Path to tokenizer config file")
    parser.add_argument("output", type=str, help="Path to output file")
    args = parser.parse_args()

    template = load_chat_template(args.config)

    if template is not None:
        with open(args.output, "w") as f:
            f.write(template)