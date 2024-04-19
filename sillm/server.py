import argparse
import time
import uuid

import fastapi
import uvicorn
import pydantic

import sillm
import sillm.utils as utils

app = fastapi.FastAPI()

@app.get("/")
def index():
    return {
        "message": "Welcome to the SiLLM API!"
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [app.model.description()]
    }

class ChatCompletion(pydantic.BaseModel):
    messages: list
    model: str
    logprobs: bool | None = False
    max_tokens: int | None = 4096
    temperature: float | None = 0.7

@app.post("/v1/chat/completions")
def chat_completions(completion: ChatCompletion):
    result = {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": app.model.id,
        "system_fingerprint": app.fingerprint
    }

    prompt = app.template.apply_chat_template(completion.messages, add_generation_prompt=True)
    
    generate_args = {
        "max_tokens": completion.max_tokens,
        "temperature": completion.temperature
    }

    content = ""
    for s, metadata in app.model.generate(prompt, **generate_args):
        content += s

    result["choices"] = [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": content
        },
        "logprobs": None,
        "finish_reason": metadata["finish_reason"]
    }]
    result["usage"] = metadata["usage"]

    return result

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model", type=str, help="The directory that contains the models")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Port to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--template", type=str, default=None, help="Chat template (chatml, llama2, alpaca, etc.)")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()

    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)
    app.logger = logger
    
    # Load model
    model = sillm.load(args.model)
    app.model = model

    # Initialize system fingerprint
    app.fingerprint = "sillm_" + str(uuid.uuid4())

    # Initialize messages template
    if args.template:
        template = sillm.Template(model.tokenizer, template_name=args.template)
    else:
        template_name = sillm.Template.guess_template(model.args)
        if template_name:
            template = sillm.Template(model.tokenizer, template_name=template_name)
        elif model.tokenizer.has_template:
            template = sillm.AutoTemplate(model.tokenizer)
        else:
            template = sillm.Template(model.tokenizer, template_name="empty")
            logger.warn("No conversation template found - falling back to empty template.")
    app.template = template

    uvicorn.run(app, host=args.host, port=args.port)