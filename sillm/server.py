import sys
import argparse
import logging

import fastapi
import uvicorn

import sillm

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
        "data": [model.description() for model in app.models]
    }

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model", type=str, help="The directory that contains the models")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Port to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()

    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(level=log_level, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    
    # Load model
    model = sillm.load(args.model)
    app.models = [model]

    logging.warning("API server is not fully implemented yet.")

    uvicorn.run(app, host=args.host, port=args.port)