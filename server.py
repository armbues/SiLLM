import argparse
import pathlib

import fastapi
import uvicorn

app = fastapi.FastAPI()

@app.get("/")
def index():
    return {
        "message": "Welcome to the OpenAI API! Documentation is available at https://platform.openai.com/docs/api-reference"
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [model.to_dict() for model in app.models]
    }

class ModelDescription():
    def __init__(self, model_path):
        self.path = pathlib.Path(model_path)
        self.id = self.path.stem
        self.created = int(self.path.stat().st_ctime)

    def to_dict(self):
        return {
            "id": self.id,
            "object": "model",
            "created": self.created
        }

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_dir", type=str, help="The directory that contains the models")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Port to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    
    model_dir = pathlib.Path(args.model_dir)
    assert model_dir.is_dir()

    app.models = []
    # Load GGUF model metadata
    for model_gguf in model_dir.glob("*.gguf"):
        model = ModelDescription(model_gguf)
        app.models.append(model)

    uvicorn.run(app, host=args.host, port=args.port)