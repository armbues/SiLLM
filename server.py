import argparse

import fastapi
import uvicorn

app = fastapi.FastAPI()

@app.get("/")
def index():
    return {}

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Port to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)