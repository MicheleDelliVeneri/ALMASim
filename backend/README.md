# ALMASim FastAPI Backend

FastAPI backend for ALMASim simulation service.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables (optional, defaults are provided):
```bash
export MAIN_DIR=./almasim
export OUTPUT_DIR=./outputs
```

3. Run the server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc


