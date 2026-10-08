# YouTube RAG Assistant

HTML/CSS/JavaScript + FastAPI version of the YouTube RAG project.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Open:

http://127.0.0.1:8000

API docs:

http://127.0.0.1:8000/docs

Create a `.env` file using `.env.example` and add your API keys.

## Production

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```
