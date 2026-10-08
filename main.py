from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import extract_video_id, build_video_index, retrieve_relevant_chunks, generate_answer

app = FastAPI(title="YouTube RAG Assistant")

app.mount("/static", StaticFiles(directory="static"), name="static")

current_video_id = None
current_video_index = None


class VideoRequest(BaseModel):
    video_url: str


class QuestionRequest(BaseModel):
    url: str
    question: str


@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as file:
        return file.read()


@app.post("/load")
def load_video(request: VideoRequest):
    global current_video_id, current_video_index

    video_id = extract_video_id(request.video_url)

    if not video_id:
        return {"error": "Invalid YouTube URL."}

    try:
        if current_video_id != video_id:
            current_video_index = build_video_index(video_id)
            current_video_id = video_id

        return {"message": "Video loaded successfully."}

    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    global current_video_id, current_video_index

    try:
        # Get URL and question from frontend
        url = request.url
        question = request.question

        # Extract YouTube video ID
        video_id = extract_video_id(url)

        if not video_id:
            return {"error": "Invalid YouTube URL"}

        # Build the index only if this is a new video
        if video_id != current_video_id:
            current_video_index = build_video_index(video_id)
            current_video_id = video_id

        # Retrieve relevant chunks
        ranked_documents = retrieve_relevant_chunks(
            question,
            current_video_index
        )

        # Generate answer using Groq
        answer = generate_answer(
            question,
            ranked_documents
        )

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}