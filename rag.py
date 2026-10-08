import re
import os
from functools import lru_cache

import requests
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from groq import Groq
from langchain_core.documents import Document


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in environment variables."
    )


# ============================================================
# EXTRACT YOUTUBE VIDEO ID
# ============================================================

def extract_video_id(url):
    if not url:
        return None

    patterns = [
        r"(?:youtube\.com/watch\?v=)([^&]+)",
        r"(?:youtu\.be/)([^?&]+)",
        r"(?:youtube\.com/shorts/)([^?&]+)",
        r"(?:youtube\.com/embed/)([^?&]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD CROSS ENCODER
# ============================================================

@lru_cache(maxsize=1)
def load_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


# ============================================================
# LOAD GROQ
# ============================================================

@lru_cache(maxsize=1)
def load_groq():

    return Groq(
        api_key=GROQ_API_KEY
    )


# ============================================================
# GET TRANSCRIPT USING FREETRANSCRIPTAPI
# ============================================================

def get_transcript(video_id):

    api_key = os.getenv("FREETRANSCRIPT_API_KEY")

    if not api_key:
        raise Exception(
            "FREETRANSCRIPT_API_KEY is missing from Railway Variables."
        )

    url = "https://api.freetranscriptapi.com/v1/transcript"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {
        "video_url": video_id,
        "lang": "en"
    }

    try:

        print("========================================")
        print("Fetching transcript from FreeTranscriptAPI")
        print("Video ID:", video_id)
        print("========================================")

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=30
        )

        print(
            "FreeTranscriptAPI status:",
            response.status_code
        )

        # ----------------------------------------------------
        # HANDLE API ERRORS
        # ----------------------------------------------------

        if not response.ok:

            try:
                error_data = response.json()

                raise Exception(
                    f"FreeTranscriptAPI error: {error_data}"
                )

            except ValueError:

                raise Exception(
                    f"FreeTranscriptAPI error "
                    f"{response.status_code}: "
                    f"{response.text}"
                )

        # ----------------------------------------------------
        # PARSE RESPONSE
        # ----------------------------------------------------

        data = response.json()

        transcript_data = data.get("transcript")

        if not transcript_data:

            raise Exception(
                "FreeTranscriptAPI returned no transcript "
                "for this video."
            )

        # ----------------------------------------------------
        # EXTRACT TEXT
        # ----------------------------------------------------

        text_parts = []

        for item in transcript_data:

            if isinstance(item, dict):

                text = item.get(
                    "text",
                    ""
                ).strip()

                if text:
                    text_parts.append(text)

        text = " ".join(text_parts)

        if not text.strip():

            raise Exception(
                "FreeTranscriptAPI returned an empty transcript."
            )

        print("Transcript fetched successfully.")
        print("Transcript length:", len(text))
        print(
            "Transcript segments:",
            len(transcript_data)
        )

        return text

    except requests.exceptions.Timeout:

        raise Exception(
            "FreeTranscriptAPI timed out after 30 seconds."
        )

    except requests.exceptions.RequestException as e:

        raise Exception(
            f"Could not connect to FreeTranscriptAPI: {e}"
        )

    except Exception:

        raise


# ============================================================
# CREATE TEXT CHUNKS
# ============================================================

def create_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""
        ]
    )

    documents = splitter.create_documents(
        [text]
    )

    chunks = [
        {
            "id": index,
            "text": document.page_content.strip()
        }
        for index, document in enumerate(documents)
        if document.page_content.strip()
    ]

    print(
        "Number of transcript chunks:",
        len(chunks)
    )

    return chunks


# ============================================================
# TOKENIZE TEXT FOR BM25
# ============================================================

def tokenize(text):

    return re.findall(
        r"\b[\w'-]+\b",
        text.lower()
    )


# ============================================================
# BUILD VIDEO INDEX
# ============================================================

def build_video_index(video_id):

    print("========================================")
    print("Building video index")
    print("Video ID:", video_id)
    print("========================================")

    # --------------------------------------------------------
    # GET TRANSCRIPT
    # --------------------------------------------------------

    transcript = get_transcript(
        video_id
    )

    # --------------------------------------------------------
    # CREATE CHUNKS
    # --------------------------------------------------------

    chunks = create_chunks(
        transcript
    )

    if not chunks:

        raise ValueError(
            "No usable transcript chunks found."
        )

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # BUILD FAISS
    # --------------------------------------------------------

    print("Building FAISS index...")

    vectorstore = FAISS.from_texts(
        chunk_texts,
        embedding=load_embeddings()
    )

    print("FAISS index built.")

    # --------------------------------------------------------
    # BUILD BM25
    # --------------------------------------------------------

    print("Building BM25 index...")

    tokenized_chunks = [
        tokenize(text)
        for text in chunk_texts
    ]

    bm25 = BM25Okapi(
        tokenized_chunks
    )

    print("BM25 index built.")

    print("========================================")
    print("Video index built successfully")
    print("========================================")

    return {
        "chunks": chunks,
        "chunk_texts": chunk_texts,
        "vectorstore": vectorstore,
        "bm25": bm25
    }


# ============================================================
# FAISS SEMANTIC SEARCH
# ============================================================

def semantic_search(
    question,
    vectorstore,
    k=25
):

    return vectorstore.similarity_search(
        question,
        k=k
    )


# ============================================================
# BM25 KEYWORD SEARCH
# ============================================================

def keyword_search(
    question,
    chunks,
    bm25,
    k=25
):

    question_tokens = tokenize(
        question
    )

    scores = bm25.get_scores(
        question_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    return [
        {
            "id": index,
            "text": chunks[index]["text"]
        }
        for index in ranked_indices[:k]
    ]


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    semantic_documents,
    keyword_chunks,
    k=60
):

    scores = {}
    documents = {}

    # --------------------------------------------------------
    # FAISS RESULTS
    # --------------------------------------------------------

    for rank, document in enumerate(
        semantic_documents,
        start=1
    ):

        text = document.page_content.strip()

        if not text:
            continue

        documents[text] = document

        scores[text] = (
            scores.get(text, 0)
            + 1 / (k + rank)
        )

    # --------------------------------------------------------
    # BM25 RESULTS
    # --------------------------------------------------------

    for rank, chunk in enumerate(
        keyword_chunks,
        start=1
    ):

        text = chunk["text"].strip()

        if not text:
            continue

        if text not in documents:

            documents[text] = Document(
                page_content=text
            )

        scores[text] = (
            scores.get(text, 0)
            + 1 / (k + rank)
        )

    # --------------------------------------------------------
    # SORT BY FUSION SCORE
    # --------------------------------------------------------

    return sorted(
        documents.values(),
        key=lambda document:
            scores[
                document.page_content.strip()
            ],
        reverse=True
    )


# ============================================================
# ADD NEIGHBORING CHUNKS
# ============================================================

def add_neighbor_chunks(
    ranked_documents,
    all_chunks,
    neighbor_distance=1
):

    text_to_index = {
        chunk["text"]: chunk["id"]
        for chunk in all_chunks
    }

    selected_indices = set()

    for document in ranked_documents:

        text = document.page_content.strip()

        if text not in text_to_index:
            continue

        current_index = text_to_index[
            text
        ]

        start = max(
            0,
            current_index - neighbor_distance
        )

        end = min(
            len(all_chunks),
            current_index
            + neighbor_distance
            + 1
        )

        for index in range(
            start,
            end
        ):

            selected_indices.add(
                index
            )

    return [
        Document(
            page_content=all_chunks[index]["text"]
        )
        for index in sorted(
            selected_indices
        )
    ]


# ============================================================
# CROSS ENCODER RERANKING
# ============================================================

def rerank_documents(
    question,
    documents,
    top_n=10
):

    if not documents:
        return []

    reranker = load_reranker()

    pairs = [
        (
            question,
            document.page_content
        )
        for document in documents
    ]

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(
            documents,
            scores
        ),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked[:top_n]


# ============================================================
# COMPLETE RETRIEVAL PIPELINE
# ============================================================

def retrieve_relevant_chunks(
    question,
    video_index
):

    chunks = video_index["chunks"]

    vectorstore = video_index[
        "vectorstore"
    ]

    bm25 = video_index[
        "bm25"
    ]

    # --------------------------------------------------------
    # FAISS SEARCH
    # --------------------------------------------------------

    semantic_documents = semantic_search(
        question,
        vectorstore,
        k=25
    )

    # --------------------------------------------------------
    # BM25 SEARCH
    # --------------------------------------------------------

    keyword_chunks = keyword_search(
        question,
        chunks,
        bm25,
        k=25
    )

    # --------------------------------------------------------
    # HYBRID RETRIEVAL
    # --------------------------------------------------------

    fused_documents = reciprocal_rank_fusion(
        semantic_documents,
        keyword_chunks
    )

    # --------------------------------------------------------
    # ADD NEIGHBORING CHUNKS
    # --------------------------------------------------------

    expanded_documents = add_neighbor_chunks(
        fused_documents[:15],
        chunks,
        neighbor_distance=1
    )

    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------

    combined_documents = (
        fused_documents[:25]
        + expanded_documents
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    unique_documents = []

    seen_texts = set()

    for document in combined_documents:

        text = document.page_content.strip()

        if (
            text
            and text not in seen_texts
        ):

            seen_texts.add(text)

            unique_documents.append(
                document
            )

    # --------------------------------------------------------
    # CROSS ENCODER RERANKING
    # --------------------------------------------------------

    ranked_documents = rerank_documents(
        question,
        unique_documents,
        top_n=10
    )

    return ranked_documents


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    question,
    ranked_documents
):

    if not ranked_documents:

        return (
            "I couldn't find a clear answer "
            "in the video."
        )

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for index, (
        document,
        score
    ) in enumerate(
        ranked_documents,
        start=1
    ):

        text = document.page_content.strip()

        if text:

            context_parts.append(
                f"[Evidence {index}]\n{text}"
            )

    context = "\n\n---\n\n".join(
        context_parts
    )

    # Limit prompt size
    context = context[:18000]

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a precise question-answering assistant for YouTube videos.

Answer the user's question using ONLY the supplied evidence.

USER QUESTION:

{question}

SUPPLIED EVIDENCE:

{context}

INSTRUCTIONS:

1. Answer the exact question directly.
2. Do not use outside knowledge.
3. Do not guess or invent missing information.
4. For questions containing words such as after, before, next,
   previous, first, second, later, earlier, following, or preceding,
   carefully reconstruct the sequence from the evidence.
5. If multiple evidence passages describe the same event, combine them.
6. If the evidence contains a transcription error or inconsistent spelling,
   use surrounding evidence to understand the intended entity, but do not
   invent unsupported facts.
7. Give the direct answer first.
8. Normally answer in 1-3 concise sentences.
9. Do not mention transcript, evidence, chunks, FAISS, BM25, embeddings,
   reranking, RAG, context, or retrieval.
10. Do not repeat the user's question.
11. If the supplied information genuinely does not support an answer, say:
    I couldn't find a clear answer in the video.
"""

    # --------------------------------------------------------
    # CALL GROQ
    # --------------------------------------------------------

    try:

        response = load_groq().chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0,
            top_p=1,
            seed=42,

            reasoning_effort="low",
            include_reasoning=False,

            max_completion_tokens=500
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        if not answer:

            return (
                "I couldn't find a clear answer "
                "in the video."
            )

        answer = answer.strip()

        # Remove ANSWER / FINAL ANSWER labels
        answer = re.sub(
            r"^\s*\*?(FINAL ANSWER|ANSWER)\s*\*?:\s*",
            "",
            answer,
            flags=re.IGNORECASE
        )

        return answer.strip()

    except Exception as e:

        raise Exception(
            f"Groq error: {str(e)}"
        ) from e
