from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from faster_whisper import WhisperModel
import os
import time

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_size = "large-v3"
model = WhisperModel(model_size, device="cpu")

class Transcription(BaseModel):
    segments: List[str]
    language: str
    language_probability: float
    proccess_time: str

@app.post("/speech-to-text/")
async def speech_to_text(file: UploadFile):
    try:
        start = time.process_time()
        # Save the uploaded file
        file_path = os.path.join("./", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Perform transcription
        segments, info = model.transcribe(file_path, beam_size=5)
        # Create transcription response
        transcription = Transcription(
            segments=[segment.text for segment in segments],
            language=info.language,
            language_probability=info.language_probability,
            proccess_time=f"time was {time.process_time() - start}"
        )
        return transcription
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthcheck/")
async def healthcheck():
    return {"status": "ok"}