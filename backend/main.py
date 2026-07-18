import os
import shutil
import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from models import JobStatus
from analysis import transcribe_audio, analyze_visual_energy, build_unified_timeline
from director import direct_highlight_reel
from render import render_highlight_reel

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="AI Director's Cut")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

JOBS: dict[str, JobStatus] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _set(job_id: str, **kwargs):
    job = JOBS[job_id]
    for k, v in kwargs.items():
        setattr(job, k, v)


def process_job(job_id: str, video_path: str):
    try:
        _set(job_id, stage="analyzing_audio", progress=10, message="Transcribing speech with Whisper...")
        transcript = transcribe_audio(video_path)
        _set(job_id, transcript=transcript)

        _set(job_id, stage="analyzing_video", progress=35, message="Scanning frames for visual energy...")
        visual_energy = analyze_visual_energy(video_path)

        timeline = build_unified_timeline(transcript, visual_energy)

        _set(job_id, stage="directing", progress=55, message="Consulting the AI Director for cut choices...")
        decision = direct_highlight_reel(timeline)
        _set(job_id, director=decision)

        _set(job_id, stage="rendering", progress=75, message="Cutting, scoring, and narrating the reel...")
        work_dir = os.path.join(OUTPUT_DIR, job_id, "work")
        output_path = os.path.join(OUTPUT_DIR, job_id, "highlight_reel.mp4")
        render_highlight_reel(video_path, decision, work_dir, output_path)

        rel_output = f"/outputs/{job_id}/highlight_reel.mp4"
        _set(job_id, stage="done", progress=100, message="Highlight reel ready!", output_video=rel_output)

    except Exception as e:
        traceback.print_exc()
        _set(job_id, stage="error", progress=100, message="Something went wrong.", error=str(e))


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".m4v")):
        raise HTTPException(400, "Please upload an MP4 or MOV file.")

    job_id = str(uuid.uuid4())[:8]
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)
    video_path = os.path.join(job_upload_dir, file.filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    os.makedirs(os.path.join(OUTPUT_DIR, job_id), exist_ok=True)

    JOBS[job_id] = JobStatus(
        job_id=job_id, stage="queued", progress=0, message="Queued for processing...",
        source_video=f"/uploads/{job_id}/{file.filename}",
    )

    EXECUTOR.submit(process_job, job_id, video_path)

    return {"job_id": job_id, "source_video": f"/uploads/{job_id}/{file.filename}"}


@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@app.get("/api/health")
async def health():
    return {"status": "ok"}
