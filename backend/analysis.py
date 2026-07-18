"""
Media Ingestion & Analysis Engine (ML Layer)
- Audio: faster-whisper transcription with timestamps
- Visual: OpenCV frame-difference based "visual energy" curve
"""
import cv2
import numpy as np
from faster_whisper import WhisperModel
from models import TranscriptSegment, VisualEnergyPoint

_whisper_model = None


def get_whisper_model(model_size: str = "base"):
    """Lazy-load whisper model once and reuse across requests."""
    global _whisper_model
    if _whisper_model is None:
        # CPU-friendly default; swap "base" -> "small"/"medium" if you have GPU time to spare
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_audio(video_path: str, model_size: str = "base") -> list[TranscriptSegment]:
    """Run speech-to-text on the video's audio track and return timestamped segments."""
    model = get_whisper_model(model_size)
    segments, _info = model.transcribe(video_path, vad_filter=True)
    results = []
    for seg in segments:
        results.append(TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip()))
    return results


def analyze_visual_energy(video_path: str, sample_rate_fps: float = 2.0) -> list[VisualEnergyPoint]:
    """
    Sample frames at sample_rate_fps and compute a motion/scene-change "energy" score
    per sample using frame differencing. Returns a normalized (0-1) energy curve.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(native_fps / sample_rate_fps), 1)

    prev_gray = None
    raw_scores = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                score = float(np.mean(diff))
            else:
                score = 0.0

            timestamp = frame_idx / native_fps
            raw_scores.append((timestamp, score))
            prev_gray = gray
        frame_idx += 1

    cap.release()

    if not raw_scores:
        return []

    max_score = max(s for _, s in raw_scores) or 1.0
    points = [
        VisualEnergyPoint(time=t, energy=round(s / max_score, 4))
        for t, s in raw_scores
    ]
    return points


def build_unified_timeline(
    transcript: list[TranscriptSegment],
    visual_energy: list[VisualEnergyPoint],
) -> dict:
    """
    Combine transcript + visual energy into a single structure the LLM
    can reason over efficiently (compact, not frame-by-frame verbose).
    """
    # Downsample visual energy into ~1 point per second to keep the LLM prompt small
    bucketed = {}
    for p in visual_energy:
        bucket = int(p.time)
        bucketed.setdefault(bucket, []).append(p.energy)
    energy_per_second = [
        {"t": t, "energy": round(sum(v) / len(v), 3)} for t, v in sorted(bucketed.items())
    ]

    return {
        "transcript": [seg.model_dump() for seg in transcript],
        "visual_energy_per_second": energy_per_second,
        "duration_seconds": max(
            [seg.end for seg in transcript] + [p["t"] for p in energy_per_second] + [0]
        ),
    }
