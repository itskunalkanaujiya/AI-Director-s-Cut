from pydantic import BaseModel
from typing import List, Optional


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class VisualEnergyPoint(BaseModel):
    time: float
    energy: float  # 0-1 normalized motion/scene-change score


class Clip(BaseModel):
    start_time: float
    end_time: float
    reason: Optional[str] = None


class DirectorDecision(BaseModel):
    clips: List[Clip]
    intro_script: str
    outro_script: str
    title: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    stage: str          # queued, analyzing_audio, analyzing_video, directing, rendering, done, error
    progress: int        # 0-100
    message: str
    transcript: Optional[List[TranscriptSegment]] = None
    director: Optional[DirectorDecision] = None
    output_video: Optional[str] = None
    source_video: Optional[str] = None
    error: Optional[str] = None
