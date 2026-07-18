"""
The Local Render Engine (Heavy Lifting Layer)
- Slices the source video per the Director's JSON and concatenates the clips.
- Generates intro/outro TTS narration and overlays it onto the reel with
  simple audio ducking (original audio volume reduced under the voiceover).
"""
import os
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeAudioClip,
    afx,
)
from models import DirectorDecision
from tts import synthesize_speech

DUCK_VOLUME = 0.25  # how much to lower original audio under narration
NARRATION_PAD = 0.3  # seconds of silence padding around narration


def render_highlight_reel(
    source_video_path: str,
    decision: DirectorDecision,
    work_dir: str,
    output_path: str,
) -> str:
    os.makedirs(work_dir, exist_ok=True)

    source = VideoFileClip(source_video_path)

    # 1. Slice + concatenate the chosen clips
    subclips = []
    for clip in decision.clips:
        start = max(0, clip.start_time)
        end = min(source.duration, clip.end_time)
        if end - start < 0.2:
            continue
        subclips.append(source.subclip(start, end))

    if not subclips:
        raise RuntimeError("No valid subclips to render.")

    reel = concatenate_videoclips(subclips, method="compose")

    # 2. Generate narration audio
    intro_wav = os.path.join(work_dir, "intro.wav")
    outro_wav = os.path.join(work_dir, "outro.wav")
    synthesize_speech(decision.intro_script, intro_wav)
    synthesize_speech(decision.outro_script, outro_wav)

    intro_audio = AudioFileClip(intro_wav) if os.path.exists(intro_wav) else None
    outro_audio = AudioFileClip(outro_wav) if os.path.exists(outro_wav) else None

    # 3. Duck original reel audio, then layer narration at start and end
    original_audio = reel.audio
    ducked_audio = original_audio.fx(afx.volumex, DUCK_VOLUME) if original_audio else None

    audio_tracks = []
    if ducked_audio:
        audio_tracks.append(ducked_audio)

    if intro_audio:
        audio_tracks.append(intro_audio.set_start(NARRATION_PAD))
    if outro_audio:
        outro_start = max(reel.duration - outro_audio.duration - NARRATION_PAD, 0)
        audio_tracks.append(outro_audio.set_start(outro_start))

    if audio_tracks:
        final_audio = CompositeAudioClip(audio_tracks).set_duration(reel.duration)
        reel = reel.set_audio(final_audio)

    # 4. Write final output
    reel.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=source.fps or 24,
        logger=None,
        threads=4,
    )

    # cleanup
    for c in subclips:
        c.close()
    source.close()
    reel.close()
    if intro_audio:
        intro_audio.close()
    if outro_audio:
        outro_audio.close()

    return output_path
