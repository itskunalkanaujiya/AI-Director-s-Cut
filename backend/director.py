"""
The Creative Director (GenAI Layer)
Sends the unified timeline (transcript + visual energy) to an LLM hosted on the
Hugging Face Inference API (via huggingface_hub.InferenceClient) and gets back a
strict JSON object with the top clips + intro/outro narration scripts.
"""
import json
import os
import re
from huggingface_hub import InferenceClient
from models import DirectorDecision, Clip

_client = None

SYSTEM_PROMPT = """You are a world-class Hollywood video editor and "Creative Director" \
for an automated highlight-reel tool. You are given a timestamped transcript and a \
per-second visual energy score (0-1, higher = more motion/scene change) for a raw video.

Your job:
1. Pick the 3-5 most engaging, self-contained clips to stitch into a highlight reel.
   - Prefer moments with high visual energy AND/OR emotionally/narratively interesting speech.
   - Clips should be between 3 and 20 seconds each.
   - Clips must not overlap, and start_time < end_time, both within the video duration.
   - Order clips chronologically.
2. Write a short, punchy INTRO narration script (1-2 sentences, hypes up what's coming).
3. Write a short OUTRO narration script (1-2 sentences, wraps things up memorably).
4. Suggest a short catchy title for the highlight reel.

Respond with ONLY valid JSON, no markdown fences, no commentary, matching exactly this schema:
{
  "clips": [{"start_time": float, "end_time": float, "reason": string}, ...],
  "intro_script": string,
  "outro_script": string,
  "title": string
}
"""

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_PROVIDER = "auto"  # let HF route to an available provider (Cerebras/Groq/Together/etc.)


def get_client() -> InferenceClient:
    global _client
    if _client is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        if not token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. "
                "Get a token from https://huggingface.co/settings/tokens and export it "
                "(needs 'Make calls to Inference Providers' permission)."
            )
        provider = os.environ.get("HF_PROVIDER", DEFAULT_PROVIDER)
        _client = InferenceClient(provider=provider, api_key=token)
    return _client


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response, since open
    models don't always strictly obey 'JSON only' instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the largest {...} block in the response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from model response:\n{text[:500]}")


def direct_highlight_reel(timeline: dict, model: str | None = None) -> DirectorDecision:
    client = get_client()
    model = model or os.environ.get("HF_MODEL", DEFAULT_MODEL)

    user_prompt = (
        "Here is the unified timeline data (transcript + visual energy) for the raw video:\n\n"
        f"{json.dumps(timeline, indent=2)}\n\n"
        "Return the JSON decision now."
    )

    response = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1500,
        temperature=0.7,
    )

    text = response.choices[0].message.content.strip()
    data = _extract_json(text)

    duration = timeline.get("duration_seconds", float("inf"))
    clips = []
    for c in data.get("clips", []):
        start, end = float(c["start_time"]), float(c["end_time"])
        if start >= end:
            continue
        start = max(0.0, start)
        end = min(duration, end)
        clips.append(Clip(start_time=start, end_time=end, reason=c.get("reason")))

    if not clips:
        raise RuntimeError("Director returned no valid clips.")

    return DirectorDecision(
        clips=clips,
        intro_script=data.get("intro_script", ""),
        outro_script=data.get("outro_script", ""),
        title=data.get("title"),
    )
