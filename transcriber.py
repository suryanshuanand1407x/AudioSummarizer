# transcriber.py

import os
import whisper

def load_whisper_model(model_size: str = "small"):
    """
    Load a Whisper model (downloads if not already cached).
    """
    return whisper.load_model(model_size)

# Load once at import time
_whisper_model = load_whisper_model("small")


def transcribe(audio_path: str, whisper_model=_whisper_model) -> dict:
    """
    Transcribe the given audio file (.wav, .mp3, .flac) using Whisper.
    Returns a dict: {"text": "<UTF-8 transcript>", "language": "<detected code>"}
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    result = whisper_model.transcribe(audio_path, fp16=False)
    return {
        "text": result["text"].strip(),
        "language": result.get("language", None)
    }


if __name__ == "__main__":
    # Quick CLI test: python transcriber.py path/to/audio.wav
    import sys
    if len(sys.argv) != 2:
        print("Usage: python transcriber.py <path/to/audio.wav>")
        sys.exit(1)

    path = sys.argv[1]
    out = transcribe(path)
    print("Detected language:", out["language"])
    print("Transcribed text:", out["text"])