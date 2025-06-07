# batch_transcribe.py
import os
from pathlib import Path
from transcriber import transcribe

def transcribe_all(root_dataset_dir: str):
    """
    Walks through each language folder under root_dataset_dir,
    finds any .wav/.mp3/.flac files, and writes a .txt transcript
    in that language's transcripts/ subfolder.
    """
    for lang_folder in os.listdir(root_dataset_dir):
        lang_path = os.path.join(root_dataset_dir, lang_folder)
        # Make sure this is indeed a directory like "Hindi/", "Tamil/", etc.
        if not os.path.isdir(lang_path):
            continue

        # Locate (or ensure) transcripts/ subfolder
        transcripts_dir = os.path.join(lang_path, "transcripts")
        os.makedirs(transcripts_dir, exist_ok=True)

        # Loop over all audio files in this language folder (not recursing deeper)
        for fname in os.listdir(lang_path):
            # Skip anything that isn’t an audio file
            if not fname.lower().endswith((".wav", ".mp3", ".flac")):
                continue

            audio_path = os.path.join(lang_path, fname)
            base_name = Path(fname).stem  # e.g. "utt001"

            # Skip if transcript already exists
            out_txt = os.path.join(transcripts_dir, base_name + ".txt")
            if os.path.exists(out_txt):
                print(f"[SKIP] {lang_folder}/{fname} → transcript already exists.")
                continue

            print(f"[TRANSCRIBE] {lang_folder}/{fname} → transcripts/{base_name}.txt")
            try:
                # transcribe() returns a dict: {"text": "...", "language": "hi" (or similar)}
                result = transcribe(audio_path)
                text = result["text"].strip()
            except Exception as e:
                print(f"  ❗ Error transcribing {audio_path}: {e}")
                # Write an empty file so we don’t retry endlessly
                text = ""
            # Write the transcript
            with open(out_txt, "w", encoding="utf-8") as fout:
                fout.write(text + "\n")

    print("🎉 All done generating transcripts.")


if __name__ == "__main__":
    # Adjust this path if your dataset is elsewhere
    DATASET_DIR = os.path.join(os.path.dirname(__file__), "data", "iiit_spoken_language_datasets")
    transcribe_all(DATASET_DIR)