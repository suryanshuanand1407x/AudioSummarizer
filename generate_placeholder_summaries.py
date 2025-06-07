# generate_placeholder_summaries.py

import os
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# The error shows we need punkt_tab as well
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab')

# If punkt_tab is not available, let's modify our code to use punkt instead

def generate_placeholder_summaries(transcript_dir, summary_dir):
    """
    Generate placeholder summaries for each transcript.
    This is a temporary solution until proper summaries are created.
    
    The placeholder summary is created by taking the first 1-2 sentences of the transcript,
    depending on the length of the first sentence.
    """
    # Create summary directory if it doesn't exist
    os.makedirs(summary_dir, exist_ok=True)
    
    # Count variables for reporting
    total_files = 0
    processed_files = 0
    skipped_files = 0
    
    # Process each transcript file
    for file in os.listdir(transcript_dir):
        if file.endswith(".txt"):
            total_files += 1
            transcript_path = os.path.join(transcript_dir, file)
            summary_path = os.path.join(summary_dir, file)
            
            # Skip if summary already exists
            if os.path.exists(summary_path):
                print(f"[SKIP] {file} → summary already exists.")
                skipped_files += 1
                continue
            
            # Read transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:  # Skip empty transcripts
                print(f"[SKIP] {file} → empty transcript.")
                skipped_files += 1
                continue
            
            # Generate placeholder summary
            try:
                # Try using the standard sent_tokenize function
                sentences = sent_tokenize(text)
                
                if not sentences:  # If sentence tokenization fails
                    summary = text[:min(100, len(text))]
                elif len(sentences) == 1 or len(sentences[0]) < 50:  # Short first sentence
                    summary = ' '.join(sentences[:min(2, len(sentences))])
                else:  # First sentence is long enough
                    summary = sentences[0]
            except Exception as e:
                print(f"Warning: Sentence tokenization failed: {e}")
                # Fallback: simple splitting by punctuation
                simple_sentences = [s.strip() + '.' for s in text.replace('।', '.').replace('?', '?.').replace('!', '!.').split('.') if s.strip()]
                
                if not simple_sentences:  # If simple tokenization fails
                    summary = text[:min(100, len(text))]
                elif len(simple_sentences) == 1 or len(simple_sentences[0]) < 50:  # Short first sentence
                    summary = ' '.join(simple_sentences[:min(2, len(simple_sentences))])
                else:  # First sentence is long enough
                    summary = simple_sentences[0]
            
            # Ensure summary isn't too long or too short
            if len(summary) > 150:
                summary = summary[:150] + "..."
            elif len(summary) < 10 and len(text) > 10:
                # If summary is too short but text is longer, take more text
                summary = text[:min(100, len(text))]
            
            # Write summary
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            print(f"[GENERATED] {file} → placeholder summary created.")
            processed_files += 1
    
    print(f"\nSummary Generation Complete:")
    print(f"  Total transcript files: {total_files}")
    print(f"  Summaries generated: {processed_files}")
    print(f"  Files skipped: {skipped_files}")
    print("\nNote: These are placeholder summaries for fine-tuning purposes.")
    print("For better results, consider creating actual summaries manually.")

if __name__ == "__main__":
    # Set paths
    base_dir = os.path.dirname(__file__)
    transcript_dir = os.path.join(base_dir, "data", "iiit_spoken_language_datasets", "Hindi", "transcripts")
    summary_dir = os.path.join(base_dir, "data", "iiit_spoken_language_datasets", "Hindi", "summaries")
    
    # Check if transcript directory exists
    if not os.path.exists(transcript_dir):
        print(f"Error: Transcript directory not found: {transcript_dir}")
        print("Please run batch_transcribe.py first to generate transcripts.")
    else:
        generate_placeholder_summaries(transcript_dir, summary_dir)