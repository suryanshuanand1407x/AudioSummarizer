# summarizer.py

import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Summarizer:
    def __init__(self, model_path: str = None):
        """
        Loads a fine-tuned summarization model from local path.
        Assumes model.safetensors and tokenizer files are inside `model_path`.
        """
        # Detect available device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Apple Silicon GPU
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")  # NVIDIA GPU
        else:
            self.device = torch.device("cpu")

        # Set default model path if not provided
        if model_path is None:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "models", "summarizer", "finetuned_bart", "checkpoint-112")

        # Load tokenizer and model from local directory
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

    def summarize(self, text: str, max_length: int = 300, min_length: int = 50) -> str:
        """
        Generate a summary from the input text.
        """
        if not text.strip():
            return ""

        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        ).to(self.device)

        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=4,
                length_penalty=2.0,
                max_length=max_length,
                min_length=min_length,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        # Decode and post-process the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Remove unwanted characters (e.g., accidental Gujarati or non-Devanagari)
        summary = re.sub(r'[^\u0900-\u097F\s\u200d\u200c.,!?]', '', summary)

        return summary


# Singleton instance for convenience
_summarizer = Summarizer()

def summarize(text: str) -> str:
    return _summarizer.summarize(text)


if __name__ == "__main__":
    # Demo usage
    example = (
        "भारत नवीकरणीय ऊर्जा अपनाने में महत्वपूर्ण प्रगति कर रहा है। "
        "सरकार ने सौर, पवन और जल विद्युत को बढ़ावा देने के लिए कई पहल शुरू की हैं।"
    )
    print("Original:\n", example)
    print("\nSummary:\n", summarize(example))
