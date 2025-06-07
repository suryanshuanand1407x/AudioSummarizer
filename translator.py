# translator.py

import torch
from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-hi-en"):
        """
        Loads a MarianMT model for Hindi→English translation, forcing safetensors.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer normally
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)

        # Load the model using safetensors if available
        # 'use_safetensors=True' tells HF to prefer .safetensors over .bin if the checkpoint has it.
        self.model = MarianMTModel.from_pretrained(
            model_name,
            use_safetensors=True,
            trust_remote_code=False
        ).to(self.device)

    def translate_hi_to_en(self, text_hi: str) -> str:
        """
        Translate Hindi text (UTF-8) to English.
        """
        if not text_hi.strip():
            return ""

        # Prepare batch (size 1)
        batch = self.tokenizer.prepare_seq2seq_batch([text_hi], return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **batch,
            num_beams=4,
            max_length=512,
            early_stopping=True
        )
        translated = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return translated


# Singleton instance
_translator = Translator()

def translate_to_english(text_hi: str) -> str:
    return _translator.translate_hi_to_en(text_hi)


if __name__ == "__main__":
    sample_hi = "यह एक परीक्षण वाक्य है जो हिंदी में लिखा गया था।"
    print("Hindi:", sample_hi)
    print("→ English:", translate_to_english(sample_hi))