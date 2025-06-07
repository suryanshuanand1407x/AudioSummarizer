import tkinter as tk
from tkinter import filedialog, scrolledtext, StringVar
from pathlib import Path

from transcriber import transcribe
from summarizer import summarize
from translator import translate_to_english


class AudioSummarizerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Audio → Text Summarizer")

        # --- Row 0: Load Audio Button & Selected Path ---
        self.load_btn = tk.Button(root, text="Load Audio", command=self.load_audio)
        self.load_btn.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.selected_path = StringVar(value="No file selected")
        self.path_label = tk.Label(root, textvariable=self.selected_path, fg="gray")
        self.path_label.grid(row=0, column=1, columnspan=3, padx=10, pady=10, sticky="w")

        # --- Row 1: Buttons for Summaries ---
        self.gen_summ_btn = tk.Button(root, text="Generate Summary", command=self.generate_summary)
        self.gen_summ_btn.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.gen_summ_en_btn = tk.Button(root, text="Generate Summary in English", command=self.generate_summary_english)
        self.gen_summ_en_btn.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # --- Row 2: Original‐Language Summary Output ---
        self.orig_label = tk.Label(root, text="Original‐Language Summary:")
        self.orig_label.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="nw")

        self.orig_output_box = scrolledtext.ScrolledText(root, width=80, height=10, wrap=tk.WORD)
        self.orig_output_box.grid(row=3, column=0, columnspan=4, padx=10, pady=5)
        self.orig_output_box.configure(state="disabled")

        # --- Row 4: English Summary Output ---
        self.en_label = tk.Label(root, text="English Summary:")
        self.en_label.grid(row=4, column=0, padx=10, pady=(10, 0), sticky="nw")

        self.en_output_box = scrolledtext.ScrolledText(root, width=80, height=10, wrap=tk.WORD)
        self.en_output_box.grid(row=5, column=0, columnspan=4, padx=10, pady=5)
        self.en_output_box.configure(state="disabled")

        # Internal state
        self.audio_file_path = None
        self.last_transcript = ""  # store the latest transcript string

    def load_audio(self):
        """Open a file dialog to select an audio file."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")]
        )
        if file_path:
            self.audio_file_path = file_path
            display_name = Path(file_path).name
            self.selected_path.set(display_name)

            # Clear previous outputs
            self._clear_output(self.orig_output_box)
            self._clear_output(self.en_output_box)
            self.last_transcript = ""

    def generate_summary(self):
        """
        Generate a summary in the audio's original language (Hindi),
        using the fine-tuned BART model.
        """
        if not self.audio_file_path:
            self._append_output(self.orig_output_box, "❗ No audio file selected. Please click 'Load Audio' first.\n")
            return

        # Transcribe if not done yet
        if not self.last_transcript:
            try:
                self._append_output(self.orig_output_box, f"▶️  Transcribing: {Path(self.audio_file_path).name}\n\n")
                result = transcribe(self.audio_file_path)
                self.last_transcript = result["text"]
            except Exception as e:
                self._append_output(self.orig_output_box, f"❗ Error during transcription: {e}\n")
                return

        # Summarize in Hindi using fine-tuned model
        try:
            self._append_output(self.orig_output_box, "▶️  Generating Hindi summary using checkpoint-112...\n\n")
            summary_hi = summarize(self.last_transcript)
        except Exception as e:
            self._append_output(self.orig_output_box, f"❗ Error during summarization: {e}\n")
            return

        self._append_output(self.orig_output_box, "----- Hindi Summary -----\n")
        self._append_output(self.orig_output_box, summary_hi + "\n")
        self._append_output(self.orig_output_box, "-------------------------\n\n")

    def generate_summary_english(self):
        """
        Generate an English summary by:
          1) Transcribing the audio to text
          2) Translating the text from Hindi to English
          3) Summarizing the English text
        """
        if not self.audio_file_path:
            self._append_output(self.en_output_box, "❗ No audio file selected. Please click 'Load Audio' first.\n")
            return

        # Transcribe if not done yet
        if not self.last_transcript:
            try:
                self._append_output(self.en_output_box, f"▶️  Transcribing: {Path(self.audio_file_path).name}\n\n")
                result = transcribe(self.audio_file_path)
                self.last_transcript = result["text"]
            except Exception as e:
                self._append_output(self.en_output_box, f"❗ Error during transcription: {e}\n")
                return

        # Translate to English
        try:
            self._append_output(self.en_output_box, "▶️  Translating to English...\n\n")
            english_text = translate_to_english(self.last_transcript)
        except Exception as e:
            self._append_output(self.en_output_box, f"❗ Error during translation: {e}\n")
            return

        # Summarize the English text
        try:
            self._append_output(self.en_output_box, "▶️  Generating summary using checkpoint-112...\n\n")
            summary_en = summarize(english_text)
        except Exception as e:
            self._append_output(self.en_output_box, f"❗ Error during summarization: {e}\n")
            return

        self._append_output(self.en_output_box, "----- English Summary -----\n")
        self._append_output(self.en_output_box, summary_en + "\n")
        self._append_output(self.en_output_box, "---------------------------\n\n")

    def _append_output(self, text_widget: tk.Text, text: str):
        """Append `text` to the given Text widget (read-only)."""
        text_widget.configure(state="normal")
        text_widget.insert(tk.END, text)
        text_widget.see(tk.END)
        text_widget.configure(state="disabled")

    def _clear_output(self, text_widget: tk.Text):
        """Clear all content from the given Text widget."""
        text_widget.configure(state="normal")
        text_widget.delete("1.0", tk.END)
        text_widget.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioSummarizerGUI(root)
    root.mainloop()
