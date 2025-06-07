# Fine-tuning BART for Hindi Audio Summarization

This document explains how to fine-tune the BART model on the IIIT Spoken Language Dataset for Hindi audio summarization.

## Overview

The fine-tuning process involves the following steps:

1. Transcribe Hindi audio files to text (if not already done)
2. Prepare a dataset with transcripts and summaries
3. Fine-tune the BART model on this dataset
4. Update the summarizer to use the fine-tuned model

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers library
- Datasets library
- Audio files in the `data/iiit_spoken_language_datasets/Hindi` directory

## Step 1: Generate Transcripts

If you haven't already generated transcripts for your audio files, run the batch transcription script:

```bash
python batch_transcribe.py
```

This will create transcript files in the `data/iiit_spoken_language_datasets/Hindi/transcripts` directory.

## Step 2: Prepare Summaries

For proper fine-tuning, you need summary data for each transcript. The current implementation uses placeholder summaries (first sentence of each transcript) if no summaries are provided.

For better results, you should create actual summaries for each transcript. These should be placed in a directory structure like:

```
data/iiit_spoken_language_datasets/Hindi/summaries/hin_0001.txt
data/iiit_spoken_language_datasets/Hindi/summaries/hin_0002.txt
...
```

Each summary file should contain a concise summary of the corresponding transcript.

## Step 3: Fine-tune the BART Model

Run the fine-tuning script:

```bash
python finetune_bart.py
```

This script will:
1. Load the transcripts and summaries (or create placeholder summaries)
2. Prepare the dataset for training
3. Fine-tune the BART model
4. Save the fine-tuned model to `models/summarizer/finetuned_bart`

The fine-tuning process may take several hours depending on your hardware. If you have a GPU, the process will be faster.

## Step 4: Update the Summarizer

After fine-tuning, update the summarizer to use the fine-tuned model:

```bash
python update_summarizer.py
```

This will create a backup of the original summarizer.py file and update it to use the fine-tuned model.

## Using the Fine-tuned Model

After completing these steps, the application will use your fine-tuned BART model for summarization. You can test it by running the main application:

```bash
python main.py
```

## Customization

You can customize the fine-tuning process by modifying the parameters in `finetune_bart.py`. Some key parameters include:

- `batch_size`: Number of samples per batch (reduce if you encounter memory issues)
- `epochs`: Number of training epochs
- `model_name`: Base model to fine-tune (default is "facebook/bart-large-cnn")

## Troubleshooting

- **Memory Issues**: If you encounter memory issues during fine-tuning, try reducing the batch size or maximum sequence length.
- **Quality Issues**: If the summaries are not of good quality, consider providing actual summaries instead of using placeholders.
- **Model Loading Issues**: If the model fails to load, ensure that the path to the fine-tuned model is correct in the updated summarizer.py file.

## Notes

- The fine-tuned model will be better at summarizing Hindi content after translation to English, as it learns from the specific patterns and content in your dataset.
- For optimal results, provide high-quality summaries for fine-tuning rather than using the placeholder approach.