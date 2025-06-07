# finetune_bart.py

import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset as HFDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizationDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_input_length=1024, max_target_length=128):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]

        # Tokenize inputs
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                summary,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        # Create the final dictionary
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = targets["input_ids"].squeeze()
        
        # Replace padding token id with -100 so it's ignored in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def prepare_dataset(transcript_dir, summary_dir=None):
    """
    Prepare dataset from transcripts and summaries.
    If summary_dir is None, we'll need to create summaries or use a placeholder approach.
    """
    texts = []
    file_ids = []
    
    # Load all transcripts
    for file in os.listdir(transcript_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(transcript_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:  # Only add non-empty transcripts
                    texts.append(text)
                    file_ids.append(Path(file).stem)
    
    # If we don't have summaries, we need to create them or use placeholders
    # For this example, we'll create placeholder summaries (first sentence of each transcript)
    # In a real scenario, you would need actual summaries for proper fine-tuning
    if summary_dir is None or not os.path.exists(summary_dir):
        logger.warning("No summaries provided. Using first sentence as placeholder summary.")
        logger.warning("For proper fine-tuning, you need actual summaries!")
        
        # Create placeholder summaries (first sentence or first 50 characters)
        summaries = []
        for text in texts:
            sentences = text.split('.')
            if len(sentences) > 0 and len(sentences[0]) > 10:
                summaries.append(sentences[0] + '.')
            else:
                # If no clear sentence, take first 50 chars
                summaries.append(text[:min(50, len(text))])
    else:
        # Load actual summaries if available
        summaries = []
        for file_id in file_ids:
            summary_path = os.path.join(summary_dir, f"{file_id}.txt")
            if os.path.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summaries.append(f.read().strip())
            else:
                # If summary doesn't exist for this transcript, use placeholder
                text = texts[file_ids.index(file_id)]
                sentences = text.split('.')
                if len(sentences) > 0:
                    summaries.append(sentences[0] + '.')
                else:
                    summaries.append(text[:min(50, len(text))])
    
    # Create train/validation split
    train_texts, val_texts, train_summaries, val_summaries = train_test_split(
        texts, summaries, test_size=0.1, random_state=42
    )
    
    return {
        "train": {"text": train_texts, "summary": train_summaries},
        "validation": {"text": val_texts, "summary": val_summaries}
    }

def finetune_bart(dataset_dir, output_dir, model_name="facebook/bart-large-cnn", batch_size=4, epochs=3):
    """
    Fine-tune BART model on the dataset.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    # Prepare dataset
    transcript_dir = os.path.join(dataset_dir, "Hindi", "transcripts")
    # Use the existing summaries directory
    summary_dir = os.path.join(dataset_dir, "Hindi", "summaries")
    
    # Check if transcript directory exists
    if not os.path.exists(transcript_dir):
        logger.error(f"Transcript directory not found: {transcript_dir}")
        return
    
    # Prepare dataset
    dataset_dict = prepare_dataset(transcript_dir, summary_dir)
    
    # Convert to HuggingFace datasets
    train_dataset = HFDataset.from_dict({
        "text": dataset_dict["train"]["text"],
        "summary": dataset_dict["train"]["summary"]
    })
    
    val_dataset = HFDataset.from_dict({
        "text": dataset_dict["validation"]["text"],
        "summary": dataset_dict["validation"]["summary"]
    })
    
    # Tokenize datasets
    def preprocess_function(examples):
        inputs = examples["text"]
        targets = examples["summary"]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
        
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        # Use eval_strategy instead of evaluation_strategy
        eval_strategy="steps",  # Changed from evaluation_strategy
        save_strategy="steps",
        eval_steps=100,  # Evaluate every 100 steps
        save_steps=100,  # Save every 100 steps
        num_train_epochs=epochs,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    # Set paths
    dataset_dir = os.path.join(os.path.dirname(__file__), "data", "iiit_spoken_language_datasets")
    output_dir = os.path.join(os.path.dirname(__file__), "models", "summarizer", "finetuned_bart")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Fine-tune BART
    finetune_bart(dataset_dir, output_dir)