# update_summarizer.py

import os
import shutil
from pathlib import Path

def update_summarizer():
    """
    Updates the summarizer.py file to use the fine-tuned BART model instead of the pre-trained one.
    """
    # Define paths
    summarizer_path = Path("summarizer.py")
    backup_path = Path("summarizer.py.bak")
    finetuned_model_path = Path("models/summarizer/finetuned_bart")
    
    # Check if fine-tuned model exists
    if not finetuned_model_path.exists():
        print(f"Error: Fine-tuned model not found at {finetuned_model_path}")
        print("Please run finetune_bart.py first to create the fine-tuned model.")
        return False
    
    # Create backup of original summarizer.py
    if summarizer_path.exists():
        print(f"Creating backup of {summarizer_path} to {backup_path}")
        shutil.copy2(summarizer_path, backup_path)
    else:
        print(f"Error: {summarizer_path} not found")
        return False
    
    # Read the original file
    with open(summarizer_path, 'r') as f:
        content = f.read()
    
    # Replace the model path
    # Original: model_name: str = "facebook/bart-large-cnn"
    # New: model_name: str = "./models/summarizer/finetuned_bart"
    updated_content = content.replace(
        'model_name: str = "facebook/bart-large-cnn"',
        'model_name: str = "./models/summarizer/finetuned_bart"'
    )
    
    # Write the updated content back to the file
    with open(summarizer_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Successfully updated {summarizer_path} to use the fine-tuned model.")
    print("You can now use the fine-tuned BART model for summarization.")
    return True

if __name__ == "__main__":
    update_summarizer()