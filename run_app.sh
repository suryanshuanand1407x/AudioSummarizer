#!/bin/bash

# Run the application using Python 3.10 virtual environment
echo "Starting Audio → Text Summarizer with Python 3.10..."

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the main.py script with Python 3.10 from the virtual environment
"$SCRIPT_DIR/venv310/bin/python3" "$SCRIPT_DIR/main.py"