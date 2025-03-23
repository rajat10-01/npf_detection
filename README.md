# NPF Detection

This repository contains a fast deep learning algorithm, You Only Look Once (YOLO) for detecting new particle formation (NPF) events.

## Prerequisites

- Python 3.10
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rajat10-01/npf_detection.git
cd npf_detection
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your .xlsx data files in the `datasets` folder
   - A sample dataset is provided for testing
   - Please make sure your data file is in the same format as sample data, including sheet and variable names

2. Run the evaluation script:
```bash
python eval.py
```

3. Follow the prompts:
   - Press Enter when asked about the file
   - Enter site name (e.g., "delhi") when prompted
   - Results will be generated in the `results` folder

## Results

The script will generate:
- Contour plots for each day
- NPF event detection results
- Growth rate calculations
