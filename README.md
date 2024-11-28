# NPF Detection

This repository contains the implementation for New Particle Formation (NPF) events detection using Computer Vision techniques.

## Prerequisites

- Python 3.10
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/npf_detection.git
cd npf_detection
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the pre-trained model:
- Visit: https://drive.google.com/drive/folders/1nMZMuqnC2mWLId2DiSAasJM5RCXzB1cU
- Download `npf_detector_v2.pt`
- Place it in the `models` folder (create if it doesn't exist)

## Usage

1. Place your .xlsx data files in the `datasets` folder
   - A sample dataset is provided for testing

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
