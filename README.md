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

## Training the Model

### Dataset Structure
The training dataset should be organized in the following structure:
```
training_dataset/
├── images/
│   ├── train/
│   │   └── [training images]
│   └── val/
│       └── [validation images]
├── labels/
│   ├── train/
│   │   └── [training labels]
│   └── val/
│       └── [validation labels]
└── data.yaml
```

The `data.yaml` file should contain:
```yaml
names:
- npf
nc: 1
path: [path to training_dataset folder]
train: images/train
val: images/val
```

### Training Process
1. Prepare your dataset following the structure above
2. Run the training script:
```bash
python train.py --model yolov8m.pt --data ./training_dataset/data.yaml --epochs 100 --batch-size 2
```

Optional arguments:
- `--model`: Path to the model (default: yolov8m.pt)
- `--data`: Path to the data yaml file (default: ./training_dataset/data.yaml)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 2)
- `--validate`: Run validation after training
- `--val-model`: Path to model for validation (defaults to best training weights)

The training results including newly trained model will be saved in the `training_result` folder.

## Results

The script will generate:
- Contour plots for each day
- NPF event detection results
- Growth rate calculations

## Output Files

After running the evaluation script, the following output files will be generated for each site in the `results` folder:

### Contour Plots
- Daily contour plots showing particle size distribution over time
- Located in the `contour_plot` subfolder
- If mode plotting is enabled, additional plots with mode lines will be in `contour_plot_with_mode`

### NPF Event Detection
- Detection results from the YOLO model are saved in the `predictions` subfolder
- Visualizations of detected NPF events with bounding boxes

### Growth Rate Analysis
- Mode plots for NPF events are saved in the `NPF_modes` subfolder
- An Excel file named `event_data.xlsx` is generated in each site's folder with the following information:
  - **Date**: Date of the NPF event (format: DD/MM/YYYY)
  - **start_time**: Start time of the NPF event
  - **growth_rate_0_25**: Growth rate for particles <25nm (nm/hour)
  - **growth_rate_25_50**: Growth rate for particles 25-50nm (nm/hour)
  - **growth_rate_50_80**: Growth rate for particles 50-80nm (nm/hour)
  - **confidence**: Confidence score of the detection

Note: Empty cells in the growth rate columns indicate negative growth rates.


## Correspondance

For any queries, please contact:
- Email: rajat.b@alumni.iitm.ac.in
- CC: vijaykanawade03@yahoo.co.in, chandansarangi@civil.iitm.ac.in
