import os

from ultralytics import YOLO


def load_model(model_name: str = 'npf_detector_v2.pt'):
    print('Loading model: ' + model_name)
    model_path = 'models/' + model_name
    if not os.path.exists(model_path):
        os.makedirs('models', exist_ok=True)
        print("Model file does not exist in 'models' folder. Please run train.py to create the model.")
        print("Or in case already trained, please put the model file in 'models' folder.")
        exit(1)
    model = YOLO(model_path)
    print('Model loaded')
    return model


def check_folder_existence(folder_path):
    print('Checking if folder with given path exists - ' + folder_path)
    if not os.path.exists(folder_path):
        print('Creating folder ' + folder_path)
        os.makedirs(folder_path)
        print('Folder created')
    else:
        print('Folder already exists')
    return folder_path


def seconds_to_hours_minutes(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds // 60) - (hours * 60))
    return f'{hours:02d}:{minutes:02d}'
