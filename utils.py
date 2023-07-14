import os


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
