import os
import glob
import pandas as pd


class SiteData:
    def __init__(self, site_name: str, site_data: pd.DataFrame, min_size: float, max_size: float):
        self.__site_name = site_name
        self.__site_data = self.preprocess_data(site_data, min_size, max_size)
        self.min_size = min_size
        self.max_size = max_size

    def preprocess_data(self, data: pd.DataFrame, min_size, max_size) -> pd.DataFrame:
        data.rename(columns={data.columns[0]: 'DateTime'}, inplace=True)
        data['DateTime'] = pd.to_datetime(data['DateTime'], format='%d/%m/%Y %H:%M:%S')
        data.set_index('DateTime', inplace=True)
        data.dropna(axis=1, how='all', inplace=True)
        data.dropna(axis=0, inplace=True)
        last_col = data.columns[-1]
        if isinstance(last_col, str) and 'mode' in last_col.lower():
            data.pop(last_col)
        column_names = data.columns
        size_columns = [col for col in column_names if self.is_within_size_range(col, min_size, max_size)]
        data = data[size_columns]
        return data

    @staticmethod
    def is_within_size_range(column_name, min_size: float, max_size: float) -> bool:
        if min_size is not None and float(column_name) < min_size:
            return False
        if max_size is not None and float(column_name) > max_size:
            return False
        return True

    @property
    def min_particle_size(self) -> float:
        return self.__site_data.columns[0]

    @property
    def max_particle_size(self) -> float:
        return self.__site_data.columns[-1]

    @property
    def site_name(self) -> str:
        return self.__site_name

    @property
    def site_data(self) -> pd.DataFrame:
        return self.__site_data


class DataReader:
    def __init__(self, min_size: float, max_size: float, folder_path='datasets', file_extension='.xlsx'):
        self.folder_path = folder_path
        self.file_extension = file_extension
        self.min_size = min_size
        self.max_size = max_size

    def __iter__(self):
        if not os.path.exists(self.folder_path):
            print(f"Folder '{self.folder_path}' does not exist. Please create one with excel files in it.")
            return
        files = glob.glob(os.path.join(self.folder_path, f"*{self.file_extension}"))
        if len(files) == 0:
            print(f"No files with extension '{self.file_extension}' in folder '{self.folder_path}'.")
            print("Please put atleast one .xlsx file as expected.")
            return
        for file_path in files:
            if self.should_process_file(file_path):
                yield from self.process_file(file_path)

    def process_file(self, file_path):
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        for sheet_name in sheet_names:
            resp = self.should_process_sheet(sheet_name)
            if resp != 's':
                site_name = resp
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                yield SiteData(site_name, df, min_size=self.min_size, max_size=self.max_size)

    @staticmethod
    def should_process_sheet(sheet_name):
        print("")
        response = input(f"Enter the site name for sheet '{sheet_name}' or press 's' to skip: ")
        return response

    @staticmethod
    def should_process_file(file_path):
        response = input(f"Reading file: {file_path}\nPress enter to continue or type 's' to skip: ")
        return response.lower() != 's'


if __name__ == "__main__":
    data_reader = DataReader()
    for sd1 in data_reader:
        print(sd1.site_data.columns)
