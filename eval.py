from data_handler import DataReader
from npf_detection import NPFDetection
from utils import load_model

if __name__ == "__main__":
    model = load_model('npf_detector_v2.pt')
    dr = DataReader(folder_path='datasets', file_extension='.xlsx', min_size=3, max_size=1000)
    for site in dr:
        try:
            npf = NPFDetection(data=site, plot_mode=True)
            npf.plot()
            npf.predict(model=model)
            npf.gr_calculations()
        except Exception as e:
            print(f"Error while processing site {site.site_name} data: {e}")
