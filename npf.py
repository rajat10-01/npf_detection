import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as md
import datetime
import tqdm
from matplotlib import ticker

from data_handler import SiteData
from utils import check_folder_existence, seconds_to_hours_minutes
import statsmodels.api as sma
from sklearn.linear_model import TheilSenRegressor
lowess = sma.nonparametric.lowess


class NPFDetection:
    def __init__(self, data: SiteData, plot_mode: bool = False):
        self.results = None
        self.data = data.site_data
        self.site_name = data.site_name
        self.save_path = check_folder_existence(f'results/{self.site_name}')
        self.plot_mode = plot_mode
        self.mode_dp = self.get_max_conc_mode(self.data)
        self.ymin = data.min_size
        self.ymax = data.max_size
        self.grouped_data = self.data.groupby(self.data.index.date)

    @staticmethod
    def get_max_conc_mode(data: pd.DataFrame):
        """Returns the mode particle size (dp) where max concentration occurs, below 200nm."""
        mode_dp = data.idxmax(axis=1, skipna=True)
        return mode_dp[mode_dp < 200]

    def plot(self):
        """Plot contour for each day's data with optional mode plotting."""
        check_folder_existence(f'{self.save_path}/contour_plot')
        if self.plot_mode:
            check_folder_existence(f'{self.save_path}/contour_plot_with_mode')

        for date, day_data in tqdm.tqdm(self.grouped_data):
            try:
                self.plot_contour(date, day_data)
            except Exception as e:
                print(f"Error while plotting contour for {date}: {e}")

    def plot_contour(self, date: pd.Timestamp, day_data: pd.DataFrame):
        """Plots the contour plot for particle size distribution on a given day."""
        v_min, v_max = 0, np.quantile(day_data, 0.87)
        plt.figure(figsize=(12, 6))
        _X, _Y = np.meshgrid(day_data.index, day_data.columns, indexing='xy')
        _Z = day_data.T.values

        ax = plt.subplot()
        img = plt.contourf(_X, _Y, _Z, levels=800, cmap="jet", extend='both', vmax=v_max, vmin=v_min)
        self.format_contour_plot(ax, date, v_min, v_max, img)
        plt.savefig(f'{self.save_path}/contour_plot/{date}.jpg', bbox_inches="tight", dpi=300)

        if self.plot_mode:
            self.plot_modes(ax, date)
        plt.close()

    def format_contour_plot(self, ax, date, v_min, v_max, img):
        """Formats contour plot with axis, labels, and color bar."""
        ax.set_title(f'Contour Particle Size Distribution for {date}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Particle Size Dp (nm)')
        ax.set_xlim(date, date + pd.Timedelta(days=1))
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=6))

        norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=img.cmap))
        cbar.set_label('Concentration')
        cbar.ax.set_title('dN/dlogDp(cm$^{-3}$)')
        plt.xticks(rotation=45)

    def plot_modes(self, ax, date: pd.Timestamp):
        """Plots the modes on top of the contour plot."""
        modes = self.mode_dp.loc[date.strftime("%Y-%m-%d")]
        if modes.empty:
            return

        ax.scatter(modes.index, modes.values, color='blue', edgecolors='white', marker='o', s=15, alpha=0.8)
        cleaned_modes = self.outlier_remover(modes)

        if cleaned_modes.empty:
            return

        self.plot_smoothed_modes(ax, cleaned_modes)
        plt.savefig(f'{self.save_path}/contour_plot_with_mode/{date}.jpg', bbox_inches="tight", dpi=300)

    def plot_smoothed_modes(self, ax, modes):
        """Plots smoothed modes with rolling mean."""
        min_idx = self.find_min_point_idx(modes)
        if min_idx is not None:
            modes1 = modes.loc[min_idx:]
            max_idx = self.find_max_point_idx(modes1)
            if max_idx is not None:
                mav_modes = modes1.loc[:max_idx].rolling(5, center=True).mean()
                ax.plot(mav_modes.index, mav_modes.values, color='red', linewidth=2, alpha=0.8)

    @staticmethod
    def find_min_point_idx(data: pd.Series, threshold: float = 20):
        """Finds index of the minimum value if it's below a threshold."""
        min_idx = data.idxmin()
        return min_idx if data[min_idx] < threshold else None

    @staticmethod
    def find_max_point_idx(data: pd.Series, threshold: float = 40):
        """Finds index of the maximum value if it's above a threshold."""
        max_idx = data.idxmax()
        return max_idx if data[max_idx] > threshold else None

    @staticmethod
    def outlier_remover(data: pd.Series, lowess_factor: float = 0.1, std_factor: float = 1.5):
        """Removes outliers from data using LOWESS smoothing."""
        smoothed_modes = lowess(data.values, data.index, frac=lowess_factor, return_sorted=False)
        delta = np.abs(data - smoothed_modes)
        bounds = smoothed_modes.std() * std_factor
        cleaned_data = data[(data >= smoothed_modes - bounds) & (data <= smoothed_modes + bounds)]
        return cleaned_data

    def predict(self, model, folder_name: str = 'contour_plot', conf: float = 0.78, iou: float = 0.45):
        """Runs a YOLO model to predict events and saves results."""
        check_folder_existence(f'{self.save_path}/predictions')
        self.results = model.predict(
            source=f'{self.save_path}/{folder_name}', project=self.save_path,
            name='predictions', exist_ok=True, save=True, conf=conf, iou=iou, max_det=1, augment=False
        )

    def gr_calculations(self):
        """Calculates growth rates (GR) from prediction results and saves event data."""
        event_data = []
        check_folder_existence(f'{self.save_path}/NPF_modes')

        for result in self.results:
            try:
                date_str = os.path.splitext(os.path.relpath(result.path, self.save_path + '/contour_plot'))[0]
                box = result.boxes.xyxy.cpu().numpy()[0] if result.boxes.xyxy.shape[0] > 0 else None
                if box is not None:
                    event_data.append(self.process_event_data(box, date_str))
            except Exception as e:
                print(f"Error while calculating growth rate for {date_str}: {e}")

        pd.DataFrame(event_data, columns=['Date', 'x_min', 'y_min', 'x_max', 'y_max', 'start_time', 'end_time',
                                          'growth_rate_0_25', 'growth_rate_25_50', 'growth_rate_50_80']
                     ).to_excel(f'{self.save_path}/event_data.xlsx', index=False)

    def process_event_data(self, box, date_str):
        """Processes and returns a row of event data from YOLO predictions."""
        x_min, y_min, x_max, y_max = box
        start_time, end_time = self.convert_box_to_times(x_min, x_max)
        gr = self.find_gr(date_str, start_time)
        if gr is None:
            print(f'Not an NPF event: {date_str}')
            return None
        return [date_str, x_min, y_min, x_max, y_max, start_time, end_time, *gr]

    @staticmethod
    def convert_box_to_times(x_min, x_max, x1=191, x2=2421):
        """Converts bounding box coordinates into time intervals."""
        start_time_sec = ((x_min - x1) / (x2 - x1)) * 86400
        end_time_sec = ((x_max - x1) / (x2 - x1)) * 86400
        return seconds_to_hours_minutes(start_time_sec), seconds_to_hours_minutes(end_time_sec)

    def find_gr(self, date_str, start_time, size_range=None):
        """Calculates growth rates (GR) for a given date and start time."""
        if size_range is None:
            size_range = [(0, 25), (25, 50), (50, 80)]

        modes = self.mode_dp.loc[date_str].between_time(start_time, '23:59').reset_index(drop=True)
        if modes.empty:
            return None

        modes = self.outlier_remover(modes)
        smooth_modes = self.lowess_smoothening(modes)
        smooth_modes = self.remove_initial_decreasing_values(smooth_modes)

        return [self.get_slope(smooth_modes, sr) for sr in size_range]

    @staticmethod
    def lowess_smoothening(data: pd.Series, frac: float = 0.4):
        """Applies LOWESS smoothing to the given data."""
        smooth_data = lowess(data.values, data.index, frac=frac, return_sorted=False)
        return pd.Series(smooth_data, index=data.index)

    @staticmethod
    def remove_initial_decreasing_values(data: pd.Series):
        """Removes initial decreasing values in the data."""
        min_idx = data.idxmin()
        return data.loc[min_idx:] if not data.empty else pd.Series()

    @staticmethod
    def get_slope(data: pd.Series, size_range):
        """Calculates slope for the given particle size range."""
        relevant_modes = data[(data >= size_range[0]) & (data <= size_range[1])]
        if relevant_modes.empty:
            return None
        slopes = relevant_modes.diff() / relevant_modes.index.to_series().diff().dt.total_seconds() * 3600
        return slopes.mean()
