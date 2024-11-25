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
        self.save_path = check_folder_existence('results/' + self.site_name)
        self.plot_mode = plot_mode
        self.mode_dp = pd.Series(dtype=float)
        # self.ymin = float(self.data.columns[0])                #Doubt
        # self.ymax = min(float(self.data.columns[-1]), 500)              #Doubt
        self.ymin = data.min_size
        self.ymax = data.max_size
        self.preprocess_data()
        self.grouped_data = self.data.groupby(self.data.index.date)

    def preprocess_data(self):
        self.mode_dp = self.get_max_conc_mode(self.data)

    def plot_contour(self, date: pd.Timestamp, day_data: pd.DataFrame):
        v_min = np.quantile(day_data, 0.05)
        v_max = np.quantile(day_data, 0.95)
        delta = datetime.timedelta(days=1)

        # # Ensure v_min is strictly positive for log scale
        v_min = max(v_min, 1e0)

        # Create figure with specified size ratio
        fig, ax = plt.subplots(figsize=(5, 2.5))

        # Ensure all data is positive and handle zeros/negatives
        _Z = day_data.transpose().values
        
        # Create meshgrid
        _X, _Y = np.meshgrid(day_data.index, day_data.columns, copy=False, indexing='xy')

        # Create colour map with optimized levels
        img = ax.contourf(_X, _Y, _Z, levels=800, 
                          cmap="jet", 
                          vmin=v_min, 
                          vmax=v_max,
                          extend='both')  # Handle out-of-range values smoothly

        # Set axis labels and scales
        ax.set_xlabel('')
        ax.set_ylabel('D$_p$ (nm)')

        # Configure x-axis (time)
        plt.setp(ax.get_xticklabels(), rotation=0)
        ax.set_xlim(date, date + delta)
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(md.HourLocator(interval=6))

        # Configure y-axis (particle size)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        # Add date below plot
        date_str = date.strftime("%d/%m/%Y")
        plt.figtext(0.05, -0.05, date_str, ha='left')

        # Adjust layout using subplots_adjust for tighter packing
        fig.subplots_adjust(left=0.08, right=0.9, top=0.95, bottom=0.1)
        
        # Create colorbar with logarithmic scale
        cbar = fig.colorbar(img, ax=ax, shrink=0.85, pad=0.02, spacing='proportional')
        cbar.set_label('dN/dlogD$_p$ (cm$^{-3}$)', rotation=270, labelpad=15)

        # Set the colorbar scale to logarithmic
        cbar.ax.set_yscale('log')

        # Save figure
        plt.savefig(self.save_path + '/contour_plot/' + str(date) + '.jpg', bbox_inches="tight", dpi=300, pad_inches=0.02)
        if self.plot_mode:
            try:
                self.plot_modes(ax, date)
            except Exception as e:
                print(f"Error while plotting mode for {date}: {e}")

        # Close figure
        plt.close()

    def plot_modes(self, ax, date: pd.Timestamp):
        date_str = date.strftime("%Y-%m-%d")
        modes = self.mode_dp.loc[date_str]
        if modes.empty:
            plt.savefig(self.save_path + '/contour_plot_with_mode/' + str(date) + '.jpg',
                        bbox_inches="tight", dpi=300)
            return
        ax.scatter(modes.index, modes.values, color='blue', edgecolors='white', marker='o', s=15, alpha=0.8)
        modes = self.outlier_remover(modes, lowess_factor=0.3, std_factor=1)
        if modes.empty:
            plt.savefig(self.save_path + '/contour_plot_with_mode/' + str(date) + '.jpg',
                        bbox_inches="tight", dpi=300)
            return
        min_index = self.find_min_point_idx(modes)
        if min_index is not None:
            modes1 = modes.loc[min_index:]
            max_index = self.find_max_point_idx(modes1)
            if max_index is not None:
                modes1 = modes1.loc[:max_index]
                mav_modes = modes1.rolling(5, center=True).mean()
                ax.plot(mav_modes.index, mav_modes.values, color='red', alpha=0.8, linewidth=2)

        plt.savefig(self.save_path + '/contour_plot_with_mode/' + str(date) + '.jpg',
                    bbox_inches="tight", dpi=300)

    @staticmethod
    def find_min_point_idx(data: pd.Series, threshold: float = 20):
        min_idx = data.idxmin()
        if data[min_idx] < threshold:
            return min_idx
        else:
            return None

    @staticmethod
    def find_max_point_idx(data: pd.Series, threshold: float = 40):
        max_idx = data.idxmax()
        if data[max_idx] > threshold:
            return max_idx
        else:
            return None

    @staticmethod
    def outlier_remover(data: pd.Series, lowess_factor: float = 0.1, std_factor: float = 1.5):
        smoothed_modes = lowess(data.values, data.index, frac=lowess_factor, return_sorted=False)
        delta = np.abs(data - smoothed_modes)
        lb = smoothed_modes - std_factor * delta.std()
        ub = smoothed_modes + std_factor * delta.std()
        cleaned_data = data[(data >= lb) & (data <= ub)]
        return cleaned_data

    @staticmethod
    def lowess_fit(data: pd.Series, lowess_factor: float = 0.05):
        smooth_data = lowess(data.values, data.index, frac=lowess_factor, return_sorted=False)
        smooth_data = pd.Series(smooth_data, index=data.index)
        return smooth_data

    def plot(self):
        check_folder_existence(self.save_path + '/contour_plot')
        if self.plot_mode:
            check_folder_existence(self.save_path + '/contour_plot_with_mode')
        for date, day_data in tqdm.tqdm(self.grouped_data):
            try:
                self.plot_contour(date, day_data)
            except Exception as e:
                print(f"Error while plotting contour for {date}: {e}")

    def predict(self, model, folder_name: str = 'contour_plot', conf: float = 0.78, iou: float = 0.45):
        check_folder_existence(self.save_path + '/predictions')
        self.results = model.predict(self.save_path + '/' + folder_name, project=self.save_path,
                                     name='predictions', exist_ok=True, save=True, max_det=1, conf=conf, iou=iou,
                                     show=False, stream=True, augment=False)

    def gr_calculations(self):
        event_data = []
        check_folder_existence(self.save_path + '/NPF_modes')
        for result in self.results:
            path = os.path.relpath(result.path, self.save_path + '/contour_plot')
            date_str = os.path.splitext(path)[0]
            boxes = result.boxes.xyxy
            if boxes.shape[0] > 0:
                box = boxes[0]
                box = box.cpu().numpy()
                # Get confidence score for the first detection
                conf_score = result.boxes.conf[0].item()
                x_min, y_min, x_max, y_max = box
                x1, x2 = 208, 1228
                start_time_sec = ((x_min - x1) / (x2 - x1)) * 86400
                end_time_sec = ((x_max - x1) / (x2 - x1)) * 86400
                start_time = seconds_to_hours_minutes(start_time_sec)
                end_time = seconds_to_hours_minutes(end_time_sec)
                try:
                    gr, start_time = self.find_gr(date_str, start_time)
                    if gr[0] is None and gr[1] is None and gr[2] is None:
                        print('Not an NPF event: ', date_str)
                        continue
                    sl25, sl50, sl80 = gr[0], gr[1], gr[2]
                    event_data.append([date_str, x_min, y_min, x_max, y_max, start_time, end_time, 
                                     sl25, sl50, sl80, conf_score])
                except Exception as e:
                    print(f"Error while calculating growth rate for {date_str}: {e}")
                    continue
                plt.close()
        event_data = pd.DataFrame(event_data, columns=['Date', 'x_min', 'y_min', 'x_max', 'y_max', 'start_time',
                                                      'end_time', 'growth_rate_0_25', 'growth_rate_25_50',
                                                      'growth_rate_50_80', 'confidence'])
        # Change Date from %Y-%m-%d to %d/%m/%Y
        event_data['Date'] = pd.to_datetime(event_data['Date']).dt.strftime('%d/%m/%Y')
        event_data.to_excel(self.save_path + '/event_data.xlsx', index=False)

    def find_gr(self, date, start_time, size_range=None):
        if size_range is None:
            size_range = [(0, 25), (25, 50), (50, 80)]
        date = pd.to_datetime(date)
        date_str = date.strftime("%Y-%m-%d")
        modes = self.mode_dp.loc[date_str]
        modes = modes.between_time(start_time, '23:59')
        modes.reset_index()
        modes = self.remove_high_start_values(modes)
        plt.scatter(modes.index, modes.values, color='blue', alpha=0.8, s=10)
        grs = []
        if modes.empty:
            return [None] * len(size_range), None
        modes = self.outlier_remover(modes, lowess_factor=0.4, std_factor=1.5)
        if modes.empty:
            return [None] * len(size_range), None
        modes.plot()
        smooth_modes = self.lowess_smoothening(modes)
        if smooth_modes.empty:
            return [None] * len(size_range), None
        smooth_modes = self.remove_initial_decreasing_values(smooth_modes)
        if smooth_modes.empty:
            return [None] * len(size_range), None
        smooth_modes.plot()
        new_st = smooth_modes.index[0].time()
        for sr in size_range:
            slope = self.get_slope(smooth_modes, sr)
            grs.append(slope)
        plt.savefig(self.save_path + '/NPF_modes/' + date_str + '.jpg', bbox_inches="tight", dpi=300)
        return grs, new_st

    @staticmethod
    def get_max_conc_mode(data: pd.DataFrame):
        mode_dp = data.idxmax(axis=1, skipna=True)
        mode_dp = pd.to_numeric(mode_dp, errors='coerce')
        mode_dp = mode_dp.dropna()
        mode_dp = mode_dp[mode_dp < 200]
        return mode_dp

    @staticmethod
    def lowess_smoothening(data: pd.Series):
        lowess_data = lowess(data.values, data.index, frac=0.4, return_sorted=False)
        lowess_data = pd.Series(lowess_data, index=data.index)
        return lowess_data

    @staticmethod
    def remove_high_start_values(data: pd.Series, threshold: int = 25) -> pd.Series:
        if data.empty:
            return pd.Series()
        while data.values[0] > threshold:
            if len(data) > 3:
                data = data[1:]
            else:
                return pd.Series(dtype=float)
        return data

    @staticmethod
    def remove_initial_decreasing_values(data: pd.Series) -> pd.Series:
        if data.empty:
            return pd.Series()
        idx = data.idxmin()
        data = data.loc[idx:]
        return data

    @staticmethod
    def get_slope(modes: pd.Series, size_range):
        if modes.empty or modes.values[0] > size_range[1]:
            return None
        indices1 = modes[modes.values >= size_range[0]].index
        indices2 = modes[modes.values >= size_range[1]].index
        index1 = indices1[0] if len(indices1) > 0 else None
        index2 = indices2[0] if len(indices2) > 0 else None
        if index2 is None:
            return None
        sliced_modes = modes.loc[index1:index2]
        if size_range[0] == 0:
            sliced_modes.plot(c='orange')
        elif size_range[0] == 25:
            sliced_modes.plot(c='red')
        elif size_range[0] == 50:
            sliced_modes.plot(c='green')
        slopes = (sliced_modes.diff() / sliced_modes.index.to_series().diff().dt.total_seconds()) * 3600
        avg_slope = slopes.mean()
        return avg_slope

    @staticmethod
    def get_regression_coef(data: pd.Series):
        x = np.array(data.index).reshape(-1, 1)
        y = data.values
        lr_model = TheilSenRegressor().fit(x, y)
        return lr_model.coef_[0]
