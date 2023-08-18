
# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from datetime import timedelta
import datetime


class HydroGauge:
    def __init__(self):
        self.gauge_name = '',
        self.river_name = '',
        self.hydro_datatype = '' # flow or water level
        self.file_path = ''
        self.verbose = True
        self.l1 = 'Comment line, ignored by LISFLOOD-FP.'
        self.l2 = 'Boundary identifier'
        

    def set_verbose(self, verbose=True):
        self.verbose = verbose

    def set_gauge_name(self, gauge_name):
        self.gauge_name = gauge_name

    def set_river_name(self, river_name):
        self.river_name = river_name

    def set_hydro_datatype(self, hydro_datatype):
        self.hydro_datatype = hydro_datatype

    def set_folder_directory(self, folder_directory:str):
        self.folder_directory =  folder_directory

    def set_file_name(self, file_name: str):
        self.file_name = file_name

    def set_comment(self, comment):
        self.l1 = comment

    def set_boundary_identifier(self, bd_id):
        self.l2 = bd_id


# %%
    def read_data(self):
        data = pd.read_csv(self.folder_directory + self.file_name)
        # get data from row 5 (ignore header description because it causes tokenising data error)
        modified_data = data[5:]
        split_data = modified_data.iloc[1:,0].str.split(';', expand=True)
        self.data = split_data
        
                
        self.data.iloc[:,0] = pd.to_datetime(self.data.iloc[:,0], format="%Y-%m-%d %H:%M:%S")  #'%Y-%m-%d %H:%M:%S'
        self.data = self.data.set_index(pd.DatetimeIndex(self.data.iloc[:,0]))
        self.data.iloc[:,1] = pd.to_numeric(self.data.iloc[:,1])


        self.time = self.data.iloc[:,0]
        index_time = self.time.index
        
        self.time_max = self.time[index_time[-1]]

        self.total_tsteps = self.data[0]
        self.total_values = self.data[1]

        

    def plot_data(self, date_start='2010-01-01', date_end='2019-12-31', y_min=0, y_max=250 ): # specify the folder you want to save your file

        print("plotting data...")
        format = '%Y-%m-%d'
        
        self.read_data()
        data = self.data
        time = data.loc[date_start: date_end][0]
        value = data.loc[date_start: date_end][1]
        
        value_max = value.max()

        # create a plot
        
        fig,ax = plt.subplots(figsize=(12,3))
        ax.set_title(self.gauge_name)
        ax.set_xlabel('time')
        ax.set_ylabel(r'flow ($m^{3}/s$)')
        ax.set_ylim(ymin=y_min, ymax=y_max)

        ax.plot(time, value)
        
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(MultipleLocator(12))
       

        plt.savefig(self.folder_directory + self.gauge_name + '_' + self.hydro_datatype + '_' + date_start + '_' + date_end +'.png', bbox_inches='tight')
        plt.close()  # so that the figure won't pop up if you want to apply plt.show() somewhere else

    def produce_synthetic_hydrograph(self, date_start='2010-01-13', date_end='2010-01-20', peak_flow=None, scaling_factor=2):

        self.read_data()
        data = self.data
        time = data.loc[date_start: date_end][0]
        Q_obs = data.loc[date_start: date_end][1]
        # check if anywhere blank 
        check_nan = Q_obs.isnull()
        
        id_values_null = Q_obs.index[check_nan].tolist()

        
        # fill na using average
        if len(id_values_null) != 0:
            Q_obs = (Q_obs.ffill() + Q_obs.bfill())/2  # fill nan with average value between the first and the last nan
            Q_obs = Q_obs.bfill().ffill()
        
        print("producing synthetic hydrographs...")
        

        Q_obs_max = Q_obs.max()
        
        if peak_flow is not None:
            # assert peak_flow > Q_obs_max
            assert peak_flow > Q_obs_max, "peak flow should be larger than Q_obs_max"

            Q_syn = Q_obs * (peak_flow/Q_obs_max)

        else:
            Q_syn = Q_obs * scaling_factor

        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(time, Q_obs, time, Q_syn)
        
        plt.savefig(self.folder_directory
                     + date_start 
                     + '_' + date_end 
                     + '_sf' + str(scaling_factor)
                     +'.png', bbox_inches='tight')
        plt.close()


        return time, Q_syn, Q_obs

    