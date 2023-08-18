"""
This program is to analyse, and preprocess data for a river containing multiple gauges
"""


# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.ticker import FixedLocator
from datetime import timedelta
import inspect

import HydroGauge as HG


class River:

    def __init__(self, verbose=True):
        self.gauge_list = [],
        self.river_name = '',
        self.verbose = True
        self.sim_start = ''
        self.sim_end = ''
        self.verbose = verbose

    def set_verbose(self, verbose=True):
        self.verbose = verbose

    def set_gauges(self, gauge_list):
        self.gauge_list = gauge_list

    def set_river_name(self, river_name):
        self.river_name = river_name

    def set_folder_directory(self, folder_directory):
        self.folder_directory =  folder_directory

    def set_model_period(self, sim_start, sim_end):
        self.sim_start = sim_start
        self.sim_end = sim_end

    def clear_text(self, filepath):
        try:
            with open(filepath,'r+') as file:
                file.truncate(0)
        except FileNotFoundError:
            pass


    def make_bdy(self, time_unit='seconds', sim_start='2010-01-01', sim_end='2019-12-31', cell_width=30,
                 is_synthetic_hydrograph=False, **kwargs_syn):

        '''
        structure in .bdy:

        Line 1: Comment line, ignored by LISFLOOD-FP.Don't forget hashtag.
        Line 2: Boundary identifier (this should be consistent with notation supplied in the .river or .bci file).
        Line 3: Number of time points at which boundary information is given followed by a keyword for the time units used (either
        ‘days’, ‘hours’ or ‘seconds’).
        Line 4: Value1 Time1
        Line 5: Value2 Time2
        etc.... ... ...
        Line i: Valuei Timei       

        cell_width is the resolution of your model. It's set to 30m as default.
        '''
        self.sim_start = sim_start
        self.sim_end = sim_end 

        if is_synthetic_hydrograph:
            print("is synthetic ")
            scaling_factor = str(kwargs_syn['scaling_factor'])
            self.clear_text(filepath=self.folder_directory + self.river_name 
                        + '_' + self.sim_start + '_' +self.sim_end 
                         + '_' + 'sf' + scaling_factor 
                        + '_' + str(cell_width) + 'm.bdy')
            
        else:
            self.clear_text(filepath=self.folder_directory + self.river_name 
                            + '_' + self.sim_start + '_' +self.sim_end 
                            + '_' + str(cell_width) + 'm.bdy')
    


        for gauge in self.gauge_list:


            gauge.read_data()

            data = gauge.data
            

            if is_synthetic_hydrograph:
               
                scaling_factor = str(kwargs_syn['scaling_factor'])
                time, Q_syn, Q_obs = gauge.produce_synthetic_hydrograph(date_start=sim_start, date_end=sim_end, **kwargs_syn)
                sim_data_index_range = Q_syn.index
                
                check_nan = Q_syn.isnull()
                
                id_values_null = Q_syn.index[check_nan].tolist()
                

            else:
                sim_data = data.loc[self.sim_start: self.sim_end]
                sim_data_index_range = sim_data.index
                check_nan = sim_data.isnull()
                id_values_null = sim_data.index[check_nan.iloc[:,1]].tolist()
              
            if is_synthetic_hydrograph:

                # fill na using average
                if len(id_values_null) != 0:
                    Q_syn = (Q_syn.ffill() + Q_syn.bfill())/2  # fill nan with average value between the first and the last nan
                    Q_syn = Q_syn.bfill().ffill()

                modified_value = Q_syn/cell_width

            else:
                 # fill na using average
                if len(id_values_null) != 0:
                    sim_data.iloc[:, 1] = (sim_data.iloc[:, 1].ffill() + sim_data.iloc[:, 1].bfill())/2  # fill nan with average value between the first and the last nan
                    sim_data.iloc[:, 1] = sim_data.iloc[:, 1].bfill().ffill()  # fill the very first and very last nan

                # modify data based on cell width
                modified_value = sim_data.iloc[:, 1]/cell_width

            # count the total data points
            num_of_datapoints = len(modified_value)
            l3 = "{num_of_data}\t\t\t{time}".format(num_of_data=num_of_datapoints, time=time_unit)

            if is_synthetic_hydrograph:
                days_count = time - time.shift(1)
            else: 
                days_count = sim_data.iloc[:, 0] - sim_data.iloc[:, 0].shift(1)

            days_count_fillnat = days_count.fillna(pd.Timedelta(seconds=0))
            days_count_cum = days_count_fillnat.cumsum(axis=0)
            

            secs_count_cum = days_count_cum.dt.total_seconds()
  

            # organise the data to meet bdy format
            if is_synthetic_hydrograph: 


                with open(self.folder_directory + self.river_name 
                        + '_' + self.sim_start + '_' +self.sim_end
                         + '_' + 'sf' + scaling_factor 
                        + '_' + str(cell_width) + 'm.bdy', 'a+') as f:
                    
                    # don't forget the "#" at the beginning of the first row 
                    f.writelines(['# '+ gauge.l1, '\n', gauge.l2, '\n', l3, '\n'])

                    i = sim_data_index_range[0]
                    for i in sim_data_index_range:

                        l = "{value}\t\t\t{time}".format(value=modified_value.loc[i], time=secs_count_cum.loc[i])
                        f.writelines([l, '\n'])

            else:

                with open(self.folder_directory + self.river_name 
                        + '_' + self.sim_start + '_' +self.sim_end 
                        + '_' + str(cell_width) + 'm.bdy', 'a+') as f:
                    f.writelines(['# '+ gauge.l1, '\n', gauge.l2, '\n', l3, '\n'])

                    i = sim_data_index_range[0]
                    for i in sim_data_index_range:
                      
                        l = "{value}\t\t\t{time}".format(value=modified_value.loc[i], time=secs_count_cum.loc[i])
                        f.writelines([l, '\n'])


