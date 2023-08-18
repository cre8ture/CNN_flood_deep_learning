import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

class analysis:
    def __init__(self, verbose = True):
        self.filepath = ''
        # self.data = None
        self.verbose = verbose
        

    def set_file_name(self, file_name):
        self.file_name = file_name

    def set_folder_directory(self, folder_directory):
        self.folder_directory = folder_directory

    def preprocess(self):
        # preprocess lisflood outputs
        data = pd.read_fwf(self.folder_directory + self.file_name)
    
        return data
       
        

    def plot_results(self, file_name, save_file_prefix, sim_start='2010-01-15', sim_end='2010-01-18', cell_width=30, 
                    Q_min=0, Q_max=250, V_min=0, V_max=2.5e6, margin=0):

        self.set_file_name(file_name=file_name)
        data = self.preprocess()

        data['Vol_diff'] = data['Vol'] - data['Vol'].shift(1)
        data['Vol_diff'][0] = 0

        if self.verbose:
            print(f"data with volume difference:{data}")


        # %%
        # linechart 1 
        y0 = data['Qin']
        y1 = data['Qout']
        y2 = data['Vol']
        x = data['Time']

        fig, ax  = plt.subplots(1, 3, figsize=(15,10))

        ax[0].plot(x, y0)
        ax[0].set_title(f"lisflood-fp output (Qin)\n{sim_start} ~ {sim_end} {str(cell_width)}m")
        ax[0].set_xlabel("time step (s)")
        ax[0].set_ylabel(r'Qin ($m^{3}/s$)')  #r'flow ($m^{3}/s$)'
        ax[0].set_ylim(ymin=Q_min, ymax=Q_max)
        # ax[0].margins(margin)
        ax[0].grid(True,linestyle='-',color='0.75')
        # ax[0].autoscale_view()
        plt.setp( ax[0].xaxis.get_majorticklabels(), rotation=45)
        
        ax[1].plot(x, y1)
        ax[1].set_title(f"lisflood-fp output (Qout)\n{sim_start} ~ {sim_end} {str(cell_width)}m")
        ax[1].set_xlabel("time step (s)")
        ax[1].set_ylabel(r'Qout ($m^{3}/s$)')  #r'flow ($m^{3}/s$)'
        ax[1].set_ylim(ymin=Q_min, ymax=Q_max)
        # ax[1].margins(margin)
        ax[1].grid(True,linestyle='-',color='0.75')
        plt.setp( ax[1].xaxis.get_majorticklabels(), rotation=45)
        
        ax[2].plot(x, y2)
        ax[2].set_title(f"lisflood-fp output (Volume)\n{sim_start} ~ {sim_end} {str(cell_width)}m")
        ax[2].set_xlabel("time step (s)")
        ax[2].set_ylabel(r'volume ($m^{3}$)')  #r'flow ($m^{3}/s$)'
        ax[2].set_ylim(ymin=V_min, ymax=V_max)
        # ax[2].margins(margin)
        ax[2].grid(True,linestyle='-',color='0.75')
        plt.setp( ax[2].xaxis.get_majorticklabels(), rotation=45)
        
        

        plt.savefig(self.folder_directory + save_file_prefix + '.png')
        # plt.savefig(self.folder_directory + '_mass_results.png')
        plt.close()
        # plt.show()
        
    def plot_grid_data(self, file_name, save_file_prefix, extent=[315467, 319607, 228644, 232304], 
    title='max flood depth', vmin=0, vmax=13):
        
        self.set_file_name(file_name=file_name)
        data = self.preprocess()
        
      

        # remove the header of the file
        data = data.iloc[5:,:]
        if self.verbose:
            print(f"grid data: \n{data}")

        # seperate the 1st column based on \t
        data_split = data.iloc[:, 0].str.split("\t", expand=True)
        
        if self.verbose:
            print(f"data after split:\n{data_split}")   

        data_concat = pd.concat([data_split, data.iloc[:, 1:]], axis=1)
        data_concat = data_concat.apply(pd.to_numeric)
        if self.verbose:
            print(f"data: \n{data_concat} {data_concat.shape}")

        data_array = data_concat.to_numpy()
        if self.verbose:
            print(f"data_array:\n{data_array}")
        fig, ax = plt.subplots(1)
        img = ax.imshow(data_array, cmap='coolwarm', extent=extent, vmin=vmin, vmax=vmax)
        # np.flip(data_array, 0), 
        
        # make a color bar
        fig.colorbar(img)

        filename = self.filepath
        # regex = r'/(.*).[a-zA-z]+'
        # title = filename.split("/")[-1]
        # filename_prefix = filename.rsplit(".", 1)[0]
        # if self.verbose:
        #     print(f"filename: {filename} \
        #           filename_prefix: {filename_prefix}")
            
        ax.set_title(title)
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=45)
        plt.savefig(self.folder_directory + save_file_prefix + ".png")
        # plt.show()

    def make_video(self, file_prefix='res_dodder_20111021_1028_90m', dir='./lisflood'):
        
        filename_list = []
        num_list = []
        prefix = file_prefix
        cwd = os.getcwd()
        print(f"current workding dir:{cwd}")
        for file in os.listdir(dir):
            if file.startswith(file_prefix):
                print(os.path.join(dir, file))
                filename_list.append(file)
        
        regex_file = r"-\d{4}.wd"
        regex_num = r"\d+"
        for filename in filename_list:
            num_file = re.findall(regex_file, filename)
            # print(num_file)
            if len(num_file) != 0:
                num = re.findall(regex_num, num_file[0])
                num_list.append(int(num[0]))


        sorted_num_list = sorted(num_list)
        sorted_filename_list = []
        for num in sorted_num_list:
            temp_filename = file_prefix + f"-{num:04}" + ".wd"
            sorted_filename_list.append(temp_filename)

        print(sorted_filename_list)
        
        
