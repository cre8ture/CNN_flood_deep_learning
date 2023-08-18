import HydroGauge as hg
import numpy as np
import os, re, fnmatch
import pandas as pd
import matplotlib.pyplot as plt


class preprocess4DL(hg.HydroGauge):
    def __init__(self):

        super().__init__()

    def set_folder_directory(self, folder_directory: str):
        self.folder_directory = folder_directory


    def make_synthetic_hydrograph(self, date_start:str, date_end:str, 
                                   folder_directory:str,
                                   file_name:str,
                                   sf=2):
        
        super().set_file_name(file_name=file_name)
        super().set_folder_directory(folder_directory=folder_directory)
        
        time, Q_syn, Q_obs = super().produce_synthetic_hydrograph(date_start=date_start, 
                                                                      date_end=date_end, 
                                                                      scaling_factor=sf)


        return time, Q_syn, Q_obs

    
    def make_input_matrix(self, time, Q_list:list,
                          is_train_set=False,
                          is_val_set=False,
                          is_test_set=False):
        
        '''
        Q_list: list. Include all the hydrographs (observed and/or synthetic)
        is_train_set, is_val_set and is_test_set can't be True at the same time
        '''
        logicals = [is_train_set, is_test_set, is_val_set]
        assert any(logicals), "three logical values can't be false at the same time"

        # generate time step from 1 to T
        time_steps = np.arange(start=1, stop=len(time)-7)
        time_steps = np.expand_dims(time_steps, axis=1)
        

        # plot Q_list
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_ylabel("flow ($\mathregular{m^{3}/s}$)")
        ax.set_xlabel("time steps")

        curve_num = 'A'
        event_num = 1
        input_matrix = []
        for Q in Q_list:

            Qt = Q[8:]
            Qt_1 = Q[7: -1]
            Qt_2 = Q[6: -2]
            Qt_3 = Q[5: -3]
            Qt_4 = Q[4: -4]
            Qt_5 = Q[3: -5]
            Qt_6 = Q[2: -6]
            Qt_7 = Q[1: -7]
            Qt_8 = Q[0: -8]

            # check shapes of Qs
            assert Qt.shape == Qt_1.shape, "Qt and Qt_1 should be the same shape"
            assert Qt.shape == Qt_2.shape, "Qt and Qt_2 should be the same shape"
            assert Qt.shape == Qt_3.shape, "Qt and Qt_3 should be the same shape"
            assert Qt.shape == Qt_4.shape, "Qt and Qt_4 should be the same shape"
            assert Qt.shape == Qt_5.shape, "Qt and Qt_5 should be the same shape"
            assert Qt.shape == Qt_6.shape, "Qt and Qt_6 should be the same shape"
            assert Qt.shape == Qt_7.shape, "Qt and Qt_7 should be the same shape"
            assert Qt.shape == Qt_8.shape, "Qt and Qt_8 should be the same shape"

            # expand dimension 
            Qt = np.expand_dims(Qt, axis=1)
            Qt_1 = np.expand_dims(Qt_1, axis=1)
            Qt_2 = np.expand_dims(Qt_2, axis=1)
            Qt_3 = np.expand_dims(Qt_3, axis=1)
            Qt_4 = np.expand_dims(Qt_4, axis=1)
            Qt_5 = np.expand_dims(Qt_5, axis=1)
            Qt_6 = np.expand_dims(Qt_6, axis=1)
            Qt_7 = np.expand_dims(Qt_7, axis=1)
            Qt_8 = np.expand_dims(Qt_8, axis=1)
            Q_concat = np.concatenate((time_steps, Qt, Qt_1, Qt_2, Qt_3, Qt_4, Qt_5, Qt_6, Qt_7, Qt_8), axis=1)
          
            input_matrix.append(Q_concat)

            if is_train_set or is_val_set:
                ax.plot(time_steps, Qt, label='hydrograph ' + curve_num)
                curve_num = chr(ord(curve_num) + 1)
            elif is_test_set:
                ax.plot(time_steps, Qt, label='hydrograph (event ' + str(event_num)+ ')')
                event_num = event_num + 1

        input_matrix = np.array(input_matrix)
        num_of_graphs, tsteps, features = input_matrix.shape
        input_matrix = input_matrix.reshape(num_of_graphs*tsteps, features)

        if is_train_set:
            assert not is_val_set and not is_test_set, "logical values can't be True at the same time"
            np.savetxt("./data/"
                                    +"train_input_matrix.csv",
                                    input_matrix, fmt = '%.3f', delimiter=",")
            
            ax.set_title("hydrogrphs (training)")
            plt.legend()
            plt.savefig("./img/"
                        + "traing_hydrographs.png")
            plt.close()

        elif is_val_set:
            assert not is_train_set and not is_test_set, "logical values can't be True at the same time"
            np.savetxt("./data/"
                                    +"val_input_matrix.csv",
                                    input_matrix, fmt = '%.3f', delimiter=",")
            
            ax.set_title("hydrogrphs (validation)")
            plt.legend()
            plt.savefig("./img/"
                        + "validation.png")
            plt.close()

        elif is_test_set:
            assert not is_train_set and not is_val_set, "logical values can't be True at the same time"
            np.savetxt("./data/"
                                    +"test_input_matrix.csv",
                                    input_matrix, fmt = '%.3f', delimiter=",")
            
            ax.set_title("hydrogrphs (testing)")
            plt.legend()
            plt.savefig("./img/"
                        + "testing_hydrographs.png")
            plt.close()


            
    def make_target_matrix_per_hydrograph(self, folder_dir:str, 
                           file_name_identifier:str):
        file_list = []
        for f in os.listdir(folder_dir):
            if fnmatch.fnmatch(f, file_name_identifier):
                file_list.append(f)

        
        num_of_file = len(file_list)
        regex_file = r"-\d{4}.wd"
        regex_num = r"\d{4}"
        arr_list = []
        for file in file_list:
            num = re.search(regex_file, file)
            num = re.search(regex_num, num[0])
            
            data = pd.read_fwf(folder_dir + file)
            # remove headers
            data = data.iloc[5:, :]
            # seperate the 1st column based on \t
            data_split = data.iloc[:, 0].str.split("\t", expand=True)


            data_concat = pd.concat([data_split, data.iloc[:, 1:]], axis=1)
            data_concat = data_concat.apply(pd.to_numeric)
            

            data_array = data_concat.to_numpy()
            

            # flatten array
            data_array = data_array.flatten()
            
            num = np.array(int(num[0]))

            num_data_array = np.append(num, data_array)

            
            # expand dimension
            num_data_array = np.expand_dims(num_data_array, axis=1)
            arr_list.append(num_data_array)

        con = np.stack(arr_list,axis=1)
        
        con = con.reshape(con.shape[0], con.shape[1]*con.shape[2])
        con = con.transpose()
        

        # sort by first column (time step)
        con = con[con[:, 0].argsort()]
        
        # get rid of the time step column and the first eight rows of water depths
        output = con[8:, 1:]
        
        return output
            
    def make_target_matrix(self, matrix_list:list,
                           is_train_set=False,
                          is_val_set=False,
                          is_test_set=False):
        
        logicals = [is_train_set, is_test_set, is_val_set]
        assert any(logicals), "three logical values can't be false at the same time"

        output = np.stack(matrix_list,axis=0)
        
        output = output.reshape(output.shape[0]*output.shape[1], output.shape[2])
        

        if is_train_set:
            assert not is_val_set and not is_test_set, "logical values can't be True at the same time"
            np.savetxt("./data/"
                                    +"train_target_matrix.csv",
                                    output, fmt = '%.3f', delimiter=",")
        elif is_val_set:
            assert not is_train_set and not is_test_set, "logical values can't be True at the same time"
            np.savetxt("./data/"
                                    +"val_target_matrix.csv",
                                    output, fmt = '%.3f', delimiter=",")
        elif is_test_set:
            assert not is_train_set and not is_val_set, "logical values can't be True at the same time"
            np.savetxt("./data/"
                                    +"test_target_matrix.csv",
                                    output, fmt = '%.3f', delimiter=",")