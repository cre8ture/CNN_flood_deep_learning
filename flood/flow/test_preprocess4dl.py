# import HydroGauge as hg
# import River as rv

import Preprocess4DeepLearning 


  

def main(make_input_matrix=False,
         make_targe_matrix=False,
         is_train_set=True,
         is_test_set=True):
    
    

    if make_input_matrix:

        if is_train_set:
            p4dl = Preprocess4DeepLearning.preprocess4DL()
            p4dl.set_folder_directory(folder_directory='./flow/')

            sf_list = [2,3]

            kwargs_list = [
                {
                'date_start':'2014-11-12', 
                'date_end':'2014-11-19',
                'folder_directory':'./data/',
                'file_name':'WALDRON_S_BRIDGE_flow_complete_15min.csv',
                'sf':2
                },
                {
                'date_start':'2014-11-12', 
                'date_end':'2014-11-19',
                'folder_directory':'./data/',
                'file_name':'WALDRON_S_BRIDGE_flow_complete_15min.csv',
                'sf':3
                },
                {
                'date_start':'2018-03-12', 
                'date_end':'2018-03-19',
                'folder_directory':'./data/',
                'file_name':'WALDRON_S_BRIDGE_flow_complete_15min.csv',
                'sf':2
                },
                {
                'date_start':'2018-03-12', 
                'date_end':'2018-03-19',
                'folder_directory':'./data/',
                'file_name':'WALDRON_S_BRIDGE_flow_complete_15min.csv',
                'sf':3
                },


            ]



            time_list = []
            Q_syn_list = []
            Q_obs_list = []

            for kwargs in kwargs_list:

                time, Q_syn, Q_obs = p4dl.make_synthetic_hydrograph(**kwargs)

                time_list.append(time)
                Q_syn_list.append(Q_syn)
                Q_obs_list.append(Q_obs)

            p4dl.make_input_matrix(time=time,Q_list=Q_syn_list,
                                is_train_set=is_train_set)
            
        if is_test_set:

            p4dl = Preprocess4DeepLearning.preprocess4DL()
            p4dl.set_folder_directory(folder_directory='./flow/')
            # sf_list = [2,3]

            kwargs_list = [
                {
                'date_start':'2010-01-13', 
                'date_end':'2010-01-20',
                'folder_directory':'./data/',
                'file_name':'WALDRON_S_BRIDGE_flow_complete_15min.csv'
                
                },
                {
                'date_start':'2011-10-21', 
                'date_end':'2011-10-28',
                'folder_directory':'./data/',
                'file_name':'WALDRON_S_BRIDGE_flow_complete_15min.csv'
                
                }

            ]



            time_list = []
            Q_syn_list = []
            Q_obs_list = []

            for kwargs in kwargs_list:

                time, Q_syn, Q_obs = p4dl.make_synthetic_hydrograph(**kwargs)

                time_list.append(time)
                Q_syn_list.append(Q_syn)
                Q_obs_list.append(Q_obs)
            # use observed discharge for testing
            p4dl.make_input_matrix(time=time,Q_list=Q_obs_list,
                                is_test_set=is_test_set)

        
    if make_targe_matrix:
        if is_train_set:
            p4dl = Preprocess4DeepLearning.preprocess4DL()
            p4dl.set_folder_directory(folder_directory='./lisflood/')
            # p4dl.make_target_matrix_per_hydrograph(folder_dir='/home/chanyu_yang/LISFLOOD-FP/v8/LISFLOOD-FP-IO/30m/results_dodder_20141112_1119_sf2_30m_fill/',
            #                         file_name_identifier='*.wd')
            
            suffix_list = [
                'results_dodder_20141112_1119_sf2_30m_fill/',
                'results_dodder_20141112_1119_sf3_30m_fill/',
                'results_dodder_20180312_0319_sf2_30m_fill/',
                'results_dodder_20180312_0319_sf3_30m_fill/'
            ]

            matrix_list = []
            for suffix in suffix_list:

                output = p4dl.make_target_matrix_per_hydrograph(folder_dir='./data/lisflood_results/' + suffix,
                                    file_name_identifier='*.wd')
                
                matrix_list.append(output)

            p4dl.make_target_matrix(matrix_list=matrix_list,
                                    is_train_set=is_train_set)
            
        if is_test_set:

            p4dl = Preprocess4DeepLearning.preprocess4DL()
            p4dl.set_folder_directory(folder_directory='./lisflood/')
        

            suffix_list = [
                'results_dodder_20100113_0120_30m_fill/',
                'results_dodder_20111021_1028_30m_fill/'
            ]

            matrix_list = []
            for suffix in suffix_list:

                output = p4dl.make_target_matrix_per_hydrograph(folder_dir='./data/lisflood_results/' + suffix,
                                    file_name_identifier='*.wd')
                
                matrix_list.append(output)

            p4dl.make_target_matrix(matrix_list=matrix_list,
                                    is_test_set=is_test_set)

if __name__ == "__main__":

    main(
        make_input_matrix=True,
        make_targe_matrix=True
        # is_test_set=False
    ) 




