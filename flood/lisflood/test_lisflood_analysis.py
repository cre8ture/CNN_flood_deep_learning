import LISFLOOD_Analysis as LISFLOOD_Analysis
import os

def main(mass_data=True,
        grid_data=True,
        plot_all_water_depths=True,
        customise_dir=False):

        resolution_list = [30] #, 90, 270]
        sf_list = [2,3]
        

        if mass_data:

            for res in resolution_list:
                outputs = LISFLOOD_Analysis.analysis()

                if customise_dir:

                    for sf in sf_list:
                        outputs.set_folder_directory(f'/home/chanyu_yang/LISFLOOD-FP/v8/LISFLOOD-FP-IO/30m/results_dodder_20141112_1119_sf{str(sf)}_{res}m_fill/') # you can customise the folder directory you want to process your model results
                        outputs.plot_results(file_name=f'res_dodder_20141112_1119_sf{str(sf)}_{res}m_fill.mass', 
                        sim_start= '2014-11-12', sim_end='2014-11-19',cell_width=res,
                        save_file_prefix=f'res_dodder_20141112_1119_sf{str(sf)}_{res}m_fill.mass')

                        outputs.set_folder_directory(f'/home/chanyu_yang/LISFLOOD-FP/v8/LISFLOOD-FP-IO/30m/results_dodder_20180312_0319_sf{str(sf)}_{res}m_fill/') # you can customise the folder directory you want to process your model results
                        outputs.plot_results(file_name=f'res_dodder_20180312_0319_sf{str(sf)}_{res}m_fill.mass', 
                        sim_start= '2018-03-12', sim_end='2018-03-19',cell_width=res,
                        save_file_prefix=f'res_dodder_20180312_0319_sf{str(sf)}_{res}m_fill.mass')

                else:
                
                    outputs.set_folder_directory('./lisflood/')
                    outputs.plot_results(file_name=f'res_dodder_20100113_0120_{res}m.mass', 
                    sim_start= '2010-01-13', sim_end='2010-01-20',cell_width=res,
                    save_file_prefix=f'res_dodder_20100113_0120_{res}m.mass')

                    outputs.set_folder_directory('./lisflood/')
                    outputs.plot_results(file_name=f'res_dodder_20111021_1028_{res}m.mass',
                    sim_start= '2011-10-21', sim_end='2011-10-28', cell_width=res, 
                    save_file_prefix=f'res_dodder_20111021_1028_{res}m.mass')

      
        if grid_data:
            
            for res in resolution_list:
                outputs = LISFLOOD_Analysis.analysis()

                if customise_dir:
                    for sf in sf_list:
                        outputs.set_folder_directory(f'/home/chanyu_yang/LISFLOOD-FP/v8/LISFLOOD-FP-IO/30m/results_dodder_20141112_1119_sf{str(sf)}_{res}m_fill/')  # you can customise the folder directory you want to process your model results
                        outputs.plot_grid_data(file_name=f'res_dodder_20141112_1119_sf{str(sf)}_{res}m_fill.max',
                                        save_file_prefix=f'res_dodder_20141112_1119_sf{str(sf)}_{res}m_fill.max',
                                        extent=[315467, 319607, 228644, 232304],
                                            title=f'max flood depth\n(cell width: {res}m, scaling factor:{str(sf)}) \n2014-11-12 - 2014-11-19')
                        
                        outputs.set_folder_directory(f'/home/chanyu_yang/LISFLOOD-FP/v8/LISFLOOD-FP-IO/30m/results_dodder_20180312_0319_sf{str(sf)}_{res}m_fill/')  # you can customise the folder directory you want to process your model results
                        outputs.plot_grid_data(file_name=f'res_dodder_20180312_0319_sf{str(sf)}_{res}m_fill.max',
                                        save_file_prefix=f'res_dodder_20180312_0319_sf{str(sf)}_{res}m_fill.max',
                                        extent=[315467, 319607, 228644, 232304],
                                            title=f'max flood depth\n(cell width: {res}m, scaling factor:{str(sf)}) \n2018-03-12 - 2018-03-19')
                        

                else:
                    outputs.set_folder_directory('./lisflood/')
                    outputs.plot_grid_data(file_name=f'res_dodder_20100113_0120_{res}m.max',
                                        save_file_prefix=f'res_dodder_20100113_0120_{res}m.max',
                                        extent=[315467, 319607, 228644, 232304],
                                            title=f'max flood depth\n(cell width: {res}m) \n2010-01-13 - 2010-01-20')


                    outputs.set_folder_directory('./lisflood/')
                    outputs.plot_grid_data(file_name=f'res_dodder_20111021_1028_{res}m.max',
                                        save_file_prefix=f'res_dodder_20111021_1028_{res}m.max',
                                        extent=[315467, 319607, 228644, 232304],
                                            title=f'max flood depth\n(cell width: {res}m) \n2011-10-21 - 2011-10-28')


        

            if plot_all_water_depths:
                
                file_prefix='res_dodder_20111021_1028_90m'; file_extension='.wd';  dir='./lisflood'

                filename_list = []
                num_list = []
                prefix = file_prefix
                cwd = os.getcwd()
                print(f"current workding dir:{cwd}")
                for file in os.listdir(dir):
                    if file.startswith(file_prefix) and file.endswith(file_extension):
                        file = os.path.join(dir, file)
                        filename_list.append(file)

                for filename in filename_list:
                    outputs.plot_grid_data(file_path=filename, title="water depths (90m resolution)\n 2011-10-21 ~ 2011-10-28")

if __name__ == '__main__':
    main(mass_data=True,
         grid_data=True,
         plot_all_water_depths=False,
         customise_dir=True)