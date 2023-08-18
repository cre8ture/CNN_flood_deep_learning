import HydroGauge as hg
import River as rv

def main(FRANKFORT=True, 
        WS_BR=True, 
        flow=True, 
        water_level=True,
        make_bdy=True,
        plot_data=True,
        produce_synthetic_hydrograph=True,
        original_data=True,
        synthetic_hydrograph=True):
    
    gauge_list = []
    
    if flow:



        if FRANKFORT:

            gauge = hg.HydroGauge()
            gauge.set_verbose = False
            gauge.set_river_name('Dodder')
            gauge.set_gauge_name('FRANKFORT')
            gauge.set_hydro_datatype('flow')
            gauge.set_folder_directory('./data/')
            gauge.set_file_name('FRANKFORT_flow_complete_15min.csv')
            gauge.set_comment(comment='Gauge in tributary Slang')
            gauge.set_boundary_identifier(bd_id = 'FRANKFORT')
            
            if plot_data:
                gauge.plot_data()
                gauge.plot_data(date_start='2010-01-13', date_end='2010-01-20', y_max=20)
                gauge.plot_data(date_start='2011-10-21', date_end='2011-10-28', y_max=20)
                gauge.plot_data(date_start='2014-11-12', date_end='2014-11-19', y_max=20)
                gauge.plot_data(date_start='2018-03-12', date_end='2018-03-19', y_max=20)

            if produce_synthetic_hydrograph:
                gauge.produce_synthetic_hydrograph(scaling_factor=2)

            gauge_list.append(gauge)
            
            

        if WS_BR:

            gauge = hg.HydroGauge()
            gauge.set_verbose = False
            gauge.set_river_name('Dodder')
            gauge.set_gauge_name('WALDRON_S_BRIDGE')
            gauge.set_hydro_datatype('flow')
            gauge.set_folder_directory('./data/')
            gauge.set_file_name('WALDRON_S_BRIDGE_flow_complete_15min.csv')
            gauge.set_comment(comment='Gauge in River Dodder')
            gauge.set_boundary_identifier(bd_id = 'WALDRON_S_BRIDGE')

            if plot_data:
            
                gauge.plot_data()
                gauge.plot_data(date_start='2010-01-13', date_end='2010-01-20')
                gauge.plot_data(date_start='2011-10-21', date_end='2011-10-28')
                gauge.plot_data(date_start='2014-11-12', date_end='2014-11-19')
                gauge.plot_data(date_start='2018-03-12', date_end='2018-03-19')


            gauge_list.append(gauge)


    if make_bdy:

        river = rv.River()
        river.set_verbose = False
        river.set_river_name("Dodder")
        river.set_gauges(gauge_list=gauge_list)
        river.set_folder_directory('./flow/')


        if original_data:
            
            # flood event 1: 2010-01-13 ~ 2010-01-20
            river.make_bdy(sim_start='2010-01-13', sim_end='2010-01-20', cell_width=30)
            
            # flood event 2: 2011-10-21 ~ 2011-10-28
            river.make_bdy(sim_start='2011-10-21', sim_end='2011-10-28', cell_width=30)
            
            # river.make_bdy(sim_start='2014-11-12', sim_end='2014-11-19', cell_width=30)
            
            # river.make_bdy(sim_start='2018-03-12', sim_end='2018-03-19', cell_width=30)
        
        if synthetic_hydrograph:

            # flood event 3: 2014-11-12 ~ 2014-11-19
            river.make_bdy(sim_start='2014-11-12', sim_end='2014-11-19', cell_width=30, 
                is_synthetic_hydrograph=synthetic_hydrograph, scaling_factor=2)
            river.make_bdy(sim_start='2014-11-12', sim_end='2014-11-19', cell_width=30, 
                is_synthetic_hydrograph=synthetic_hydrograph, scaling_factor=3)
            
             # flood event 4: 2018-03-12 ~ 2018-03-19
            river.make_bdy(sim_start='2018-03-12', sim_end='2018-03-19', cell_width=30, 
                is_synthetic_hydrograph=synthetic_hydrograph, scaling_factor=2)
            river.make_bdy(sim_start='2018-03-12', sim_end='2018-03-19', cell_width=30, 
                is_synthetic_hydrograph=synthetic_hydrograph, scaling_factor=3)
            
if __name__ == "__main__":

    main(FRANKFORT=True,
         WS_BR=True,
         flow=True,
         water_level=False,
        make_bdy=True,
        plot_data=True,
        original_data=True,
        synthetic_hydrograph=True)




