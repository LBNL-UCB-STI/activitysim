import os
import sys
import yaml
import time 
import numpy as np
import pandas as pd
import openmatrix as omx
from shutil import copytree, rmtree
from activitysim.core import config
from results import get_scenario_resutls, plot_results


import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger(__name__)


##############################
### RUN BASE YEAR SCENARIO ###
##############################

# Run activitySim 

#################################
## RUN POLICES WITH SCENARIOS ###
#################################

## General Guidelines
#1. Create directory and copy needed information
#2. Modify whatever input needed 
#3. Run scenarios with input modfies

RESULTS_DIR = 'policy_analysis'

def formatted_print(string, width=50, fill_char='#'):
    print('\n')
    if len(string) + 2 > width:
        width = len(string) + 4
    string = string.upper()
    print(fill_char * width)
    print('{:#^{width}}'.format(' ' + string + ' ', width=width))
    print(fill_char * width, '\n')
    
def run_time(func):
    start = time.time
    func
    end = time.time
    logging.info(('Simulation Finished. Total time: {:. 2f} minutes').format((end - start)/60))

def copy_directory(source, destination, replace = False):
    """ Creates a copy of source (directory) in destination (another directory):
    Parameters
    -----------
    - source: directory path
    - destiantion: directory path 
    
    Returns 
    None 
    """
    try:
        logging.info('Creating directory {} '.format(destination))
        copytree(source, destination)
    except FileExistsError:
        if replace:
            logging.info('Directory {} already exist - Replacing file'.format(destination))
            rmtree(destination)
            copytree(source, destination)
        else:
            logging.info('Directory {} already exist - Skip'.format(destination))
        
def overrride_settings(settings_path, new_settings_dict):
    ''' Modifies the settings.yaml file in the configs files'''
    
    #Reads settings file
    logging.debug('Modifying settings.yaml file')
    a_yaml_file = open(settings_path)
    settings = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    
    #Modify Settings
    for key, value in new_settings_dict.items():
        settings[key] = value
    
    #Dumps modified settings
    with open(settings_path, 'w') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False)

def read_policy_settings():
    a_yaml_file = open('policies.yaml')
    settings = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    return settings

def input_transformation(policy_name, scenario, settings):
    logging.info('Input modification to run scenario')
    configs_chages = ['transit_frequency','transit_operations', 'tolls', 'hov_lanes','transit_fare']
    constant_chages = ['shared_tnc_price','shared_tnc_waiting_times', 'VMT_fees']
    
    if policy_name == 'transit_fare_nothing_here':
        input_transit_reduction(policy_name, scenario, settings)
        
    elif policy_name in configs_chages:
        input_chage_configs(policy_name, scenario, settings)
    
    elif policy_name in constant_chages:
        input_chage_constant(policy_name, scenario, settings)
        
#     elif policy_name == 'transit_frequency':
#         input_chage_configs(policy_name, scenario, settings)
#     elif policy_name == 'transit_operations':
#         input_transit_operations(policy_name, scenario, settings)
#     elif policy_name == 'shared_tnc_price':
#         input_shared_tnc_price(policy_name, scenario, settings)
#     elif policy_name == 'shared_tnc_waiting_times':
#         input_shared_tnc_waiting_times(policy_name, scenario, settings)
#     elif policy_name == 'tolls':
#         input_tolls(policy_name, scenario, settings)
#     elif policy_name == 'VMT_fees':
#         input_VMT_fees(policy_name, scenario, settings)
#     elif policy_name == 'hov_lanes':
#         input_hov_lanes(policy_name, scenario, settings)

    else:   
        logging.info('Policy {} cannot be run at this time'.format(policy_name))

def run_scenario(policy_name, settings):
    """ Runs scenarios for a given policy"""
    
    scenarios = settings['policies'][policy_name]['scenarios']
    replace = settings['policies'][policy_name]['replace']
    policy_folder_path = os.path.join('policy_analysis', policy_name)
    os.chdir(policy_folder_path)
    
    for scenario in scenarios.keys():
        print(os.getcwd())
        start = time.time()
        formatted_print(('Simulation for Policy {} - {}').format(policy_name, scenario))
        logging.info(('Simulation for Policy {} - {}').format(policy_name, scenario))
        path = os.path.join(policy_folder_path, scenario)
        copy_directory('../../bay_area_base', scenario, replace)
        os.chdir(scenario)
#         input_transformation(policy_name, scenario, settings)
#         overrride_settings('configs/settings.yaml', settings['asim_settings'])
#         os.system('python simulation.py')
#         os.system('rm -r output/*.h5')
#         os.system('rm -r output/*.log')
#         os.system('rm -r output/trip_mode_choice/*.csv')
#         os.system('rm -r ~/.local/share/Trash/files/*')#Delete files in Trash to liberate space. 
#         results = get_scenario_resutls(policy_name, scenario, settings)
#         save_yaml('results.yaml', results)
    
        os.chdir('../')
        end = time.time()
        running_time_message = "Simulation Finished. Total time: {:.2f} minutes".format((end - start)/60)
        logging.info(running_time_message)
    
    ## FIX ME: Plot results
    plot_results(policy_name, settings)
    os.chdir('../../')
        
##########################################
## Public transportation fare reduction ##

def transit_fare_reduction(skims_path, reduction, modes = None):
    ''' 
    Reduces transit fare by reduction 
    Parameters
    -----------
    - skims: skims file or path 
    - reduction: float. Number between 0 and 1
    - modes: str. Can be "bus", 'rail', 'lightrail' ## To add functionality
    '''
    assert 0<= reduction <=1, 'reduction parameter should be between zero and one'
    logging.info('Modifying transit skims for reduction{:.0f}'.format(reduction*100))
    
    skims = omx.open_file(skims_path,'a')
    skims_modified = omx.open_file(skims_path[:-4] + '_transit_fare_reduction.omx','w')

    matrices = skims.list_matrices()
    for matrix in matrices: 
        try:
            if 'FAR' in(matrix):
                skims_modified[matrix] = np.array(skims[matrix])*(1-reduction)
            else:
                skims_modified[matrix] = np.array(skims[matrix])
        except NodeError:
            pass 
    skims.close()
    skims_modified.close()
    
def input_transit_reduction(policy_name, scenario, settings):
    logging.info('Creating transit skims for {}'.format(scenario))
    
    file_exist = os.path.exists('data/skims_transit_fare_reduction.omx')
    
    if file_exist:
        logging.info('Modified transit skims for {} already exist - Skip'.format(scenario))
    else:
        reduction = settings['policies'][policy_name]['scenarios'][scenario]
        skim_path = os.path.join('data','skims.omx')
        settings_path = os.path.join('configs','settings.yaml')
        
        transit_fare_reduction(skim_path, reduction, modes = None)
        overrride_settings(settings_path, {'skims_file': 'skims_transit_fare_reduction.omx'})
        overrride_settings(settings_path, settings['asim_settings'])

def read_yaml(path):
#     logging.debug('Modifying settings.yaml file')
    a_yaml_file = open(path)
    return yaml.load(a_yaml_file, Loader=yaml.FullLoader)

def save_yaml(path, settings):
    with open(path, 'w') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False)

# def read_mode_choice_configs(policy_name, scenario):
#     configs_path = os.path.join('policy_analysis', policy_name, scenario, 'configs/configs')
#     tour = pd.read_csv(configs_path + '/tour_mode_choice.csv', index_col = 'Label')
#     trip = pd.read_csv(configs_path + '/trip_mode_choice.csv')
#     return tour, trip

# def read_mode_choice_constants(policy, scenario):
#     configs_path = os.path.join('policy_analysis', policy_name, scenario, 'configs/configs')
#     tour = read_yaml(configs_path + '/tour_mode_choice.yaml')
#     trip = read_yaml(configs_path + '/trip_mode_choice.yaml')
#     return tour, trip

# def save_mode_choice_configs(policy_name, scenario, tour, trip):
#     configs_path = os.path.join('policy_analysis', policy_name, scenario, 'configs/configs')
#     tour.to_csv(configs_path + '/tour_mode_choice.csv')
#     trips.to_csv(configs_path +'/trip_mode_choice.csv')

def add_factor(iterable, factor):
    modified = [iterable[i] + ']*' + str(factor) for i in range(len(iterable) - 1)]
    return "".join(modified + [iterable[-1]])

def add_number(iterable, number):
    modified = [iterable[i] + ']+' + str(number) for i in range(len(iterable) - 1)]
    return "".join(modified + [iterable[-1]])

def modify_mode_choice_configs(tour, trip, regex_expression, factor, change_type):
    tour_to_modify = tour[tour.Description.str.contains(regex_expression, regex = True).fillna(False)].Expression
    trip_to_modify = trip[trip.Description.str.contains(regex_expression, regex = True).fillna(False)].Expression
    
    if change_type == 'product':
        tour_modified = tour_to_modify.str.split(']').apply(lambda s: add_factor(s, factor))
        trip_modified = trip_to_modify.str.split(']').apply(lambda s: add_factor(s, factor))
    elif change_type == 'summation':
        tour_modified = tour_to_modify.str.split(']').apply(lambda s: add_number(s, factor))
        trip_modified = trip_to_modify.str.split(']').apply(lambda s: add_number(s, factor))
    
    tour.loc[tour_to_modify.index, 'Expression'] = tour_modified
    trip.loc[trip_to_modify.index, 'Expression'] = trip_modified
    return tour, trip
  
def modify_mode_choice_constants(dict_settings, factor, variables, change_type):
    constants = dict_settings['CONSTANTS']
    
    for value in variables:
        if type(constants[value]) is float:
            if change_type == 'product':
                constants[value] *= factor
            elif change_type == 'summation':
                constants[value] += factor
        
        elif type(constants[value]) is dict:
            if change_type == 'product':
                my_dict = constants[value].update((x, y*factor) for x, y in constants[value].items())
            elif change_type == 'summation':
                my_dict = constants[value].update((x, y+factor) for x, y in constants[value].items())

    return dict_settings

def input_chage_configs(policy_name, scenario, settings):
    #Read config files
    configs_path = os.path.join('configs/configs') #'policy_analysis', policy_name, scenario, 
    tour = pd.read_csv(configs_path + '/tour_mode_choice.csv', index_col = 'Label')
    trip = pd.read_csv(configs_path + '/trip_mode_choice.csv')
    
    #Changes variables
    regex_expression = settings['policies'][policy_name]['regex']
    change_type = settings['policies'][policy_name]['chage_type']
    factor = settings['policies'][policy_name]['scenarios'][scenario]
    tour, trip = modify_mode_choice_configs(tour, trip, regex_expression, factor, change_type)
    
    #Save results 
    tour.to_csv(configs_path + '/tour_mode_choice.csv')
    trip.to_csv(configs_path + '/trip_mode_choice.csv', index = False)
    
def input_chage_constant(policy_name, scenario, settings):
    #Read YAML files
    path = os.path.join('configs/configs')
    tour = read_yaml(path + '/tour_mode_choice.yaml')
    trip = read_yaml(path + '/trip_mode_choice.yaml')
    
    #Change variables
    change_type = settings['policies'][policy_name]['chage_type']
    factor = settings['policies'][policy_name]['scenarios'][scenario]
    variables = settings['policies'][policy_name]['variables']
    tour_settings = modify_mode_choice_constants(tour, factor, variables, change_type)
    trip_settings = modify_mode_choice_constants(trip, factor, variables, change_type)
    
    #Save New YAML files
    save_yaml(path + '/tour_mode_choice.yaml', tour)
    save_yaml(path + '/trip_mode_choice.yaml', trip)
    
        
settings = read_policy_settings()
[run_scenario(policy_name, settings) for policy_name in settings['policies'].keys()]

#########################
### SUMARIZE RESULTS ###
#########################

#Save plot s