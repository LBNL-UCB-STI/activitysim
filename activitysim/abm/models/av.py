import pandas as pd
import numpy as np
import logging

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import inject

from activitysim.core.util import reindex

logger = logging.getLogger(__name__)

@inject.step()
def autonomous_vehicles(persons_merged, persons, households,
       skim_dict, skim_stack,
       chunk_size, trace_hh_id, locutor):
    
    """
    Rate-base model for Autonomous vehicles Ownership. 

    Returns:
    ---------
    Simulated Autonomous Vehicle ownership
    """
    trace_label = 'autonomous_vehicle'
    
    #Settings
    model_settings = config.read_model_settings('av.yaml')
    av_rates_path = config.config_file_path(model_settings['av_rates'])
    av_rates = pd.read_csv(av_rates_path, index_col = 'year')
    year = inject.get_injectable('year')
    scenario = inject.get_injectable('scenario')
    
    #Choosers
    households = households.to_frame()
    persons = persons.to_frame()
#     persons_merged = persons_merged.to_frame()
    
    try: 
        rate = av_rates.loc[year, scenario]
        vot_reduction = av_rates.loc['vot_reduction',scenario]
    except KeyError:
        rate = 0
        vot_reduction = 0
        
    logger.info('Simulation year: {}'.format(year))
    logger.info('Simulation scenario: {}.'.format(scenario))
    logger.info('Autonomous Vehicles penetration rate:{}'.format(rate))
    logger.info('Value of time Reduction :{}'.format(vot_reduction))
    
    av = np.random.choice(np.array([0,1]), size = len(households), p = np.array([1 - rate, rate]))
    
    households['av'] = av
    households['hh_value_of_time'] = households['hh_value_of_time'].mask(households.av.astype(bool), 
                                                                         households['hh_value_of_time']*(1 - vot_reduction))
    persons['_hh_vot'] = reindex(households.hh_value_of_time, persons.household_id)
    persons['value_of_time'] = persons['_hh_vot'].where(persons.age>=18, persons._hh_vot * 0.667)
#     persons_merge['value_of_time'] = persons.value_of_time
    
    pipeline.replace_table("persons", persons)
    pipeline.replace_table("households", households)
    
    if trace_hh_id:
        tracing.trace_df(persons,
                         label=trace_label,
                         warn_if_empty=True)