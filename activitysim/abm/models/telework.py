import pandas as pd
import numpy as np
import logging

pd.options.mode.chained_assignment = None  # default='warn'


from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject
from activitysim.core import logit

# from .util import expressions
# from .util import estimation

logger = logging.getLogger(__name__)

@inject.step()
def telework(
        persons_merged, persons, households,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id, locutor):
    
    """
    Rate-base telework as an option model. 

    Returns:
    ---------
    Simulation result of telework as an option. 
    """

    trace_label = 'telework'

    #Read Files
    model_settings = config.read_model_settings('telework.yaml')
    frequency_rates_path = config.config_file_path(model_settings['frequency_rates'])
    day_rates_path = config.config_file_path(model_settings['daily_rates'])

    # telework_option_anotate = pd.read_csv('annotate_telework_option.csv', comment = "#" )
    telework_frequency_rates = pd.read_csv(frequency_rates_path, comment='#')
    telework_daily_rates = pd.read_csv(day_rates_path, comment='#')

    #Choosers
    persons_merged = persons_merged.to_frame()
    choosers = persons_merged[persons_merged['telework_option']] #Only those who have telework as an option
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    # Simulation Telework Frequency
    frequency_probs = pd.concat([telework_frequency_rates] * len(choosers))
    frequency_probs.set_index(choosers.index, inplace=True)
    choices, rands = logit.make_choices(frequency_probs, trace_label='telework_frequencies')
    choosers['telework_frequency'] = choices

    # Simulation Telework daily
    prob_telecommute = telework_daily_rates['rate'].to_dict()
    choosers['telework_rate'] = choosers['telework_frequency'].replace(prob_telecommute)
    telework_probs = choosers[['telework_rate']]
    telework_probs.insert(0,'0', 1 - telework_probs.telework_rate)
    choices, rands = logit.make_choices(telework_probs, trace_label='telework_daily')
    
    persons = persons.to_frame()
    persons['telework'] = choices.reindex(persons.index).fillna(0).astype(bool)
    persons['ptype'] = persons['ptype'].mask(persons['telework'], 4)

    pipeline.replace_table("persons", persons)
    tracing.print_summary('telework', persons.telework, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(persons,
                         label=trace_label,
                         warn_if_empty=True)