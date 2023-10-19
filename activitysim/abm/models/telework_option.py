
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

import logging

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject
from activitysim.core import logit

logger = logging.getLogger(__name__)

#######################
### Helper Functions ##
#######################

def find_index(array_, value):
    """
    Returns the index where value is first found in array. If value is not found, returns NaN
    
    Parameters:
    ------------
    - array: n-dimensional array. Array of shape (n,m)
    - value: 1d-array of shape (m,)
    """
    
    not_found = True
    i = 0
    while not_found:
        try: 
            comparison = array_[i,:] == value
        except IndexError:
            comparison  = False
            return np.nan
        
        if comparison.all():
            not_found = False
        else:
            i += 1
    return i

def find_rate(rates, category):
    "The df has the categories, and find the category combination in array and returns its index"

    index = []
    for cat in np.array(rates.drop(columns = 'rate')):
        i = find_index(category, cat)
        index.append(i)
    return index

def annotate(df, annotation):
    """ Annotates a dataframe with annotation
    Parameters: 
    ------------
    - df: Pandas DataFrame. Dataframe that reflects the annotation. 
    - annotation: Pandas DataFrame. DataFrame with Expressions to annotate in Dataframe. 
        This dataframe should have at least two columns: 
        - Target: str.  Name of the new column to annotate. 
        - Expression: str. Expression to evaluate with python eval. 
    
    Return: 
    --------
    Annotated dataFrame
    """
    for index, row in annotation.iterrows():
        default_local_dict = {'pd':pd, 'np': np, 'df':df}
        name = row['Target']
        expression = row['Expression']
        df[name] = eval(expression, {}, default_local_dict)
    return df

def create_dict_rate(rates, category):
    rates_copy = rates.copy(deep = True)

    corresponding_category = find_rate(rates_copy, category)
    rates_copy['category'] = corresponding_category
    
    return rates_copy.dropna().set_index('category')['rate'].to_dict()


@inject.step()
def telework_option(
        persons_merged, persons, households,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id, locutor):
    
    """
    Rate-base telework as an option model. 

    Returns:
    ---------
    Simulation result of telework as an option. 
    """

    trace_label = 'telework_option'

    #Read Files
    model_settings = config.read_model_settings('telework_option.yaml')
    annotate_path = config.config_file_path(model_settings['annotation_file'])
    rates_path = config.config_file_path(model_settings['rates_file'])

    # telework_option_anotate = pd.read_csv('annotate_telework_option.csv', comment = "#" )
    telework_option_anotate = pd.read_csv(annotate_path, comment='#')
    telework_option_rates = pd.read_csv(rates_path)

    #Choosers
    choosers = persons_merged.to_frame()
    choosers = choosers[choosers.ptype.isin([1,2])] # Choosers are full- or part-time workers only
    choosers = annotate(choosers, telework_option_anotate)
    
    logger.info("Running %s with %d persons", trace_label, len(choosers))


    # Preprocessing: Add rates to choosers. 
    telework_rate_categories = list(telework_option_rates.columns[:-1])
    category, category_index = np.unique(choosers[telework_rate_categories].astype(int).to_numpy(), axis=0, return_inverse=True)
    choosers['telework_option_category'] = category_index
    dict_cat_rate = create_dict_rate(telework_option_rates, category) #Dict categories and rates
    choosers['telework_option_rate'] = choosers.telework_option_category.replace(dict_cat_rate)

    # Simulation
    probs = choosers[['telework_option_rate']]
    probs.insert(0,'0', 1 - probs.telework_option_rate)
    choices, rands = logit.make_choices(probs, trace_label=trace_label)

    # Simulation Result. Who telecommutes today. 
    persons = persons.to_frame()
    persons['telework_option'] = choices.reindex(persons.index).fillna(0).astype(bool)
    pipeline.replace_table("persons", persons)
    tracing.print_summary('telework_option', persons.telework_option, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(persons,
                         label=trace_label,
                         warn_if_empty=True)