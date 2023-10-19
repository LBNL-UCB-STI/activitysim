# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 
import numpy as np 
from scipy.stats import norm

def bootsratp(df, choice, category = None,n = 1000, confidence_level = 0.95, func = 'value_counts'):
    """ Bootstrapping proceidure to estimate the confidence intervals for rates
    
    Parameters:
    ------------
    - df: Pandas dataframe 
    - category: 'str' or list. Category to estimate the confidence interval. 
    - choice: 'str'. Variable to estimate the rate
    - n: int. Number of ireration in bootstrapp procedure
    - confidence_level: float. A number for 0 to 1
    
    return:
    -----------
    Dataframe with lower and upper bound at the given confidence interval
    """

    # assert category in df.columns
    assert choice in df.columns
    assert confidence_level <= 1.0
    assert confidence_level >= 0.0
    assert isinstance(n, int)

    # df = df[category + [choice]].copy()
    df['count_col'] = 1 
    r = len(df)
    alpha = 1 - confidence_level

    pcts = []
    if category is None:

        if func == 'value_counts':
            rates = df[choice].value_counts(normalize = True)
            count = df[choice].value_counts()
            for _ in range(n):
                sample = df.sample(r, replace = True)
                group_pcts = sample[choice].value_counts(normalize = True)
                pcts.append(group_pcts)
        elif func == 'mean':
            rates = df[choice].mean()
            for _ in range(n):
                sample = df.sample(r, replace = True)
                group_pcts = df.sample(r, replace = True)[choice].mean()
                pcts.append(group_pcts)
    else:
        # df = df[category + [choice]].copy()
        # df['count_col'] = 1 
        # assert 'count_col' in df.columns
        if func == 'value_counts':
            rates = df.groupby(category)[choice].value_counts(normalize=True)
            count = df.groupby(category)[choice].value_counts()
            for _ in range(n):
                group_pcts = df.sample(r, replace = True).groupby(category)[choice].value_counts(normalize=True)
                pcts.append(group_pcts)
        elif func == 'mean':
            rates = df.groupby(category)[choice].mean()
            rates = df.groupby(category).agg({choice:'mean', 'count_col':'count'})
            for _ in range(n):
                group_pcts = df.sample(r, replace = True).groupby(category)[choice].mean()
                pcts.append(group_pcts)

    pcts = pd.concat(pcts, axis = 1)
    pcts['mean'] , pcts['std']= pcts.mean(axis = 1), pcts.std(axis = 1)

    if func == 'mean':
        pcts['count'] = rates['count_col']

    if func == 'value_counts':
        pcts['count'] = count
    
    # pcts[['mean','std']] = mean , std
    pcts['lower_bound'] = norm.ppf((alpha/2), pcts['mean'], pcts['std'])
    pcts['upper_bound'] = norm.ppf((1 - alpha/2), pcts['mean'], pcts['std'])

    # lower_bound = pcts.quantile((alpha/2), axis = 1).rename('lower_bound')
    # upper_bound = pcts.quantile((1 - alpha/2), axis = 1).rename('upper_bound')

    # ci = pd.concat((lower_bound, upper_bound), axis = 1)

    # rates_final = pd.concat((rates, ci), axis = 1)
    # rates_final
    
    return pcts[['mean', 'count','lower_bound', 'upper_bound']]


# %%
# size = 1000
# df = pd.DataFrame({'a': np.random.randint(1,5, size = size), 'b': np.random.randint(1,4, size = size)})
# bootsratp(df, 'a', 'b', n = 1000, confidence_level = 0.95)