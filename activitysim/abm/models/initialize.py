# ActivitySim
# See full license in LICENSE.txt.
import logging
import warnings
import os
import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline

from activitysim.core.steps.output import write_data_dictionary
from activitysim.core.steps.output import write_tables
from activitysim.core.steps.output import track_skim_usage

from .util import expressions

from activitysim.abm.tables import shadow_pricing


# We are using the naming conventions in the mtc_asim.h5 example
# file for our default list. This provides backwards compatibility
# with previous versions of ActivitySim in which only 'input_store'
# is given in the settings file.
DEFAULT_TABLE_LIST = [
    {'tablename': 'households',
     'h5_tablename': 'households',
     'index_col': 'household_id'},
    {'tablename': 'persons',
     'h5_tablename': 'persons',
     'index_col': 'person_id'},
    {'tablename': 'land_use',
     'h5_tablename': 'land_use_taz',
     'index_col': 'TAZ'}
]

logger = logging.getLogger(__name__)


def annotate_tables(model_settings, trace_label):

    annotate_tables = model_settings.get('annotate_tables', [])

    if not annotate_tables:
        logger.warning("annotate_tables setting is empty - nothing to do!")

    t0 = tracing.print_elapsed_time()

    for table_info in annotate_tables:

        tablename = table_info['tablename']
        df = inject.get_table(tablename).to_frame()

        # - rename columns
        column_map = table_info.get('column_map', None)
        if column_map:

            warnings.warn("annotate_tables option 'column_map' renamed 'rename_columns' and moved"
                          "to settings.yaml. Support for 'column_map' in annotate_tables will be "
                          "removed in future versions.",
                          FutureWarning)

            logger.info("renaming %s columns %s" % (tablename, column_map,))
            df.rename(columns=column_map, inplace=True)

        # - annotate
        annotate = table_info.get('annotate', None)
        if annotate:
            logger.info("annotated %s SPEC %s" % (tablename, annotate['SPEC'],))
            expressions.assign_columns(
                df=df,
                model_settings=annotate,
                trace_label=trace_label)

        # fixme - narrow?

        # - write table to pipeline
        pipeline.replace_table(tablename, df)


@inject.step()
def initialize_landuse():

    trace_label = 'initialize_landuse'

    model_settings = config.read_model_settings('initialize_landuse.yaml', mandatory=True)

    beam_geometries_path = config.setting('beam_geometries')
    data_file_path = config.data_file_path(beam_geometries_path, mandatory=True)

    beam_geom_dataframe = pd.read_csv(data_file_path)
    pipeline.rewrap("beam_geoms", beam_geom_dataframe)

    annotate_tables(model_settings, trace_label)

    # create accessibility (only required if multiprocessing wants to slice accessibility)
    land_use = pipeline.get_table('land_use')
    accessibility_df = pd.DataFrame(index=land_use.index)
    pipeline.replace_table("accessibility", accessibility_df)


@inject.step()
def initialize_households():

    trace_label = 'initialize_households'

    model_settings = config.read_model_settings('initialize_households.yaml', mandatory=True)
    annotate_tables(model_settings, trace_label)

    # - initialize shadow_pricing size tables after annotating household and person tables
    # since these are scaled to model size, they have to be created while single-process
    shadow_pricing.add_size_tables()

    # - preload person_windows
    t0 = tracing.print_elapsed_time()
    inject.get_table('person_windows').to_frame()
    t0 = tracing.print_elapsed_time("preload person_windows", t0, debug=True)


@inject.injectable(cache=True)
def preload_injectables():
    """
    preload bulky injectables up front - stuff that isn't inserted into the pipeline
    """

    logger.info("preload_injectables")

    inject.add_step('track_skim_usage', track_skim_usage)
    inject.add_step('write_data_dictionary', write_data_dictionary)
    inject.add_step('write_tables', write_tables)

    table_list = config.setting('input_table_list')

    # default ActivitySim table names and indices
    if table_list is None:
        logger.warning(
            "No 'input_table_list' found in settings. This will be a "
            "required setting in upcoming versions of ActivitySim.")

        new_settings = inject.get_injectable('settings')
        new_settings['input_table_list'] = DEFAULT_TABLE_LIST
        inject.add_injectable('settings', new_settings)

    # FIXME undocumented feature
    if config.setting('write_raw_tables'):

        # write raw input tables as csv (before annotation)
        csv_dir = config.output_file_path('raw_tables')
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)  # make directory if needed

        table_names = [t['tablename'] for t in table_list]
        for t in table_names:
            df = inject.get_table(t).to_frame()
            if t == 'households':
                df.drop(columns='chunk_id', inplace=True)
            df.to_csv(os.path.join(csv_dir, '%s.csv' % t), index=True)

    t0 = tracing.print_elapsed_time()

    # FIXME - still want to do this?
    # if inject.get_injectable('skim_dict', None) is not None:
    #     t0 = tracing.print_elapsed_time("preload skim_dict", t0, debug=True)
    #
    # if inject.get_injectable('skim_stack', None) is not None:
    #     t0 = tracing.print_elapsed_time("preload skim_stack", t0, debug=True)

    return True
