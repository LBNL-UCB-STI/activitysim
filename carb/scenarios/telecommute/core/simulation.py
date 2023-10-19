# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging
import argparse
import os

from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import mp_tasks
from activitysim.core import chunk

logger = logging.getLogger('activitysim')


def cleanup_output_files():

    active_log_files = \
        [h.baseFilename for h in logger.root.handlers if isinstance(
            h, logging.FileHandler)]
    tracing.delete_output_files('log', ignore=active_log_files)

    tracing.delete_output_files('h5')
    tracing.delete_output_files('csv')
    tracing.delete_output_files('txt')
    tracing.delete_output_files('yaml')
    tracing.delete_output_files('prof')


def run(run_list, injectables=None, warm_start=False):

    warm_start_steps = [
        'school_location', 'workplace_location', 'auto_ownership_simulate']

    if run_list['multiprocess']:
        if warm_start:
            run_list['multiprocess_steps'][1].update(
                {'models': warm_start_steps})
            run_list['multiprocess_steps'][2].update(
                {'begin': 'write_tables', 'models': ['write_tables']})
            logger.info("run multiprocess warm start simulation")
        else:
            logger.info("run multiprocess simulation")
        mp_tasks.run_multiprocess(run_list, injectables)

    else:
        if warm_start:
            last_step_index = run_list['models'].index(warm_start_steps[-1])
            init_and_warm_start_steps = run_list['models'][:last_step_index]
            all_warm_start_steps = init_and_warm_start_steps + ['write_tables']
            run_list.update({'models': all_warm_start_steps})
            logger.info("run single process warm start simulation")
        else:
            logger.info("run single process simulation")
        pipeline.run(
            models=run_list['models'], resume_after=run_list['resume_after'])
        pipeline.close_pipeline()
        chunk.log_write_hwm()


def log_settings():

    settings = [
        'households_sample_size',
        'chunk_size',
        'multiprocess',
        'num_processes',
        'resume_after',
    ]

    for k in settings:
        logger.info("setting %s: %s" % (k, config.setting(k)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-w", "--warm_start", action="store_true",
        help="only run mandatory location choice models")
    parser.add_argument(
        "-h", "--household_sample_size", action="store",
        help="household sample size")
    parser.add_argument(
        "-n", "--num_processes", action="store",
        help="# of multiprocess workers to use")
    parser.add_argument(
        "-c", "--chunk_size", action="store",
        help="batch size for processing choosers")
    parser.add_argument(
        "-r", "--resume_after", action="store",
        help="re-run activitysim starting after specified model step.")
    parser.add_argument(
        "-k", "--skim_cache", action="store_true",
        help="use skim cache. default is False.")

    args = parser.parse_args()

    warm_start = args.warm_start
    if args.warm_start:
        output_tables = config.setting('output_tables')
        output_tables['prefix'] = 'warm_start_'
        output_tables['tables'] = ['households', 'persons']
        config.override_setting('output_tables', output_tables)

    if args.household_sample_size:
        config.override_setting(
            'households_sample_size', int(args.household_sample_size))

    if args.num_processes:
        config.override_setting('num_processes', int(args.num_processes))

    if args.num_processes:
        config.override_setting('chunk_size', int(args.chunk_size))

    if args.resume_after:
        config.override_setting('resume_after', args.resume_after)

    config.override_setting('read_skim_cache', args.skim_cache)

    injectables = ['data_dir', 'configs_dir', 'output_dir']
    inject.add_injectable('data_dir', 'data')
    inject.add_injectable('configs_dir', ['configs', 'configs/configs'])

    config.filter_warnings()
    tracing.config_logger()

    log_settings()

    t0 = tracing.print_elapsed_time()

    # cleanup if not resuming
    if not config.setting('resume_after', False):
        cleanup_output_files()

    run_list = mp_tasks.get_run_list()

    if run_list['multiprocess']:
        injectables = {k: inject.get_injectable(k) for k in injectables}
    else:
        injectables = None

    os.environ['MKL_NUM_THREADS'] = '1'

    run(run_list, injectables, warm_start=warm_start)

    # pipeline will be closed after run if multiprocessing
    # if you want access to tables, BE SURE TO OPEN
    # WITH '_' or all tables will be reinitialized (deleted)
    # pipeline.open_pipeline('_')

    t0 = tracing.print_elapsed_time("everything", t0)

    # make sure output data has same permissions as containing
    # directory (should only be an issue when running from inside
    # docker which will execute this script as root)
    output_dir_full_path = os.path.abspath('./output/')
    data_stats = os.stat(output_dir_full_path)
    uid = data_stats.st_uid
    gid = data_stats.st_gid
    for dirpath, dirnames, filenames in os.walk(output_dir_full_path):
        for fname in filenames:
            os.chown(os.path.join(dirpath, fname), uid, gid)
