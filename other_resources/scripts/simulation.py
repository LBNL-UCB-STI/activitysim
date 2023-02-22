# ActivitySim
# See full license in LICENSE.txt.

import logging
import sys

from activitysim.core import chunk, config, mem, mp_tasks, tracing, workflow

# from activitysim import abm


logger = logging.getLogger("activitysim")


def cleanup_output_files(whale: workflow.Whale):

    active_log_files = [
        h.baseFilename
        for h in logger.root.handlers
        if isinstance(h, logging.FileHandler)
    ]
    whale.tracing.delete_output_files("log", ignore=active_log_files)

    whale.tracing.delete_output_files("h5")
    whale.tracing.delete_output_files("csv")
    whale.tracing.delete_output_files("txt")
    whale.tracing.delete_output_files("yaml")
    whale.tracing.delete_output_files("prof")
    whale.tracing.delete_output_files("omx")


def run(run_list, injectables=None):

    if run_list["multiprocess"]:
        logger.info("run multiprocess simulation")
        mp_tasks.run_multiprocess(run_list, injectables)
    else:
        logger.info("run single process simulation")
        pipeline.run(models=run_list["models"], resume_after=run_list["resume_after"])
        pipeline.checkpoint.close_store()
        mem.log_global_hwm()


def log_settings(injectables):

    settings = [
        "households_sample_size",
        "chunk_size",
        "multiprocess",
        "num_processes",
        "resume_after",
    ]

    for k in settings:
        logger.info("setting %s: %s" % (k, config.setting(k)))

    for k in injectables:
        logger.info("injectable %s: %s" % (k, inject.get_injectable(k)))


if __name__ == "__main__":

    whale.add_injectable("data_dir", "data")
    whale.add_injectable("configs_dir", "configs")

    from activitysim.cli.run import handle_standard_args

    handle_standard_args(whale, None)

    config.filter_warnings()
    whale.config_logger()

    # log_settings(injectables)

    t0 = tracing.print_elapsed_time()

    # cleanup if not resuming
    if not whale.settings.resume_after:
        cleanup_output_files(whale)

    run_list = mp_tasks.get_run_list(whale)

    if run_list["multiprocess"]:
        # do this after config.handle_standard_args, as command line args may override injectables
        injectables = list(
            set(injectables) | set(["data_dir", "configs_dir", "output_dir"])
        )
        injectables = {k: inject.get_injectable(k) for k in injectables}
    else:
        injectables = None

    run(run_list, injectables)

    # pipeline will be close if multiprocessing
    # if you want access to tables, BE SURE TO OPEN WITH '_' or all tables will be reinitialized
    # pipeline.open_pipeline('_')

    t0 = tracing.print_elapsed_time("everything", t0)
