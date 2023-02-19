import logging
import multiprocessing
import time
from typing import Callable, Iterable

from activitysim.core.exceptions import DuplicateWorkflowNameError
from activitysim.core.workflow.accessor import FromWhale, WhaleAccessor
from activitysim.core.workflow.checkpoint import (
    CHECKPOINT_NAME,
    FINAL_CHECKPOINT_NAME,
    LAST_CHECKPOINT,
)
from activitysim.core.workflow.steps import run_named_step

# single character prefix for run_list model name to indicate that no checkpoint should be saved
NO_CHECKPOINT_PREFIX = "_"


logger = logging.getLogger(__name__)


def split_arg(s, sep, default=""):
    """
    split str s in two at first sep, returning empty string as second result if no sep
    """
    r = s.split(sep, 2)
    r = list(map(str.strip, r))

    arg = r[0]

    if len(r) == 1:
        val = default
    else:
        val = r[1]
        val = {"true": True, "false": False}.get(val.lower(), val)

    return arg, val


class Runner(WhaleAccessor):
    """
    This accessor provides the tools to actually run ActivitySim workflow steps.
    """

    def __call__(self, models, resume_after=None, memory_sidecar_process=None):
        """
        run the specified list of models, optionally loading checkpoint and resuming after specified
        checkpoint.

        Since we use model_name as checkpoint name, the same model may not be run more than once.

        If resume_after checkpoint is specified and a model with that name appears in the models list,
        then we only run the models after that point in the list. This allows the user always to pass
        the same list of models, but specify a resume_after point if desired.

        Parameters
        ----------
        models : list[str] or Callable
            A list of the model names to run, which should all have been
            registered with the @workflow.step decorator.  Alternative, give
            a single function that is or could have been so-decorated.
        resume_after : str or None
            model_name of checkpoint to load checkpoint and AFTER WHICH to resume model run
        memory_sidecar_process : MemorySidecar, optional
            Subprocess that monitors memory usage

        returns:
            nothing, but with pipeline open
        """
        if isinstance(models, Callable) and models.__name__ is not None:
            if models is self.obj._RUNNABLE_STEPS.get(models.__name__, None):
                self([models.__name__], resume_after=None, memory_sidecar_process=None)
            elif models is self.obj._LOADABLE_OBJECTS.get(models.__name__, None):
                self.obj.set(models.__name__, self.obj.get(models.__name__))
            elif models is self.obj._LOADABLE_TABLES.get(models.__name__, None):
                self.obj.set(models.__name__, self.obj.get(models.__name__))
            else:
                raise DuplicateWorkflowNameError(models.__name__)
            return

        if isinstance(models, str):
            return self.by_name(models)

        from activitysim.core.tracing import print_elapsed_time

        t0 = print_elapsed_time()

        if resume_after:
            self.obj.checkpoint.restore(resume_after)
        t0 = print_elapsed_time("open_pipeline", t0)

        if resume_after == LAST_CHECKPOINT:
            resume_after = self.obj.checkpoint.last_checkpoint[CHECKPOINT_NAME]

        if resume_after:
            logger.info("resume_after %s" % resume_after)
            if resume_after in models:
                models = models[models.index(resume_after) + 1 :]

        self.obj.trace_memory_info("pipeline.run before preload_injectables")

        # preload any bulky injectables (e.g. skims) not in pipeline
        # if inject.get_injectable("preload_injectables", None):
        #     if memory_sidecar_process:
        #         memory_sidecar_process.set_event("preload_injectables")
        #     t0 = print_elapsed_time("preload_injectables", t0)

        self.obj.trace_memory_info("pipeline.run after preload_injectables")

        t0 = print_elapsed_time()
        for model in models:
            if memory_sidecar_process:
                memory_sidecar_process.set_event(model)
            t1 = print_elapsed_time()
            self.by_name(model)
            self.obj.trace_memory_info(f"pipeline.run after {model}")

            self.log_runtime(model_name=model, start_time=t1)

        if memory_sidecar_process:
            memory_sidecar_process.set_event("finalizing")

        # add checkpoint with final tables even if not intermediate checkpointing
        if not self.obj.should_save_checkpoint():
            self.obj.checkpoint.add(FINAL_CHECKPOINT_NAME)

        self.obj.trace_memory_info("pipeline.run after run_models")

        t0 = print_elapsed_time("run_model (%s models)" % len(models), t0)

        # don't close the pipeline, as the user may want to read intermediate results from the store

    def __dir__(self) -> Iterable[str]:
        return self.obj._RUNNABLE_STEPS.keys()

    def __getattr__(self, item):
        if item in self.obj._RUNNABLE_STEPS:
            # f = lambda **kwargs: self.obj._RUNNABLE_STEPS[item](
            #     self.obj.context, **kwargs
            # )
            f = lambda **kwargs: self.by_name(item)
            f.__doc__ = self.obj._RUNNABLE_STEPS[item].__doc__
            return f
        raise AttributeError(item)

    timing_notes: set[str] = FromWhale(default_init=True)

    def log_runtime(self, model_name, start_time=None, timing=None, force=False):

        assert (start_time or timing) and not (start_time and timing)

        timing = timing if timing else time.time() - start_time
        seconds = round(timing, 1)
        minutes = round(timing / 60, 1)

        process_name = multiprocessing.current_process().name

        if self.obj.settings.multiprocess and not force:
            # when benchmarking, log timing for each processes in its own log
            if self.obj.settings.benchmarking:
                header = "component_name,duration"
                with self.obj.filesystem.open_log_file(
                    f"timing_log.{process_name}.csv", "a", header
                ) as log_file:
                    print(f"{model_name},{timing}", file=log_file)
            # only continue to log runtime in global timing log for locutor
            if not self.obj.get_injectable("locutor", False):
                return

        header = "process_name,model_name,seconds,minutes,notes"
        note = " ".join(self.timing_notes)
        with self.obj.filesystem.open_log_file(
            "timing_log.csv", "a", header
        ) as log_file:
            print(
                f"{process_name},{model_name},{seconds},{minutes},{note}", file=log_file
            )

        self.timing_notes.clear()

    def _pre_run_step(self, model_name: str):
        if model_name in [
            checkpoint[CHECKPOINT_NAME]
            for checkpoint in self.obj.checkpoint.checkpoints
        ]:
            raise RuntimeError("Cannot run model '%s' more than once" % model_name)

        self.obj.rng().begin_step(model_name)

        # check for args
        if "." in model_name:
            step_name, arg_string = model_name.split(".", 1)
            args = dict(
                (k, v)
                for k, v in (
                    split_arg(item, "=", default=True) for item in arg_string.split(";")
                )
            )
        else:
            step_name = model_name
            args = {}

        # check for no_checkpoint prefix
        if step_name[0] == NO_CHECKPOINT_PREFIX:
            step_name = step_name[1:]
            checkpoint = False
        else:
            checkpoint = self.obj.should_save_checkpoint(model_name)

        self.obj.add_injectable("step_args", args)

        self.obj.trace_memory_info(f"pipeline.run_model {model_name} start")

        from activitysim.core.tracing import print_elapsed_time

        t0 = print_elapsed_time()
        logger.info(f"#run_model running step {step_name}")

        self.step_name = step_name
        self.checkpoint = checkpoint
        self.t0 = t0

    def by_name(self, model_name):
        """
        Run the specified model and add checkpoint for model_name

        Since we use model_name as checkpoint name, the same model may not be run more than once.

        Parameters
        ----------
        model_name : str
            model_name is assumed to be the name of a registered orca step
        """
        self._pre_run_step(model_name)

        instrument = self.obj.settings.instrument
        if instrument is not None:
            try:
                from pyinstrument import Profiler
            except ImportError:
                instrument = False
        if isinstance(instrument, (list, set, tuple)):
            if self.step_name not in instrument:
                instrument = False
            else:
                instrument = True

        if instrument:
            from pyinstrument import Profiler

            with Profiler() as profiler:
                self.obj.context = run_named_step(self.step_name, self.obj.context)
            out_file = self.obj.filesystem.get_profiling_file_path(
                f"{self.step_name}.html"
            )
            with open(out_file, "wt") as f:
                f.write(profiler.output_html())
        else:
            self.obj.context = run_named_step(self.step_name, self.obj.context)

        from activitysim.core.tracing import print_elapsed_time

        self.t0 = print_elapsed_time(
            "#run_model completed step '%s'" % model_name, self.t0, debug=True
        )
        self.obj.trace_memory_info(f"pipeline.run_model {model_name} finished")

        self.obj.add_injectable("step_args", None)

        self.obj.rng().end_step(model_name)
        if self.checkpoint:
            self.obj.checkpoint.add(model_name)
        else:
            logger.info(
                "##### skipping %s checkpoint for %s" % (self.step_name, model_name)
            )
