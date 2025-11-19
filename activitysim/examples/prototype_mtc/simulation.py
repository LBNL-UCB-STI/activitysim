# ActivitySim
# See full license in LICENSE.txt.

from __future__ import annotations

import argparse
import sys

import dfols
import yaml
import numpy as np
from scipy.optimize import least_squares
import pandas as pd
import os

from activitysim import abm  # register injectables
from activitysim.cli.run import add_run_args, run

import logging

formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
# calibration_logger = None


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


counter = 0
_inferred_num_errors = 0  # Global to store inferred errors for log header/fallback


def calibrate(config_file, args, regularization=None):
    global calibration_logger
    print("Running calibration!!!")
    # global config # Declare intent to use the global config variable
    global _inferred_num_errors  # Declare intent to use the global error count

    try:
        with open(config_file, "r") as f:
            calibration_config = yaml.safe_load(f)  # Load config into a local variable
    except FileNotFoundError:
        print(
            f"Error: Calibration config file not found at {config_file}.",
            file=sys.stderr,
        )
        sys.exit(1)  # Exit if calibration config is missing
    except Exception as e:
        print(
            f"Error loading calibration config file {config_file}: {e}", file=sys.stderr
        )
        sys.exit(1)

    # --- Determine Paths from args and config ---
    # Use command-line args as primary source for base directories
    # FIX: Use args.output, args.config, and args.data based on run.py add_run_args
    output_dir = args.output
    # args.config is a list, take the first one as the primary config dir base
    configs_base_dir = args.config[0] if args.config else None
    data_base_dir = args.data[0] if args.data else None  # args.data is also a list

    # Get filenames/relative paths from calibration config
    main_settings_file_name = calibration_config.get("main_settings_file")
    calibration_data_file_name = calibration_config.get(
        "calibration_data_file",
        "data/trip_counts_by_mode_and_distance_per_capita.csv",  # Default name/path relative to data_base_dir
    )
    tunable_coefficients_config = calibration_config.get("tunable_coefficients", {})
    sample_size_schedule = calibration_config.get("sample_size_schedule", {})

    # --- Validate essential inputs ---
    if not output_dir:
        print("Error: Output directory not specified. Use -o flag.", file=sys.stderr)
        sys.exit(1)
    if not configs_base_dir:
        print("Error: Config directory not specified. Use -c flag.", file=sys.stderr)
        sys.exit(1)
    if not data_base_dir:
        print("Error: Data directory not specified. Use -d flag.", file=sys.stderr)
        sys.exit(1)
    if not main_settings_file_name:
        print(
            "Error: 'main_settings_file' (filename relative to data_dir) not specified in calibration config.",
            file=sys.stderr,
        )
        sys.exit(1)

    def objective(x, x0=None, scale=1.0):
        """
        Objective function for calibration.
        Runs the simulation with current coefficients and calculates errors.
        """

        global counter
        global calibration_logger

        counter += 1  # Increment counter at the start of each iteration

        print(f"\n--- Calibration Iteration {counter} ---")
        print(f"Current coefficients (x): {x}")

        # Construct full paths needed for this iteration using base dirs from calibrate scope
        main_settings_file_path = os.path.join(
            configs_base_dir, main_settings_file_name
        )
        calibration_data_file_path = os.path.join(
            data_base_dir, calibration_data_file_name
        )

        # Update the configuration files with the coefficients x
        # Pass the configs_base_dir and tunable_coefficients_config derived earlier
        update_coefficients(x, configs_base_dir, tunable_coefficients_config)

        # Update sample size based on schedule AFTER all coefficients for this iteration are processed
        # Pass the main_settings_file_path derived earlier
        if (
            counter in sample_size_schedule
        ):  # Use the sample_size_schedule derived earlier
            update_sample_size(main_settings_file_path, sample_size_schedule[counter])

        # Run the ActivitySim simulation
        # The 'run' function uses the args object which already has the correct paths
        print("Running ActivitySim simulation...")
        run(args)  # Use the args object directly

        # Calculate the errors
        # Pass the output_dir and calibration_data_file_path derived earlier
        errors = calculate_errors(output_dir, calibration_data_file_path)

        print(f"Calculated errors: {errors}")
        print(" -------------------------------- ")
        print(" \n")
        print(f"Aggregate errors: {np.linalg.norm(errors)}")
        print(" \n")
        print(" -------------------------------- ")

        log_data = [counter] + list(x) + list(errors)
        # Ensure all elements are strings before joining
        calibration_logger.info(",".join(map(str, log_data)) + "\n")
        print(f" --> Logged calibration progress for iteration {counter}")
        # print(f"Logged progress for iteration {counter}") # Optional: Add this line for more logging detail
        if x0 is None:
            return errors  # The optimizer expects the array of errors
        else:
            return np.concatenate([errors, scale * (x - x0)])

    def calculate_errors(model_output_dir, calibration_data_file_path):
        """Calculates the differences between the simulation output and the calibration data."""
        print("Calculating errors...")
        # global config # REMOVE this global - paths are passed in
        global _inferred_num_errors  # Access global for fallback

        # calibration_data_full_path = os.path.join(
        #     os.path.dirname(__file__),
        #     calibration_data_path_in_config
        # )
        calibration_data_file_path = calibration_data_file_path
        calibration_data = pd.read_csv(calibration_data_file_path)
        calibration_data = calibration_data.set_index("distance_bin", drop=True)
        calibration_data_file_path = calibration_data_file_path
        print(f"Calibration data: {calibration_data}")

        # Load simulation output
        # Simulation output path is relative to the model_output_dir specified in config
        simulation_output_path = os.path.join(
            model_output_dir, "summarize/trips_by_mode_and_distance_bin.csv"
        )

        if not os.path.exists(simulation_output_path):
            print(
                f"Warning: Simulation output file not found at {simulation_output_path} after iteration {counter}. Returning zero errors.",
                file=sys.stderr,
            )
            # If output doesn't exist, it's a failed simulation run.
            # Return an array of zeros matching the expected shape derived from calibration data.
            expected_shape = calibration_data.shape
            return np.zeros(
                _inferred_num_errors if _inferred_num_errors > 0 else expected_shape
            ).flatten()

        try:
            simulation_output = pd.read_csv(simulation_output_path)
            # Validate expected columns before setting index
            if "distance_bin" not in simulation_output.columns:
                print(
                    f"Warning: Simulation output file {simulation_output_path} is missing 'distance_bin' column. Cannot calculate errors. Returning zero errors.",
                    file=sys.stderr,
                )
                expected_shape = calibration_data.shape
                return np.zeros(
                    _inferred_num_errors if _inferred_num_errors > 0 else expected_shape
                ).flatten()

            simulation_output = simulation_output.set_index("distance_bin", drop=True)
            print(f"Simulation output loaded from: {simulation_output_path}")

            # FIX: Align columns and indices between simulation output and calibration data
            expected_cal_cols = [
                "HOV",
                "Non-Motorized",
                "Ride Hail",
                "SOV",
                "Transit",
            ]  # Redefine or pass in if needed
            if "distance_bin" in calibration_data.columns:  # Check if index is set
                expected_cal_cols = (
                    calibration_data.columns
                )  # Use columns from loaded calibration data

            sim_output_aligned = simulation_output.reindex(
                columns=expected_cal_cols, fill_value=0.0
            )
            aligned_simulation_output = sim_output_aligned.reindex(
                index=calibration_data.index, fill_value=0.0
            )
            aligned_calibration_data = (
                calibration_data  # Calibration data is the source of truth for shape
            )

        except Exception as e:
            print(
                f"Error loading simulation output from {simulation_output_path}: {e}. Returning zero errors.",
                file=sys.stderr,
            )
            # If file exists but fails to load, treat as a bad run? Return zeros.
            expected_shape = calibration_data.shape
            return np.zeros(aligned_calibration_data.shape).flatten()

        # --- Load total population ---
        # Population count path is relative to the model_output_dir specified in config
        population_count_path = os.path.join(
            model_output_dir, "summarize/persons_count.csv"
        )
        if not os.path.exists(population_count_path):
            print(
                f"Warning: Population count file not found at {population_count_path} after iteration {counter}. Cannot normalize. Returning zero errors.",
                file=sys.stderr,
            )
            return np.zeros(aligned_calibration_data.shape).flatten()

        try:
            # Assuming persons_count.csv has a single value which is the total population
            # Add error handling in case the file is empty or malformed
            pop_df = pd.read_csv(population_count_path)
            if pop_df.empty or pop_df.shape[0] == 0 or pop_df.shape[1] == 0:
                print(
                    f"Warning: Population count file {population_count_path} is empty or malformed. Cannot normalize. Returning zero errors.",
                    file=sys.stderr,
                )
                return np.zeros(aligned_calibration_data.shape).flatten()
            print(pop_df)
            total_population = pop_df.values[0][0]  # Get the first cell value

            if total_population == 0:
                print(
                    f"Warning: Total population is zero in {population_count_path}. Cannot normalize. Returning zero errors.",
                    file=sys.stderr,
                )
                return np.zeros(aligned_calibration_data.shape).flatten()

            print(
                f"Total population loaded from {population_count_path}: {total_population}"
            )
        except Exception as e:
            print(
                f"Error loading total population from {population_count_path}: {e}. Returning zero errors.",
                file=sys.stderr,
            )
            return np.zeros(aligned_calibration_data.shape).flatten()

        # Normalize simulation output
        simulation_output_normalized = aligned_simulation_output / total_population

        # Calculate differences
        differences = simulation_output_normalized - aligned_calibration_data

        print(f"Differences: {differences}")

        return differences.fillna(0.0).values.flatten()

    def update_sample_size(main_settings_file_path, new_sample_size):
        """Updates the sample size in the main settings file using YAML."""
        # main_settings_file = config_dict["main_settings_file"] # Path is passed in
        print(f"Updating sample size in {main_settings_file_path} to {new_sample_size}")
        try:
            with open(main_settings_file_path, "r") as f:
                all_lines = f.readlines()
                # settings = yaml.safe_load(f)
            with open(main_settings_file_path, "w") as f:
                for line in all_lines:
                    if line.startswith("households_sample_size"):
                        newline = "households_sample_size: {0} \n".format(
                            new_sample_size
                        )
                    else:
                        newline = line
                    f.writelines(newline)
        except FileNotFoundError:
            print(
                f"Error: settings file not found at {main_settings_file_path}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"Error updating settings file {main_settings_file_path}: {e}",
                file=sys.stderr,
            )

    def update_coefficients(x, configs_base_dir, tunable_coefficients_config):
        """
        Updates coefficients in configuration CSV files based on the provided values.
        Modifies files line by line to preserve original formatting.
        """
        i = 0  # Counter for the index in the x array
        for coef_file_name, coef_details in tunable_coefficients_config.items():
            file_loc = os.path.join(
                configs_base_dir, coef_file_name
            )  # Use configs_base_dir passed in
            try:
                with open(file_loc, "r") as f:
                    all_lines = f.readlines()
                new_lines = []
                modified_any = False  # Flag to check if any modification was made
                for line in all_lines:
                    modified_line = line  # Start with the original line
                    for coef_name in coef_details.keys():
                        if line.strip().startswith(f"{coef_name},"):
                            if i < len(x):
                                new_value = x[i]
                                print(f"  - Updating '{coef_name}' to {new_value}")

                                # Split the line by comma, replace the second element (index 1), and re-join
                                parts = line.split(",")
                                if len(parts) > 1:
                                    # Preserve original trailing characters/whitespace on the value part
                                    original_trailing = parts[1][
                                        len(parts[1].rstrip()) :
                                    ]
                                    parts[1] = str(new_value) + original_trailing
                                    modified_line = ",".join(parts)
                                    modified_any = True  # Add flag
                                    i += 1
                                else:
                                    print(
                                        f"  - Warning: Coefficient line for '{coef_name}' in {file_loc} doesn't seem to have a value part: '{line.strip()}'",
                                        file=sys.stderr,
                                    )
                                break
                            else:  # Not enough values in x
                                print(
                                    f"Error: More coefficients expected than provided by optimizer x.",
                                    file=sys.stderr,
                                )
                                break  # Stop processing this file's coefs
                    new_lines.append(modified_line)
                if modified_any:
                    with open(file_loc, "w") as f:
                        f.writelines(new_lines)
            except FileNotFoundError:
                print(
                    f"Error: Coefficient file not found at {file_loc}. Skipping updates for this file.",
                    file=sys.stderr,
                )
                i += len(coef_details)  # Skip indices for this file's coefs
            except Exception as e:
                print(
                    f"Error processing coefficient file {file_loc}: {e}. Skipping updates for this file.",
                    file=sys.stderr,
                )
                i += len(coef_details)  # Skip indices for this file's coefs

    def run_simulation(args):
        # This is a placeholder.  The actual implementation will run the ActivitySim simulation.
        print("Running ActivitySim simulation...")

        # Run the simulation
        run(args)

    # calibration_log_path = os.path.join(output_dir, "calibration_progress.csv")
    calibration_log_path = os.path.join(output_dir, "calibration_progress.csv")

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Open in write mode ('w') to create/overwrite the file each calibration run
        # Write header to the log file
        header_parts = ["Iteration"]
        # Collect coefficient names for the header
        coef_names_list = []
        for coef_file, coef_details in tunable_coefficients_config.items():
            coef_names_list.extend(coef_details.keys())
        header_parts.extend(coef_names_list)

        # We need to know the expected number of errors to create error headers
        # Inferring from calibration data file path constructed using args and config filename.
        # Use calibration_data_file_path derived earlier
        calibration_data_file_path = os.path.join(
            data_base_dir, calibration_data_file_name
        )  # Re-construct here for log setup

        num_errors = 0
        try:
            # Temporarily read just to get shape, assuming format is consistent and file exists
            dummy_cal_data = pd.read_csv(calibration_data_file_path).set_index(
                "distance_bin", drop=True
            )

            num_errors = dummy_cal_data.shape[0] * dummy_cal_data.shape[1]
            print(f"Inferred {num_errors} error values from calibration data shape.")
        except Exception as e:
            print(
                f"Warning: Could not determine expected number of errors from calibration data file {calibration_data_file_path} ({e}). Cannot create full log header with Error_N columns.",
                file=sys.stderr,
            )

        if num_errors > 0:
            header_parts.extend([f"Error_{i+1}" for i in range(num_errors)])
        elif len(header_parts) > 1:
            header_parts.append("Errors...")  # Placeholder

        if header_parts:
            calibration_logger.info(",".join(header_parts) + "\n")
        _inferred_num_errors = num_errors  # Store inferred num_errors globally for calculate_errors fallback
    except Exception as e:
        print(
            f"Error setting up calibration log file {calibration_log_path}: {e}. Proceeding without logging calibration progress.",
            file=sys.stderr,
        )

    # --- Prepare optimization ---
    x0_list = []
    bounds_lower = []
    bounds_upper = []
    for coef_file, coef_details in tunable_coefficients_config.items():
        for coef_name, values in coef_details.items():
            if isinstance(values, float):
                print("This is a problem")
            if (
                "initial_value" not in values
                or "bounds" not in values
                or not isinstance(values["bounds"], list)
                or len(values["bounds"]) != 2
            ):
                print(
                    f"Error: Missing 'initial_value' or 'bounds' for coefficient '{coef_name}' in file '{coef_file}'. Skipping.",
                    file=sys.stderr,
                )
                continue
            x0_list.append(values["initial_value"])
            bounds_lower.append(values["bounds"][0])
            bounds_upper.append(values["bounds"][1])

    x0 = np.array(x0_list)
    bounds = (np.array(bounds_lower), np.array(bounds_upper))

    if len(x0) == 0:
        print(
            "Error: No valid tunable coefficients found in the calibration config. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Calibration progress will be logged to: {calibration_log_path}")

    # Run the calibration using dfols
    print("Starting optimization with DFO-LS...")
    try:
        result = dfols.solve(
            objective,
            x0,
            bounds=bounds,
            scaling_within_bounds=True,
            objfun_has_noise=True,
            maxfun=5000,
            user_params={"restarts.use_restarts": False},
            argsf=(x0, regularization)
        )

        print("\n--- Optimization Result ---")
        print(result)
        # print("\n--- Final Coefficients ---") # Already printed by DFO-LS result
        print("\nUpdating config files with optimal coefficients...")
        update_coefficients(result.x, configs_base_dir, tunable_coefficients_config)
        print("Config files updated.")
    except Exception as e:
        print(f"An error occurred during optimization: {e}", file=sys.stderr)
    finally:
        calibration_logger.info(
            f"Calibration progress log file closed: {calibration_log_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()

    if args.calibration_config:
        global calibration_logger
        calibration_logger = setup_logger(
            "calibration_logger", os.path.join(args.output, "calibration_progress.log")
        )
        calibration_logger.info(
            "Calibration progress will also be logged to: {0}".format(
                os.path.join(args.output, "calibration_progress.csv")
            )
        )
        reg = args.regularization
        print(
            "Calibration progress will also be logged to: {0}".format(
                os.path.join(args.output, "calibration_progress.csv")
            )
        )

        calibrate(args.calibration_config, args, reg)
    else:
        # This branch runs the standard simulation using the args provided
        run(args)
