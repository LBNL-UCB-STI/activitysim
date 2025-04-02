from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import random
from shapely import wkt
from shapely.geometry import Point, MultiPoint
import geopandas as gpd
import logging
import warnings

from activitysim.core.configuration import PydanticReadable
from activitysim.core import config, los, workflow, expressions, mem
from activitysim.core.configuration.base import PreprocessorSettings
# from activitysim.core import pipeline, orca, config
# from activitysim.core import inject
# from activitysim.core.mem import force_garbage_collect
from activitysim.core.simulate import set_skim_wrapper_targets

logger = logging.getLogger("activitysim")
warnings.filterwarnings("ignore", "GeoSeries.isna", UserWarning)


def random_points_in_polygon(number, polygon):
    """
    Generate n number of points within a polygon
    Input:
    -number: n number of points to be generated
    - polygon: geopandas polygon
    Return:
    - List of shapely points
    source: https://gis.stackexchange.com/questions/294394/
        randomly-sample-from-geopandas-dataframe-in-python
    """
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    i = 0
    while i < number:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(point):
            points.append(point)
            i += 1
    return points  # returns list of shapely point


def sample_geoseries(geoseries, size, overestimate=2):
    """
    Generate at most "size" number of points within a polygon
    Input:
    - size: n number of points to be generated
    - geoseries: geopandas polygon
    - overestimate = int to multiply the size. It will account for
        points that may fall outside the polygon
    Return:
    - List points
    source: https://gis.stackexchange.com/questions/294394/
        randomly-sample-from-geopandas-dataframe-in-python
    """
    polygon = geoseries.unary_union
    min_x, min_y, max_x, max_y = polygon.bounds
    ratio = polygon.area / polygon.envelope.area
    overestimate = 2
    # np.random.uniform can't specify dtype, so let's try a different method
    # samples = np.random.uniform(
    #     (min_x, min_y), (max_x, max_y), (int(size / ratio * overestimate), 2)
    # )
    samples = np.random.default_rng().random(size=(int(size / ratio * overestimate), 2), dtype=np.float32) * np.array(
        [(max_x - min_x), (max_y - min_y)]) + np.array([min_x, min_y])
    multipoint = MultiPoint(samples)
    multipoint = multipoint.intersection(polygon)
    samples = np.array(multipoint.geoms)
    return samples[np.random.choice(len(samples), size)]


def get_trip_coords(trips, zones, persons, state, max_points_per_zone=300):
    """
    Assign coordinates to trips with efficient point generation and bounded memory usage

    Parameters:
    -----------
    trips : DataFrame
        The trips dataframe to assign coordinates to
    zones : DataFrame
        Spatial zones with geometry information
    persons : DataFrame
        Person data containing home coordinates
    state : dict
        State dictionary for memory tracking
    max_points_per_zone : int, default=100
        Maximum number of random points to generate per zone
    """
    # Lowercase purposes
    trips["purpose"] = trips["purpose"].str.lower()

    logger.info(f"Generating up to {max_points_per_zone} random points per zone")

    # Generate points only for zones that appear in trips and have valid geometry
    unique_zones = set(trips['origin'].unique())
    valid_zones = zones[~(zones["geometry"].is_empty | zones["geometry"].isna())]
    valid_zone_ids = set(valid_zones.zone_id) & unique_zones

    rand_point_zones = {}
    for zone in valid_zone_ids:
        polygon = zones[zones.zone_id == zone].geometry
        # Generate at most max_points_per_zone points per zone
        points = sample_geoseries(polygon, max_points_per_zone, overestimate=1.5)
        rand_point_zones[zone] = points

    logger.info(f"Generated points for {len(rand_point_zones)} zones. Assigning trip locations.")

    # Preserve original index and sort for groupby efficiency
    original_index = trips.index.copy()
    trips.sort_values(["person_id", "origin", "purpose"], inplace=True)

    # Use vectorized approach with capped number of points
    for (person_id, origin, purpose), group in trips.groupby(["person_id", "origin", "purpose"]):
        if origin in rand_point_zones:
            zs = rand_point_zones[origin]
            if len(zs) > 0:  # Make sure we have points
                z = random.choice(zs)
                trips.loc[group.index, "x"] = z.x
                trips.loc[group.index, "y"] = z.y

    # Restore original order
    trips = trips.reindex(original_index)

    # Clear dictionary and force garbage collection
    del rand_point_zones
    mem.trace_memory_info("Just generated random points", force_garbage_collect=True, state=state)

    logger.info("Done assigning trip locations. Adopting home trip locations.")

    # Home location assignment using vectorized operations
    origin_purpose_is_home = (
            trips.groupby("person_id")["purpose"].shift(periods=1).fillna("home") == "home"
    )

    # Only process if we have any home origins
    if origin_purpose_is_home.any():
        # Get relevant person IDs
        home_person_ids = trips.loc[origin_purpose_is_home, "person_id"]

        # Create a lookup for home coordinates
        home_coords_dict = persons[['home_x', 'home_y']]

        # Use vectorized assignment
        trips.loc[origin_purpose_is_home, ["x", "y"]] = home_coords_dict.loc[home_person_ids].values

    logger.info("Done adopting home trip locations.")

    return trips


def get_trip_coords_old(trips, zones, persons, state, size=500):
    # Generates random points within each zone for zones
    # that are not empty geometries (i.e. contain no blocks)
    trips["purpose"] = trips["purpose"].str.lower()
    rand_point_zones = {}
    for zone in zones[~(zones["geometry"].is_empty | zones["geometry"].isna())].zone_id:
        size = 200
        polygon = zones[zones.zone_id == zone].geometry
        points = sample_geoseries(polygon, size, overestimate=2)
        rand_point_zones[zone] = points

    def assignLoc(grp):
        zs = rand_point_zones[grp.iloc[0]["origin"]]
        z = random.choice(zs)
        grp["x"] = z[0]
        grp["y"] = z[1]
        return grp

    logger.info("Done generating random points in zones. Assigning trip locations.")

    # trips = trips.groupby(["person_id", "origin", "purpose"]).apply(assignLoc)
    # Process in chunks to reduce memory usage
    original_index = trips.index.copy()
    trips.sort_values(["person_id", "origin", "purpose"], inplace=True)
    for (person_id, origin, purpose), group in trips.groupby(["person_id", "origin", "purpose"]):
        if origin in rand_point_zones:
            zs = rand_point_zones[origin]
            z = random.choice(zs)
            trips.loc[group.index, "x"] = z.x
            trips.loc[group.index, "y"] = z.y

    trips = trips.reindex(original_index)

    # Clear dictionary and force garbage collection
    del rand_point_zones
    mem.trace_memory_info("Just generated random points", force_garbage_collect=True, state=state)

    # retain home coords from urbansim data bc they will typically be
    # higher resolution than zone, so we don't need the semi-random coords

    logger.info("Done assigning trip locations. Adopting home trip locations.")

    origin_purpose_is_home = (
            trips.groupby("person_id")["purpose"].shift(periods=1).fillna("home") == "home"
    )
    trips.loc[origin_purpose_is_home, ["x", "y"]] = persons[["home_x", "home_y"]].reindex(
        trips.loc[origin_purpose_is_home, "person_id"]).values

    logger.info("Done adopting home trip locations.")

    return trips


def generate_departure_times(trips, state):
    """
    Generate randomized departure times respecting hour bins
    Assumes trips are already sorted in correct sequence with monotonically increasing hours
    """
    logger.info("Generating randomized departure times within hour bins")

    # Store original index
    orig_index = trips.index.copy()

    # Create working copy with just what we need
    work_df = trips[["trip_id", "person_id", "depart", "TOTAL_TIME_MINS"]].copy()

    # Generate random fractions
    rng = np.random.default_rng()
    work_df["frac"] = rng.random(size=len(work_df), dtype=np.float32)

    # Initialize start times (default to hour + random fraction)
    work_df['start_time'] = work_df['depart'] + work_df['frac']

    # Pre-allocate result
    result_times = pd.Series(index=orig_index, dtype=np.float32, name="depart")

    # Process each person
    for person_id, person_trips in work_df.groupby("person_id"):
        if person_trips.empty:
            continue

        # Get data as arrays for faster processing
        trip_ids = person_trips.index.values
        start_times = person_trips['start_time'].values
        durations = person_trips['TOTAL_TIME_MINS'].values / 60.0

        # Forward pass to resolve overlaps
        n_trips = len(person_trips)
        for i in range(1, n_trips):
            prev_end = start_times[i - 1] + durations[i - 1]
            if start_times[i] < prev_end:
                start_times[i] = prev_end

        # Update result Series directly
        result_times.loc[trip_ids] = start_times

    # Check for missing values
    missing_count = result_times.isna().sum()
    if missing_count > 0:
        logger.warning(f"Missing departure times for {missing_count} trips ({missing_count / len(result_times):.1%})")

    mem.trace_memory_info("Generated departure times", force_garbage_collect=True, state=state)
    return result_times

def generate_departure_times_old(trips, state):
    orig_index = trips.index.copy()
    # Select only required columns and convert to efficient dtypes
    ordered_trips2 = trips[
        [
            "trip_id",
            "person_id",
            "depart",
            "TOTAL_TIME_MINS",
        ]
    ].reset_index()
    del trips

    # Use numpy's more efficient random number generator
    ordered_trips2["frac"] = np.random.default_rng().random(size=len(ordered_trips2), dtype=np.float32)
    ordered_trips2.index.name = "og_df_idx"

    def getTotalTime(df):
        df["frac"] = df["frac"].sort_values(ascending=True).values
        cannotSpillIntoNextWindow = df.iloc[-1]["mustFinishWithinHour"]
        if cannotSpillIntoNextWindow:
            allowableDuration = 60.0
        else:
            allowableDuration = 60.0 + df.iloc[-1]["TOTAL_TIME_MINS"]
        totalBuffer = np.max([(allowableDuration - df["TOTAL_TIME_MINS"].sum()) / 60.0, 0.0])
        df["newStartTime"] = (
                df["depart"]
                + df["frac"] * totalBuffer
                + df["TOTAL_TIME_MINS"].shift(1).fillna(0.0).cumsum() / 60.0
        )

        i = 0
        while True:
            # Calculate gaps between trips
            df["gapAfterTrip"] = -(
                    (df["newStartTime"] + df["TOTAL_TIME_MINS"] / 60.0)
                    - df["newStartTime"].shift(-1).fillna(100).values
            )
            if i > 15:
                logger.warning("Bad trip times still {0}".format(df.loc[df["gapAfterTrip"] < 0, :]))

                break

            # If all gaps are non-negative, we're done
            if np.all(df["gapAfterTrip"].values >= 0):
                break

            # Find problematic trips (negative gaps)
            negative_gaps = df["gapAfterTrip"] < 0

            # Adjust start times for trips that need fixing
            df.loc[negative_gaps, "newStartTime"] += df.loc[negative_gaps, "gapAfterTrip"]

            i += 1

        return df

    def process(df):
        df["mustFinishWithinHour"] = (
                df["depart"] >= df["depart"].shift(-1).fillna(24) - 1
        )
        df = df.groupby("depart").apply(getTotalTime)
        return df[["trip_id", "newStartTime"]]

    mem.trace_memory_info("Just generated departure times", force_garbage_collect=True, state=state)
    df2 = ordered_trips2.groupby(["person_id"]).apply(process)
    # df2.set_index("trip_id", inplace=True)
    df2 = df2.reindex(orig_index)
    return df2.newStartTime.rename("depart")


class MatrixTableSettings(PydanticReadable):
    name: str
    data_field: str


class MatrixSettings(PydanticReadable):
    file_name: Path
    tables: List[MatrixTableSettings] = []
    is_tap: bool = False


class TimePeriodSettings(PydanticReadable):
    first_hour: int
    last_hour: int


class ConstantsSettings(PydanticReadable):
    time_periods: Dict[str, TimePeriodSettings] = {}
    OCC_SHARED2: float = 0.0
    OCC_SHARED3: float = 0.0


class GenerateBeamPlansSettings(PydanticReadable):
    """
    Settings for generating beam plans.
    """

    preprocessor: PreprocessorSettings | None = None
    HH_EXPANSION_WEIGHT_COL: str = "sample_rate"
    SAVE_TRIPS_TABLE: bool = False
    MATRICES: List[MatrixSettings] = []
    CONSTANTS: Dict[str, Any] = {}


@workflow.step
def generate_beam_plans(
        state: workflow.State,
        trips,
        tours,
        persons,
        network_los: los.Network_LOS,
        model_settings: GenerateBeamPlansSettings | None = None,
        model_settings_file_name: str = "generate_beam_plans.yaml",
        trace_label: str = "generate_beam_plans",
) -> None:
    tourPurposeCategory = pd.CategoricalDtype(["non_mandatory", "joint", "mandatory", "atwork"], ordered=True)
    # Convert to frames only once and work in-place where possible
    col_to_keep = ['trip_id', 'person_id', 'tour_id',
                   'trip_num', 'outbound', 'purpose', 'primary_purpose', 'destination',
                   'origin', 'depart', 'trip_mode']
    trips.drop(columns=[col for col in trips.columns if col not in col_to_keep], inplace=True)
    tour_col_to_keep = ['tour_id', 'person_id', 'number_of_participants', 'start', 'end', 'tour_mode', 'parent_tour_id',
                        'tour_num', 'tour_category']
    tours.drop(columns=[col for col in tours.columns if col not in tour_col_to_keep], inplace=True)
    tours["parent_tour_id"] = tours["parent_tour_id"].astype(pd.Int64Dtype())
    tours.index = tours.index.astype(pd.Int64Dtype())
    tours["parent_tour_num"] = 0
    tours.tour_category.astype(tourPurposeCategory)
    tours.sort_values(["person_id", "start", "tour_category"], inplace=True)
    tours["tour_ordinal"] = tours.groupby("person_id").cumcount()
    tours.loc[~tours.parent_tour_id.isna(), "parent_tour_num"] = tours.loc[
        tours.loc[~tours.parent_tour_id.isna(), "parent_tour_id"], "tour_num"].values

    trips['trip_mode'] = trips['trip_mode'].astype("category")
    trips['purpose'] = trips['purpose'].astype("category")
    trips['primary_purpose'] = trips['primary_purpose'].astype("category")
    trips['origin'] = trips['origin'].astype("category")
    trips['destination'] = trips['destination'].astype("category")
    trips['trip_num'] = trips['trip_num'].astype(pd.Int16Dtype())
    tours['tour_num'] = tours['tour_num'].astype(pd.Int16Dtype())
    trips['depart'] = trips['depart'].astype(np.float32)

    trips = pd.merge(trips.reset_index(), tours[['tour_num', 'parent_tour_num', 'tour_mode', 'tour_ordinal']],
                     on="tour_id")

    trips["trip_num"] += trips["tour_ordinal"] * 100
    trips.loc[~trips.outbound, "trip_num"] += 50
    trips.loc[trips.parent_tour_num > 0, "trip_num"] += 10
    trips.drop(columns=["tour_ordinal"]).sort_values(["person_id", "depart", "trip_num"], inplace=True)

    if model_settings is None:
        model_settings = GenerateBeamPlansSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    constants = config.get_model_constants(model_settings)
    # - run preprocessor to annotate choosers
    preprocessor_settings = model_settings.preprocessor
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

    if state.is_table("beam_geoms"):
        zones = _process_beam_geoms(state.get_table("beam_geoms"))
    else:
        zones = _process_land_use(state.get_table("land_use"))

    # Setup skims
    trips["trip_period"] = network_los.skim_time_period_label(trips.depart)

    # Modify trips dataframe in-place where possible
    _annotate_trips(trips, tours)
    trips.drop(columns=['isAtWork', 'actuallyInbound'], inplace=True)

    inner_chunk_size = 20000
    nChunks, lastChunkSize = divmod(trips.shape[0], inner_chunk_size)
    lastInd = 0

    skim_dict = network_los.get_default_skim_dict()
    skims = _setup_skims(skim_dict)

    constants = config.get_model_constants(model_settings)

    for ii in range(nChunks):
        logger.info("Starting on {0} of {1} chunks".format(ii, nChunks))
        splitPerson = trips['person_id'].values[inner_chunk_size * (ii + 1)]
        splitInd = np.argmax(trips['person_id'].values == splitPerson)
        trips_sub = trips.iloc[lastInd:(splitInd - 1)].copy()
        trips_sub = _process_trip_chunk(trips_sub, constants, skims, model_settings, state, trace_label)
        trips.iloc[lastInd:(splitInd - 1)] = trips_sub[trips.columns].values
        lastInd = splitInd
    if lastChunkSize > 0:
        trips_sub = trips.iloc[lastInd:].copy()
        trips_sub = _process_trip_chunk(trips_sub, constants, skims, model_settings, state, trace_label)
        trips.iloc[lastInd:] = trips_sub[trips.columns].values

    trips.drop(columns=["outbound", "tour_num", "parent_tour_num", "tour_ordinal", "tour_start", "tour_end", "trip_num",
                        "inbound"], inplace=True)

    # Get coordinates and times
    logger.info("Adding trip coordinates")
    trips = get_trip_coords(trips, zones, persons, state)

    # trips.set_index("trip_id", inplace=True)

    logger.info("Generating departure times")
    trips["departure_time"] = generate_departure_times(trips, state)
    # trips["departure_time_2"] = generate_departure_times_old(trips, state)

    # Add tour information efficiently using map
    trips["number_of_participants"] = trips["tour_id"].map(tours["number_of_participants"])
    trips["tour_mode"] = trips["tour_id"].map(tours["tour_mode"])
    trips.rename(columns={
        "TOTAL_TIME_MINS": "trip_dur_min",
        "TOTAL_COST_DOLLARS": "trip_cost_dollars"
    }, inplace=True)

    # Create final plans more efficiently
    final_plans = _create_final_plans(trips)
    # save back to pipeline
    state.add_table("beam_plans", final_plans)


def _fix_sequence_fast(person_trips, trip_indices, origins, destinations, depart_times, time_to_trips, tour_starts,
                       tour_ends):
    """Fix sequence respecting tour time windows with statistical tracking"""
    # Number of trips
    n_trips = len(origins)
    if n_trips <= 1:
        return person_trips, {"status": "no_fix_needed", "trips_shifted": 0, "hours_shifted": 0}

    # Create a mapping to keep track of reordering
    fixed_idx_mapping = np.arange(n_trips)

    # Group trips by departure time
    time_periods = sorted(time_to_trips.keys())

    # Track if we've modified departure times
    modified_depart_times = {}  # {trip_idx: (original_time, new_time)}
    # Track which trips have been moved to avoid double-processing
    moved_trips = set()

    # Process each time period
    for t_idx, time_period in enumerate(time_periods):
        # Get indices of trips in this time period
        period_indices = [i for i in time_to_trips[time_period] if i not in moved_trips]
        if len(period_indices) <= 1:
            continue

        # Find the destination from the last trip of previous time period
        prev_destination = None
        if t_idx > 0:
            for prev_t_idx in range(t_idx - 1, -1, -1):
                prev_time = time_periods[prev_t_idx]
                prev_period_indices = [i for i in time_to_trips[prev_time] if i not in moved_trips]
                if prev_period_indices:
                    # Sort by current mapping order
                    prev_period_indices.sort(key=lambda i: fixed_idx_mapping.tolist().index(i))
                    last_prev_idx = prev_period_indices[-1]
                    prev_destination = destinations[last_prev_idx]
                    break

        # Identify separate chains in this time period
        chains = []
        remaining_indices = set(period_indices)

        # First, find chain that connects with previous time period
        if prev_destination is not None:
            # Find trips that start at prev_destination
            connecting_trips = [i for i in period_indices if origins[i] == prev_destination]

            if connecting_trips:
                # Start a chain with the connecting trip
                first_trip = connecting_trips[0]
                first_chain = [first_trip]
                remaining_indices.remove(first_trip)

                # Build the chain
                current_dest = destinations[first_trip]
                while True:
                    next_trips = [i for i in remaining_indices if origins[i] == current_dest]
                    if not next_trips:
                        break
                    next_trip = next_trips[0]
                    first_chain.append(next_trip)
                    remaining_indices.remove(next_trip)
                    current_dest = destinations[next_trip]

                chains.append(first_chain)
            else:
                # Look for connecting trips in other time periods within tour window
                connecting_across_time = []

                for adj_time in time_periods:
                    if adj_time != time_period:  # Don't recheck current time period
                        adj_period_indices = [i for i in time_to_trips[adj_time] if i not in moved_trips]
                        for idx in adj_period_indices:
                            # Check if trip connects and can be moved to current time
                            if origins[idx] == prev_destination and \
                                    tour_starts[idx] <= time_period <= tour_ends[idx]:
                                connecting_across_time.append((idx, abs(adj_time - time_period)))

                if connecting_across_time:
                    # Sort by time difference to minimize schedule disruption
                    connecting_across_time.sort(key=lambda x: x[1])
                    adj_trip = connecting_across_time[0][0]
                    orig_time = depart_times[adj_trip]

                    # Prevent removing from the list twice
                    if adj_trip in time_to_trips[orig_time] and adj_trip not in moved_trips:
                        # Remove from original time period
                        time_to_trips[orig_time].remove(adj_trip)
                        # Mark as moved
                        moved_trips.add(adj_trip)

                        # Add to current time period
                        if adj_trip not in time_to_trips[time_period]:
                            time_to_trips[time_period].append(adj_trip)

                        # Mark for departure time update
                        modified_depart_times[adj_trip] = time_period

                        # Start a chain with this trip
                        first_chain = [adj_trip]
                        if adj_trip in remaining_indices:
                            remaining_indices.remove(adj_trip)

                        # Build the chain
                        current_dest = destinations[adj_trip]
                        while True:
                            next_trips = [i for i in remaining_indices if origins[i] == current_dest]
                            if not next_trips:
                                break
                            next_trip = next_trips[0]
                            first_chain.append(next_trip)
                            remaining_indices.remove(next_trip)
                            current_dest = destinations[next_trip]

                        chains.append(first_chain)

        # Now identify other separate chains
        while remaining_indices:
            # Start a new chain
            start_trip = list(remaining_indices)[0]
            current_chain = [start_trip]
            remaining_indices.remove(start_trip)

            # Forward building - add trips that follow this one
            current_dest = destinations[start_trip]
            while True:
                next_trips = [i for i in remaining_indices if origins[i] == current_dest]
                if not next_trips:
                    # Check other time periods
                    next_across_time = []

                    for adj_time in time_periods:
                        if adj_time != time_period:
                            adj_period_indices = [i for i in time_to_trips[adj_time] if i not in moved_trips]
                            for idx in adj_period_indices:
                                if origins[idx] == current_dest and \
                                        tour_starts[idx] <= time_period <= tour_ends[idx]:
                                    next_across_time.append((idx, abs(adj_time - time_period)))

                    if next_across_time:
                        next_across_time.sort(key=lambda x: x[1])
                        adj_trip = next_across_time[0][0]
                        orig_time = depart_times[adj_trip]

                        # Safely remove from original time period
                        if adj_trip in time_to_trips[orig_time] and adj_trip not in moved_trips:
                            time_to_trips[orig_time].remove(adj_trip)
                            moved_trips.add(adj_trip)

                            # Add to current time period
                            if adj_trip not in time_to_trips[time_period]:
                                time_to_trips[time_period].append(adj_trip)

                            # Mark for departure time update
                            modified_depart_times[adj_trip] = time_period

                            # Add to chain
                            current_chain.append(adj_trip)
                            current_dest = destinations[adj_trip]
                        else:
                            break  # Trip already moved, can't use it
                    else:
                        break  # No more connecting trips
                else:
                    next_trip = next_trips[0]
                    current_chain.append(next_trip)
                    remaining_indices.remove(next_trip)
                    current_dest = destinations[next_trip]

            # Backward building - add trips that precede this one
            current_orig = origins[start_trip]
            while True:
                prev_trips = [i for i in remaining_indices if destinations[i] == current_orig]
                if not prev_trips:
                    # Check other time periods
                    prev_across_time = []

                    for adj_time in time_periods:
                        if adj_time != time_period:
                            adj_period_indices = [i for i in time_to_trips[adj_time] if i not in moved_trips]
                            for idx in adj_period_indices:
                                if destinations[idx] == current_orig and \
                                        tour_starts[idx] <= time_period <= tour_ends[idx]:
                                    prev_across_time.append((idx, abs(adj_time - time_period)))

                    if prev_across_time:
                        prev_across_time.sort(key=lambda x: x[1])
                        adj_trip = prev_across_time[0][0]
                        orig_time = depart_times[adj_trip]

                        # Safely remove from original time period
                        if adj_trip in time_to_trips[orig_time] and adj_trip not in moved_trips:
                            time_to_trips[orig_time].remove(adj_trip)
                            moved_trips.add(adj_trip)

                            # Add to current time period
                            if adj_trip not in time_to_trips[time_period]:
                                time_to_trips[time_period].append(adj_trip)

                            # Mark for departure time update
                            modified_depart_times[adj_trip] = time_period

                            # Add to chain
                            current_chain.insert(0, adj_trip)
                            current_orig = origins[adj_trip]
                        else:
                            break  # Trip already moved, can't use it
                    else:
                        break  # No more connecting trips
                else:
                    prev_trip = prev_trips[0]
                    current_chain.insert(0, prev_trip)
                    remaining_indices.remove(prev_trip)
                    current_orig = origins[prev_trip]

            chains.append(current_chain)

        # Now reorder the trips within this time period based on the chains
        new_order = []
        for chain in chains:
            new_order.extend(chain)

        # Update the fixed_idx_mapping for this time period
        if new_order:  # Only proceed if we have trips to reorder
            # Get positions of all trips currently in this time period (including moved trips)
            all_period_trips = time_to_trips[time_period].copy()
            period_positions = []

            for idx in all_period_trips:
                try:
                    # Find position in the fixed_idx_mapping
                    pos = np.where(fixed_idx_mapping == idx)[0][0]
                    period_positions.append(pos)
                except IndexError:
                    # Skip if not found - shouldn't happen but being defensive
                    logger.warning(f"Trip index {idx} not found in fixed_idx_mapping")
                    continue

            period_positions.sort()

            # Apply the new ordering
            for new_idx, trip_idx in enumerate(new_order):
                if new_idx < len(period_positions):
                    pos = period_positions[new_idx]
                    fixed_idx_mapping[pos] = trip_idx

    # Apply the modified departure times to the dataframe
    for trip_idx, new_time in modified_depart_times.items():
        try:
            # Find the position in the reordered dataframe
            pos = np.where(fixed_idx_mapping == trip_idx)[0][0]
            # Update the dataframe
            person_trips.iloc[pos, person_trips.columns.get_loc('depart')] = new_time

        except IndexError:
            logger.warning(f"Could not update departure time for trip {trip_idx} - not found in mapping")

    # Return reordered trips
    # Calculate statistics
    trips_shifted = len(modified_depart_times)
    total_hours_shifted = 0

    if trips_shifted > 0:
        # Calculate total shift and average
        for trip_idx, new_time in modified_depart_times.items():
            old_time = depart_times[trip_idx]  # Get the original time from the depart_times array
            total_hours_shifted += abs(new_time - old_time)

        # Determine status
        status = "fixed_with_shifts"
    else:
        # If we didn't need to shift any trips but still fixed the sequence
        status = "fixed_without_shifts"

    # Create stats dictionary
    stats = {
        "status": status,
        "trips_shifted": trips_shifted,
        "hours_shifted": total_hours_shifted,
        "avg_hours_shifted": total_hours_shifted / trips_shifted if trips_shifted > 0 else 0
    }

    # Return reordered trips and statistics
    return person_trips.iloc[fixed_idx_mapping].reset_index(drop=True), stats


def _fix_trips_chunk(trips_chunk):
    """Fix trip sequences for a chunk of people efficiently"""
    # First, detect inconsistencies for all persons at once (vectorized)
    topo_mask = ((trips_chunk["destination"].shift() == trips_chunk["origin"]) |
                 (trips_chunk["person_id"].shift() != trips_chunk["person_id"]))
    trips_chunk["is_bad"] = ~topo_mask

    # Find which persons have inconsistencies
    persons_with_bad_trips = trips_chunk[trips_chunk["is_bad"]]["person_id"].unique()

    if len(persons_with_bad_trips) == 0:
        return trips_chunk  # No inconsistencies to fix

    logger.info(f"Found {len(persons_with_bad_trips)} persons with inconsistent trips")

    # Process only those persons who have inconsistencies
    all_fixed_trips = []
    unchanged_mask = ~trips_chunk["person_id"].isin(persons_with_bad_trips)

    # Add all persons with no issues directly
    if unchanged_mask.any():
        all_fixed_trips.append(trips_chunk[unchanged_mask])

    # Process each person with inconsistencies
    for person_id in persons_with_bad_trips:
        person_trips = trips_chunk[trips_chunk["person_id"] == person_id].copy()

        # Use numpy arrays for faster processing
        trip_indices = np.array(person_trips.index)
        origins = np.array(person_trips["origin"])
        destinations = np.array(person_trips["destination"])
        depart_times = np.array(person_trips["depart"])

        # Create a fast lookup dictionary for each time period
        time_to_trips = {}
        for i, time in enumerate(depart_times):
            if time not in time_to_trips:
                time_to_trips[time] = []
            time_to_trips[time].append(i)

        # Fix the sequence using the efficient approach
        fixed_person_trips = _fix_sequence_fast(
            person_trips,
            trip_indices,
            origins,
            destinations,
            depart_times,
            time_to_trips
        )
        all_fixed_trips.append(fixed_person_trips)

    return pd.concat(all_fixed_trips) if all_fixed_trips else trips_chunk


def _process_trip_chunk(trips, constants, skims, model_settings, state, trace_label):
    # Determine chunk size based on available memory
    total_trips = len(trips)
    chunk_size = min(500000, total_trips)
    num_chunks = (total_trips + chunk_size - 1) // chunk_size

    logger.info(f"Processing {total_trips} trips in {num_chunks} chunks")

    result_chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_trips)

        logger.info(f"Processing chunk {i + 1}/{num_chunks}, trips {start_idx}-{end_idx}")

        # Process this chunk
        chunk = trips.iloc[start_idx:end_idx].copy()
        processed_chunk = _process_single_chunk(chunk, constants, skims, model_settings, state, trace_label)
        result_chunks.append(processed_chunk)

        # Clean up memory
        del chunk
        mem.trace_memory_info(f"Processed chunk {i + 1}", force_garbage_collect=True, state=state)

    # Combine results
    return pd.concat(result_chunks, ignore_index=True)


def _process_single_chunk(chunk, constants, skims, model_settings, state, trace_label):
    # Sort and fix sequences
    chunk = _sort_and_fix_sequences(chunk, state)
    logger.info("Done rearranging trips")

    chunk['origin'] = chunk['origin'].astype(int)
    chunk['destination'] = chunk['destination'].astype(int)
    chunk['trip_mode'] = chunk['trip_mode'].astype(str)

    logger.info("Annotating trip chunk from skims")
    set_skim_wrapper_targets(chunk, skims)

    expressions.annotate_preprocessors(
        state,
        chunk, constants, skims,
        model_settings, trace_label=trace_label)

    # Rest of processing
    return chunk


def _process_beam_geoms(beam_geoms):
    beam_geoms["geometry"] = gpd.GeoSeries.from_wkt(beam_geoms["geometry"])
    zones = gpd.GeoDataFrame(beam_geoms, geometry="geometry", crs="EPSG:4326")
    zones.geometry = zones.geometry.buffer(0)
    return zones


def _process_land_use(land_use):
    land_use = land_use.reset_index()
    land_use["geometry"] = land_use["geometry"].apply(wkt.loads)
    zones = gpd.GeoDataFrame(land_use, geometry="geometry", crs="EPSG:4326")
    zones.geometry = zones.geometry.buffer(0)
    return zones


def _setup_skims(skim_dict):
    orig_col, dest_col = "origin", "destination"
    return {
        "odt_skims": skim_dict.wrap_3d(
            orig_key=orig_col, dest_key=dest_col, dim3_key="trip_period"
        ),
        "dot_skims": skim_dict.wrap_3d(
            orig_key=dest_col, dest_key=orig_col, dim3_key="trip_period"
        ),
        "od_skims": skim_dict.wrap("origin", "destination")
    }


def _annotate_trips(trips, tours):
    trips["inbound"] = ~trips.outbound
    trips["tour_start"] = trips.tour_id.map(tours.start).astype(pd.Int16Dtype())
    trips["tour_end"] = trips.tour_id.map(tours.end).astype(pd.Int16Dtype())
    trips["isAtWork"] = trips.purpose == "atwork"

    # Handle actuallyInbound calculation
    trips["actuallyInbound"] = trips["inbound"].copy()
    mask_work = (trips.primary_purpose == "work") & (trips.purpose.str.lower() != "home")
    trips.loc[mask_work, "actuallyInbound"] = ~trips.loc[mask_work, "inbound"]
    mask_atwork = (trips.purpose == "atwork")
    trips.loc[mask_atwork, "actuallyInbound"] = ~trips.loc[mask_atwork, "inbound"]
    trips['TOTAL_COST_DOLLARS'] = np.float32(0.0)
    trips['TOTAL_TIME_MINS'] = np.float32(0.0)
    trips['departure_time'] = np.float32(0.0)


def _fix_trip_sequence(df):
    bad_indices = np.nonzero(df.is_bad.values)[0]
    if len(bad_indices) == 0:
        return df

    first_bad_index = bad_indices[0]
    dest_last_good = df.loc[df.index[first_bad_index - 1], "destination"]
    # TODO: allow a window around time period if you don't succeed at first
    time_period = df.loc[df.index[first_bad_index], "depart"]

    mask = ((df["depart"] == time_period) &
            (df["origin"] == dest_last_good) &
            (np.arange(len(df)) > first_bad_index))

    try:
        potential_indices = np.argwhere(mask.values)[0]
        trip_index_to_move = np.random.choice(potential_indices)
        return _reorder_trips(df, first_bad_index, trip_index_to_move)
    except IndexError:
        return _shuffle_trips(df, time_period)


def _reorder_trips(df, first_bad_index, trip_index_to_move):
    df2 = df.copy()
    trips_to_shuffle = df.iloc[first_bad_index:trip_index_to_move].copy()

    df2.iloc[first_bad_index] = df2.iloc[trip_index_to_move].values
    df2.iloc[(first_bad_index + 1):(trip_index_to_move + 1)] = trips_to_shuffle.sample(frac=1).values

    df2["is_bad"] = ~(df2["origin"] == df2["destination"].shift())
    df2.at[df2.index[0], "is_bad"] = False

    return df2 if df2["original_order"].is_unique else df


def _shuffle_trips(df, time_period):
    trips_to_shuffle = df[df["depart"] == time_period]
    df2 = df.copy()
    df2.loc[trips_to_shuffle.index] = trips_to_shuffle.sample(frac=1).values
    df2["is_bad"] = ~(df2["origin"] == df2["destination"].shift())
    df2.at[df2.index[0], "is_bad"] = False
    return df2 if df2["original_order"].is_unique else df


def _sort_and_fix_sequences(trips, state):
    """Fix trip sequences with tour time window constraints"""
    # Initial sorting
    trips.sort_values(
        by=["person_id", "depart", "trip_num"],
        inplace=True
    )

    trips["original_order"] = np.arange(len(trips))

    # Calculate initial inconsistencies with vectorized operations
    topo_sort_mask = ((trips["destination"].shift() == trips["origin"]) |
                      (trips["person_id"].shift() != trips["person_id"]))
    trips.loc[:, "is_bad"] = ~topo_sort_mask
    initial_bad_count = trips["is_bad"].sum()

    if initial_bad_count == 0:
        logger.info("No inconsistent trips found - sequence already valid")
        trips.reset_index(inplace=True, drop=True)
        return trips

    logger.info(f"Pre-sorting left {initial_bad_count} inconsistent trips to fix")

    # Track statistics
    persons_total = 0
    persons_fixed_without_shifts = 0
    persons_fixed_with_shifts = 0
    total_trips_shifted = 0
    total_hours_shifted = 0

    # Find which persons have inconsistencies
    persons_with_bad_trips = trips[trips["is_bad"]]["person_id"].unique()

    # Process only those persons who have inconsistencies
    all_fixed_trips = []
    unchanged_mask = ~trips["person_id"].isin(persons_with_bad_trips)

    # Add all persons with no issues directly
    if unchanged_mask.any():
        all_fixed_trips.append(trips[unchanged_mask])

    # Process each person with inconsistencies
    for person_id in persons_with_bad_trips:
        persons_total += 1
        person_trips = trips[trips["person_id"] == person_id].copy()

        # Use numpy arrays for faster processing
        trip_indices = np.array(person_trips.index)
        origins = np.array(person_trips["origin"])
        destinations = np.array(person_trips["destination"])
        depart_times = np.array(person_trips["depart"])
        tour_starts = np.array(person_trips["tour_start"])
        tour_ends = np.array(person_trips["tour_end"])

        # Create a fast lookup dictionary for each time period
        time_to_trips = {}
        for i, time in enumerate(depart_times):
            if time not in time_to_trips:
                time_to_trips[time] = []
            time_to_trips[time].append(i)

        # Fix the sequence using the efficient approach
        fixed_person_trips, stats = _fix_sequence_fast(
            person_trips,
            trip_indices,
            origins,
            destinations,
            depart_times,
            time_to_trips,
            tour_starts,
            tour_ends
        )
        # Update our statistics
        if stats["status"] == "fixed_without_shifts":
            persons_fixed_without_shifts += 1
        elif stats["status"] == "fixed_with_shifts":
            persons_fixed_with_shifts += 1
            total_trips_shifted += stats["trips_shifted"]
            total_hours_shifted += stats["hours_shifted"]
        all_fixed_trips.append(fixed_person_trips)

    # Combine results
    result = pd.concat(all_fixed_trips) if all_fixed_trips else trips

    # Final validation
    topo_sort_mask = ((result["destination"].shift() == result["origin"]) |
                      (result["person_id"].shift() != result["person_id"]))
    result["is_bad"] = ~topo_sort_mask
    final_bad_count = result["is_bad"].sum()

    # Report results
    fixed_count = initial_bad_count - final_bad_count
    logger.info(
        f"Fixed {fixed_count} of {initial_bad_count} inconsistent trips ({fixed_count / initial_bad_count:.1%})")
    logger.info(f"Persons processed: {persons_total}")
    logger.info(
        f"  - Fixed without shifting times: {persons_fixed_without_shifts} ({persons_fixed_without_shifts / persons_total:.1%})")
    logger.info(
        f"  - Fixed by shifting times: {persons_fixed_with_shifts} ({persons_fixed_with_shifts / persons_total:.1%})")

    if persons_fixed_with_shifts > 0:
        avg_trips_shifted = total_trips_shifted / persons_fixed_with_shifts
        avg_hours_shifted = total_hours_shifted / total_trips_shifted if total_trips_shifted > 0 else 0
        logger.info(f"  - Average trips shifted per person: {avg_trips_shifted:.2f}")
        logger.info(f"  - Average hours shifted per trip: {avg_hours_shifted:.2f}")

    if final_bad_count > 0:
        logger.warning(f"Unable to fix {final_bad_count} inconsistent trips")

    # Cleanup and return
    if "is_bad" in result.columns:
        result = result.drop(columns=["is_bad"])

    result.reset_index(inplace=True, drop=True)
    mem.trace_memory_info("Just fixed trip sequence", force_garbage_collect=True, state=state)
    return result


def _create_final_plans(trips):
    # Select necessary columns
    cols = ["trip_id", "person_id", "tour_id", "departure_time", "purpose", "origin",
            "destination", "number_of_participants", "tour_mode", "trip_mode",
            "x", "y", "trip_dur_min", "trip_cost_dollars"]

    sorted_trips = trips[cols].sort_values(["person_id", "departure_time"]).reset_index()

    # Create return trips efficiently
    return_trip = sorted_trips.groupby("person_id").agg({
        "x": "first",
        "y": "first"
    }).reset_index()

    # Combine trips and create plan elements
    plans = pd.concat([sorted_trips, return_trip])
    plans["PlanElementIndex"] = plans.groupby("person_id").cumcount() * 2 + 1

    # Create activities
    plans["ActivityType"] = plans.groupby("person_id")["purpose"].shift(1).fillna("home")
    plans["ActivityElement"] = "activity"

    # Create legs efficiently
    legs = pd.DataFrame({
        "PlanElementIndex": plans.PlanElementIndex - 1,
        "person_id": plans.person_id
    })
    legs = legs[legs.PlanElementIndex != 0]
    legs["ActivityElement"] = "leg"

    # Combine and sort final plans
    final_plans = pd.concat([plans, legs]).sort_values(["person_id", "PlanElementIndex"])

    # Shift relevant columns
    shift_cols = ["trip_id", "trip_mode", "tour_id", "tour_mode", "trip_dur_min",
                  "trip_cost_dollars", "number_of_participants"]
    final_plans[shift_cols] = final_plans[shift_cols].shift()

    # Select final columns in desired order
    final_plans = final_plans[[
        "tour_id", "trip_id", "person_id", "number_of_participants",
        "tour_mode", "trip_mode", "PlanElementIndex", "ActivityElement",
        "ActivityType", "x", "y", "departure_time", "trip_dur_min",
        "trip_cost_dollars"
    ]]

    final_plans["tour_id"] = final_plans["tour_id"].fillna(-1).astype(np.int64)
    final_plans["trip_id"] = final_plans["trip_id"].fillna(-1).astype(np.int64)
    final_plans["person_id"] = final_plans["person_id"].fillna(-1).astype(np.int64)
    final_plans["trip_mode"] = final_plans["trip_mode"].astype(str)
    final_plans["tour_mode"] = final_plans["tour_mode"].astype(str)
    final_plans["ActivityType"] = final_plans["ActivityType"].astype(str)

    return final_plans
