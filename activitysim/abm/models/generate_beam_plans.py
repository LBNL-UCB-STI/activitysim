from pathlib import Path
from typing import Dict, List, Any

import networkx as nx
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
    for (person_id, origin, purpose), group in trips.groupby(["person_id", "origin", "purpose"], observed=True):
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
    trips['activity_code'] = np.zeros(len(trips), dtype=np.int8)
    trips.loc[trips['purpose'] == 'home', 'activity_code'] = np.int8(1)
    trips.loc[trips['purpose'] == 'work', 'activity_code'] = np.int8(2)

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


def identify_persons_with_problems(trips, pre=True):
    """
    Vectorized function to identify persons with problematic trip sequences
    """

    # Calculate initial inconsistencies with vectorized operations
    topo_sort_mask = ((trips["destination"].shift() == trips["origin"]) |
                      (trips["person_id"].shift() != trips["person_id"]))
    topologically_bad = ~topo_sort_mask
    repeated_activities = ((trips["activity_code"].shift() == trips["activity_code"]) &
                           (trips["person_id"].shift() == trips["person_id"]) &
                           (trips["destination"].shift() == trips["destination"]) &
                           (trips["activity_code"] > 0))
    topologically_bad_count = topologically_bad.sum()
    repeated_bad_count = repeated_activities.sum()

    if topologically_bad_count + repeated_bad_count == 0:
        logger.info("No inconsistent trips found - sequence already valid")
        trips.reset_index(inplace=True, drop=True)
        return trips

    if pre:
        logger.info(
            f"Pre-sorting left {topologically_bad_count} mismatched trips "
            f"and {repeated_bad_count} repeated destinations to fix")
    else:
        logger.info(
            f"Ended with {topologically_bad_count} mismatched trips "
            f"and {repeated_bad_count} repeated destinations to fix")

    # 1. Get persons with topological inconsistencies (if is_bad already calculated)
    problematic_persons_topo = trips.loc[topologically_bad]["person_id"].unique()
    problematic_persons_repeat = trips.loc[repeated_activities]["person_id"].unique()
    problematic_persons_total = set(problematic_persons_topo) | set(problematic_persons_repeat)

    logger.info(
        f"These are from {len(problematic_persons_topo)} and {len(problematic_persons_repeat)} persons, respectively,"
        f"leaving {len(problematic_persons_total)} total people with plans to fix.")

    return list(problematic_persons_total), topologically_bad | repeated_activities


def build_trip_sequence_graph(person_trips):
    """
    Build a directed graph representing all possible valid trip sequences.
    Topology is the hard constraint, while time sequence can be violated if necessary.
    Additionally enforces that paths should end at home locations.

    Parameters
    ----------
    person_trips : pd.DataFrame
        Trips for a single person

    Returns
    -------
    nx.DiGraph
        Graph with trips as nodes and edges representing valid sequences
    dict
        Information about home locations for constraint checking
    """
    G = nx.DiGraph()

    # Add all trips as nodes
    for idx, trip in person_trips.iterrows():
        G.add_node(idx, **trip.to_dict())

    # Identify home trips (purpose = "home")
    home_trips = person_trips[person_trips['purpose'] == 'home']

    # If no home trips exist, we'll relax this constraint
    if len(home_trips) == 0:
        home_locations = []
        logger.warning(f"Person {person_trips['person_id'].iloc[0]} has no home trips. Home constraint relaxed.")
    else:
        # Get all possible home locations (destinations of home trips)
        home_locations = home_trips['destination'].unique().tolist()

    # Create weighted edges between valid trip pairs
    for i, trip1 in person_trips.iterrows():
        for j, trip2 in person_trips.iterrows():
            if i == j:
                continue

            # Base edge weight - prefers original sequence order
            weight = abs(j - i) * 10  # Small penalty for deviating from original order

            # TOPOLOGICAL CONSTRAINT (HARD)
            # Skip if destination doesn't match origin (our primary hard constraint)
            if trip1['destination'] != trip2['origin']:
                continue  # No edge created - this is truly hard

            # TIME SEQUENCE CONSTRAINT (STRONG BUT VIOLATABLE)
            # Add high penalty if trip2 starts before trip1 ends (backward in time)
            trip1_end_time = trip1['depart'] + trip1['TOTAL_TIME_MINS'] / 60.0
            if trip2['depart'] < trip1_end_time:
                # Calculate how severe the violation is (in hours)
                time_violation = trip1_end_time - trip2['depart']
                weight += 500 + (time_violation * 100)  # Strong penalty, but not impossible

            # REPEATED ACTIVITIES CONSTRAINT (MODERATE)
            # Add penalty for repeated activities (soft constraint)
            if (trip1['activity_code'] > 0 and
                    trip1['activity_code'] == trip2['activity_code'] and
                    trip1['destination'] == trip2['destination']):
                weight += 1000  # Significant but lower than time violation

            # TOUR WINDOW CONSTRAINT (MODERATE)
            # Check if trip would be outside its allowed tour window
            tour_start2 = trip2['tour_start']
            tour_end2 = trip2['tour_end']

            # If this edge would force trip2 outside its window, add penalty
            if trip1_end_time > tour_end2:
                window_violation = trip1_end_time - tour_end2
                weight += 800 + (window_violation * 500)  # Significant penalty

            # DEPARTURE TIME SHIFT MINIMIZATION (WEAK)
            # Add small penalty for shifting departure times (minimize disruption)
            if trip1_end_time > trip2['depart']:
                # Penalize how much we need to shift trip2 later
                time_shift = trip1_end_time - trip2['depart']
                weight += time_shift * 100  # Scale based on hours shifted

            # Add the edge with computed weight
            G.add_edge(i, j, weight=weight)

    return G, home_locations


def fix_person_sequence_with_graph(person_trips):
    """
    Fix a single person's trip sequence using a graph-based approach,
    enforcing home start/end constraints.

    Parameters
    ----------
    person_trips : pd.DataFrame
        DataFrame containing trips for a single person

    Returns
    -------
    pd.DataFrame
        Fixed trips for the person
    dict
        Statistics about the fixing process
    """
    # Extract person data
    n_trips = len(person_trips)
    person_id = person_trips['person_id'].iloc[0]

    # Store original values - create a dictionary mapping from trip index to departure time
    # This avoids index alignment issues later
    original_departures = {idx: time for idx, time in zip(person_trips.index, person_trips['depart'])}

    # If only 0-1 trips, nothing to fix
    if n_trips <= 1:
        return person_trips, {'persons_fixed': 1, 'trips_shifted': 0, 'total_time_shifts': 0}

    # Build the graph for this person
    G, home_locations = build_trip_sequence_graph(person_trips)

    # Check if graph is empty (no valid paths possible)
    if G.number_of_edges() == 0:
        logger.warning(f"Person {person_id}: No valid sequence possible with current constraints")
        return person_trips, {'persons_failed': 1, 'trips_shifted': 0, 'total_time_shifts': 0}

    # Find valid chains that start and end at appropriate locations
    valid_chains = []

    # Get all trips that could be valid as first trip (origin is a home location)
    if home_locations:
        # If we have home locations, only consider trips starting from home
        potential_starts = [idx for idx in G.nodes() if idx in person_trips.index and
                            person_trips.loc[idx, 'origin'] in home_locations]
    else:
        # Otherwise, any trip could be a start
        potential_starts = [idx for idx in G.nodes() if idx in person_trips.index]

    # Get all trips that could be valid as last trip (purpose = "home")
    home_trip_indices = person_trips[person_trips['purpose'] == 'home'].index.tolist()
    potential_ends = home_trip_indices if home_trip_indices else [idx for idx in G.nodes() if idx in person_trips.index]

    # Try to find paths between all potential start/end pairs
    for start_idx in potential_starts:
        for end_idx in potential_ends:
            if start_idx == end_idx:
                continue

            try:
                # Check if there's a path from start to end
                paths = nx.all_simple_paths(G, start_idx, end_idx)
                these_paths = []

                for path in paths:
                    # Verify the first-last location constraint: first trip origin = last trip destination
                    first_trip_origin = person_trips.loc[path[0], 'origin']
                    last_trip_dest = person_trips.loc[path[-1], 'destination']

                    if len(path) == n_trips:
                        if first_trip_origin == last_trip_dest:
                            # This path forms a closed loop - perfect!
                            path_weight = nx.path_weight(G, path, weight='weight')
                            these_paths.append((path, path_weight))
                            if len(these_paths) >= 10:  # Stop after finding 10 complete paths
                                break
                these_paths.sort(key=lambda x: (-len(x[0]), x[1]))
                if these_paths:
                    valid_chains.append(these_paths[0])

            except nx.NetworkXNoPath:
                continue

    # If no valid chains were found, relax constraints and try again
    if not valid_chains:
        logger.warning(f"Person {person_id}: No valid chains with home constraints. Relaxing requirements.")

        # Try again without the home constraint
        for start_idx in G.nodes():
            if start_idx not in person_trips.index:
                continue

            for end_idx in G.nodes():
                if end_idx not in person_trips.index or start_idx == end_idx:
                    continue

                try:
                    # Check if there's a path from start to end
                    path = nx.shortest_path(G, start_idx, end_idx, weight='weight')
                    valid_chains.append((path, nx.path_weight(G, path,
                                                              weight='weight') + 3000))  # Penalty for not meeting home requirements
                except nx.NetworkXNoPath:
                    continue

    # If still no valid chains, return the original with a warning
    if not valid_chains:
        logger.warning(f"Person {person_id}: Could not find any valid chains even with relaxed constraints")
        return person_trips, {'persons_failed': 1, 'trips_shifted': 0, 'total_time_shifts': 0}

    # Find the longest chain with the lowest weight
    valid_chains.sort(key=lambda x: (-len(x[0]), x[1]))
    best_path, best_weight = valid_chains[0]

    # If the best path doesn't include all trips
    if len(best_path) < n_trips:
        logger.warning(f"Person {person_id}: Best path only includes {len(best_path)}/{n_trips} trips")

        # Get the trips in the path
        fixed_trips = person_trips.loc[best_path].copy()

        # Reinsert remaining trips with best effort
        remaining_trips = person_trips.loc[~person_trips.index.isin(best_path)]

        for idx, trip in remaining_trips.iterrows():
            fixed_trips = reinsert_trip_with_home_constraint(fixed_trips, trip, home_locations)

        # Calculate statistics
        trips_shifted = 0
        total_time_shifts = 0

        # Calculate how many trips were shifted and by how much
        for idx in fixed_trips.index:
            if idx in original_departures:  # Check if this index exists in original data
                orig_time = original_departures[idx]
                new_time = fixed_trips.loc[idx, 'depart']

                if orig_time != new_time:
                    trips_shifted += 1
                    total_time_shifts += abs(new_time - orig_time)

        return fixed_trips, {
            'persons_fixed': 1,
            'trips_shifted': trips_shifted,
            'total_time_shifts': total_time_shifts,
            'fixed_trip_count': len(best_path)
        }
    else:
        # Full fix - all trips included
        fixed_trips = person_trips.loc[best_path].copy()

        # Update departure times to ensure temporal consistency
        for i in range(1, len(best_path)):
            prev_idx = best_path[i - 1]
            curr_idx = best_path[i]

            # Get trips
            prev_trip = fixed_trips.loc[prev_idx]
            curr_trip = fixed_trips.loc[curr_idx]

            # Calculate minimum required departure time
            min_depart = prev_trip['depart'] + prev_trip['TOTAL_TIME_MINS'] / 60.0

            # Shift current trip if needed
            if curr_trip['depart'] < min_depart:
                fixed_trips.at[curr_idx, 'depart'] = min_depart

        # Calculate statistics
        trips_shifted = 0
        total_time_shifts = 0

        # Calculate how many trips were shifted and by how much
        for idx in fixed_trips.index:
            if idx in original_departures:  # Check if this index exists in original data
                orig_time = original_departures[idx]
                new_time = fixed_trips.loc[idx, 'depart']

                if orig_time != new_time:
                    trips_shifted += 1
                    total_time_shifts += abs(new_time - orig_time)

        return fixed_trips, {
            'persons_fixed': 1,
            'trips_shifted': trips_shifted,
            'total_time_shifts': total_time_shifts,
            'fixed_trip_count': len(best_path)
        }


def reinsert_trip_with_home_constraint(fixed_trips, trip, home_locations):
    """
    Attempt to reinsert a trip into an already fixed sequence
    considering home constraints

    Parameters
    ----------
    fixed_trips : pd.DataFrame
        DataFrame of already-fixed trips
    trip : Series
        The trip to reinsert
    home_locations : list
        List of locations considered as home

    Returns
    -------
    pd.DataFrame
        Updated fixed trips with the reinserted trip
    """
    # If empty, just return the trip as a single-row DataFrame
    if len(fixed_trips) == 0:
        return pd.DataFrame([trip])

    # Check if this is a home trip and we should prioritize placing it at the end
    is_home_trip = trip['purpose'] == 'home'

    if is_home_trip and fixed_trips.iloc[-1]['purpose'] != 'home':
        # This is a home trip and the current sequence doesn't end with home
        # Add it to the end
        result = pd.concat([fixed_trips, pd.DataFrame([trip])]).reset_index(drop=True)
        return result

    # Try to find the best position based on location matching
    best_pos = None
    best_score = float('inf')

    for i in range(len(fixed_trips) + 1):
        # Calculate score for inserting at position i
        score = 0

        # Check connection with previous trip (if not at the beginning)
        if i > 0:
            prev_trip = fixed_trips.iloc[i - 1]
            if prev_trip['destination'] != trip['origin']:
                score += 10000  # Major penalty for breaking the chain

            # Add penalty for time inconsistency
            prev_end_time = prev_trip['depart'] + prev_trip['TOTAL_TIME_MINS'] / 60.0
            if trip['depart'] < prev_end_time:
                score += 5000 + (prev_end_time - trip['depart']) * 1000

        # Check connection with next trip (if not at the end)
        if i < len(fixed_trips):
            next_trip = fixed_trips.iloc[i]
            if trip['destination'] != next_trip['origin']:
                score += 10000  # Major penalty for breaking the chain

            # Add penalty for time inconsistency
            trip_end_time = trip['depart'] + trip['TOTAL_TIME_MINS'] / 60.0
            if next_trip['depart'] < trip_end_time:
                score += 5000 + (trip_end_time - next_trip['depart']) * 1000

        # Special scoring for maintaining home trip at the end
        if i == len(fixed_trips) and fixed_trips.iloc[-1]['purpose'] == 'home':
            score += 8000  # Penalty for displacing a home trip from the end

        # Special handling for home trips - prefer at the end
        if is_home_trip and i < len(fixed_trips):
            score += 500  # Small penalty for placing home trips before the end

        # Update best position if this is better
        if score < best_score:
            best_score = score
            best_pos = i

    # Insert at the best position (or at the end if no good position found)
    if best_pos is None:
        best_pos = len(fixed_trips)

    result = pd.concat([
        fixed_trips.iloc[:best_pos],
        pd.DataFrame([trip]),
        fixed_trips.iloc[best_pos:]
    ]).reset_index(drop=True)

    return result



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
        processed_chunk['activity_code'] = processed_chunk['activity_code'].astype(np.int8)
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

    # Find which persons have inconsistencies
    persons_with_bad_trips, bad_trips = identify_persons_with_problems(trips)
    logger.info(f"Found {len(persons_with_bad_trips)} persons with inconsistent trips")
    trips['is_bad'] = bad_trips

    initial_bad_count = bad_trips.sum()

    # Process only those persons who have inconsistencies
    all_fixed_trips = []
    unchanged_mask = ~trips["person_id"].isin(persons_with_bad_trips)

    # Add all persons with no issues directly
    if unchanged_mask.any():
        all_fixed_trips.append(trips[unchanged_mask])

    stats = {
        'persons_total': len(persons_with_bad_trips),
        'persons_fixed': 0,
        'persons_partial': 0,
        'persons_failed': 0,
        'trips_shifted': 0,
        'total_time_shifts': 0,
        'fixed_trip_count': 0
    }

    # Process each person with inconsistencies
    for person_id in persons_with_bad_trips:
        person_trips = trips[trips["person_id"] == person_id].copy()

        fixed_trips, person_stats = fix_person_sequence_with_graph(person_trips)

        # Update statistics
        for key, value in person_stats.items():
            if key in stats:
                stats[key] += value

        all_fixed_trips.append(fixed_trips)

    # Combine results
    result = pd.concat(all_fixed_trips) if all_fixed_trips else trips
    result['activity_code'] = result['activity_code'].astype(np.int8)

    # Final validation and cleanup
    final_problem_persons, final_is_bad = identify_persons_with_problems(result)
    final_bad_count = final_is_bad.sum()

    # Calculate improvement
    fixed_count = initial_bad_count - final_bad_count
    fix_percent = fixed_count / initial_bad_count * 100 if initial_bad_count > 0 else 100

    logger.info(f"Fixed {fixed_count} of {initial_bad_count} inconsistencies ({fix_percent:.1f}%)")
    logger.info(f"Persons with remaining problems: {len(final_problem_persons)}")

    # Detailed statistics
    if stats['persons_total'] > 0:
        logger.info(f"Fully fixed persons: {stats['persons_fixed']} "
                    f"({stats['persons_fixed'] / stats['persons_total'] * 100:.1f}%)")
        logger.info(f"Trips shifted: {stats['trips_shifted']}")
        if stats['trips_shifted'] > 0:
            logger.info(f"Average shift amount: {stats['total_time_shifts'] / stats['trips_shifted']:.2f} hours")

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
