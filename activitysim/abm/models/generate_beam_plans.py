import numpy as np
import pandas as pd
import random
from shapely import wkt
from shapely.geometry import Point, MultiPoint
import geopandas as gpd
import logging
import warnings

from activitysim.abm.models.util import expressions
from activitysim.abm.models.util.expressions import skim_time_period_label
from activitysim.core import pipeline, orca, config
from activitysim.core import inject
from activitysim.core.simulate import set_skim_wrapper_targets

logger = logging.getLogger("activitysim")
warnings.filterwarnings("ignore", "GeoSeries.isna", UserWarning)

mode_to_spd = {
    "WALK": 1.25,
    "BIKE": 3.0,
}
fallback_car_mode = "SOV"

mode_to_time_columns = {
    "WLK_LOC_WLK": (["TOTIVT", "IWAIT", "XWAIT", "WACC", "WAUX", "WEGR"], 100.0),
    "HOV2TOLL": (["TIME"], 1.0),
    "HOV3": (["TIME"], 1.0),
    "SOV": (["TIME"], 1.0),
    "SOVTOLL": (["TIME"], 1.0),
    "WLK_HVY_WLK": (["TOTIVT", "IWAIT", "XWAIT", "WACC", "WAUX", "WEGR"], 100.0),
    "HOV2": (["TIME"], 1.0),
    "HOV3TOLL": (["TIME"], 1.0),
    "WLK_LRF_WLK": (["TOTIVT", "IWAIT", "XWAIT", "WACC", "WAUX", "WEGR"], 100.0),
    "WLK_COM_WLK": (["TOTIVT", "IWAIT", "XWAIT", "WACC", "WAUX", "WEGR"], 100.0),
    "WLK_LOC_DRV": (
        ["DTIM", "TOTIVT", "IWAIT", "XWAIT", "WAUX"],
        100.0,
    ),
    "WLK_HVY_DRV": (
        ["DTIM", "TOTIVT", "IWAIT", "XWAIT", "WAUX"],
        100.0,
    ),
    "WLK_LRF_DRV": (
        ["DTIM", "TOTIVT", "IWAIT", "XWAIT", "WAUX"],
        100.0,
    ),
    "DRV_LOC_WLK": (
        ["DTIM", "TOTIVT", "IWAIT", "XWAIT", "WAUX"],
        100.0,
    ),
    "DRV_HVY_WLK": (
        ["DTIM", "TOTIVT", "IWAIT", "XWAIT", "WAUX"],
        100.0,
    ),
    "DRV_LRF_WLK": (
        ["DTIM", "TOTIVT", "IWAIT", "XWAIT", "WAUX"],
        100.0,
    ),
}


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
    samples = np.random.uniform(
        (min_x, min_y), (max_x, max_y), (int(size / ratio * overestimate), 2)
    )
    multipoint = MultiPoint(samples)
    multipoint = multipoint.intersection(polygon)
    samples = np.array(multipoint)
    return samples[np.random.choice(len(samples), size)]


def get_trip_coords(trips, zones, persons, size=500):
    # Generates random points within each zone for zones
    # that are not empty geometries (i.e. contain no blocks)
    trips["purpose"] = trips["purpose"].str.lower()
    rand_point_zones = {}
    for zone in zones[~(zones["geometry"].is_empty | zones["geometry"].isna())].TAZ:
        size = 500
        polygon = zones[zones.TAZ == zone].geometry
        points = sample_geoseries(polygon, size, overestimate=2)
        rand_point_zones[zone] = points

    def assignLoc(grp):
        zs = rand_point_zones[grp.iloc[0]["origin"]]
        z = random.choice(zs)
        grp["origin_x"] = z[0]
        grp["origin_y"] = z[1]
        return grp

    trips = trips.groupby(["household_id", "origin", "purpose"]).apply(assignLoc)

    # retain home coords from urbansim data bc they will typically be
    # higher resolution than zone, so we don't need the semi-random coords
    trips = pd.merge(
        trips, persons[["home_x", "home_y"]], left_on="person_id", right_index=True
    )
    trips["origin_purpose"] = (
        trips.groupby("person_id")["purpose"].shift(periods=1).fillna("Home")
    )
    trips["x"] = trips.origin_x.where(trips.origin_purpose != "Home", trips.home_x)
    trips["y"] = trips.origin_y.where(trips.origin_purpose != "Home", trips.home_y)

    return trips


def generatePersonStartTimes(df):
    df["mustEndBy"] = np.minimum(df["depart"].shift(-1).fillna(25), df["tour_end"]) + 1
    return df


def generate_departure_times(trips, tours):
    # TO DO: fractional times must respect the original order of trips!!!!
    ordered_trips2 = (
        trips[
            [
                "person_id",
                "depart",
                "tour_start",
                "tour_end",
                "tour_id",
                "inbound",
                "trip_num",
                "TOTAL_TIME_MINS",
            ]
        ]
        .reset_index()
        .drop_duplicates("trip_id")
    )
    ordered_trips2["frac"] = np.random.rand(
        len(ordered_trips2),
    )
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
        df["gapAfterTrip"] = -(
                (df["newStartTime"] + df["TOTAL_TIME_MINS"] / 60.0)
                - df["newStartTime"].shift(-1).fillna(100).values
        )
        i = 0
        while not np.all(df["gapAfterTrip"].values >= 0):
            df.loc[
                ~(df["gapAfterTrip"].shift(1).fillna(100).values > 0), "newStartTime"
            ] -= (
                df.loc[~(df["gapAfterTrip"].values > 0), "gapAfterTrip"]
                .fillna(0.0)
                .values
            )
            df["gapAfterTrip"] = -(
                    (df["newStartTime"] + df["TOTAL_TIME_MINS"] / 60.0)
                    - df["newStartTime"].shift(-1).fillna(100).values
            )
            i += 1
            if i > 15:
                raise ValueError
        return df

    def process(df):
        df["mustFinishWithinHour"] = (
                df["depart"] >= df["depart"].shift(-1).fillna(24) - 1
        )
        df = df.groupby("depart").apply(getTotalTime)
        return df

    df2 = ordered_trips2.groupby(["person_id"]).apply(process)
    df2.set_index("trip_id", inplace=True)
    df2 = df2.reindex(trips.index)
    return df2.newStartTime.rename("depart")


def label_trip_modes(trips, skims):
    odt_skim_stack_wrapper = skims["odt_skims"]
    dot_skim_stack_wrapper = skims["dot_skims"]
    od_skim_wrapper = skims["od_skims"]

    trips["totalTime"] = -1.0
    trips["trip_mode_full"] = trips["trip_mode"].replace(
        {
            "SHARED2PAY": "HOV2TOLL",
            "SHARED3PAY": "HOV3TOLL",
            "SHARED2FREE": "HOV2",
            "SHARED3FREE": "HOV3",
            "DRIVEALONEFREE": "SOV",
            "DRIVEALONEPAY": "SOVTOLL",
        }
    )
    trips.loc[trips.trip_mode.str.startswith("WALK_"), "trip_mode_full"] = (
            trips.loc[
                trips.trip_mode.str.startswith("WALK_"), "trip_mode_full"
            ].str.replace("WALK", "WLK")
            + "_WLK"
    )
    trips.loc[
        trips.trip_mode.str.startswith("DRIVE_") & trips.outbound, "trip_mode_full"
    ] = (
            trips.loc[
                trips.trip_mode.str.startswith("DRIVE_") & trips.outbound, "trip_mode_full"
            ].str.replace("DRIVE", "DRV")
            + "_WLK"
    )
    trips.loc[
        trips.trip_mode.str.startswith("DRIVE_") & ~trips.outbound, "trip_mode_full"
    ] = (
            trips.loc[
                trips.trip_mode.str.startswith("DRIVE_") & ~trips.outbound, "trip_mode_full"
            ].str.replace("DRIVE", "WLK")
            + "_DRV"
    )
    for (mode, isOutbound), subdf in trips.groupby(["trip_mode_full", "outbound"]):
        mask = (trips["trip_mode_full"] == mode) & (trips["outbound"] == isOutbound)
        if mode in mode_to_time_columns:
            metrics, multiplier = mode_to_time_columns[mode]
            times = []
            for metric in metrics:
                try:
                    if isOutbound:
                        val = (
                                odt_skim_stack_wrapper[f"{mode}_{metric}"].loc[mask]
                                / multiplier
                        )
                    else:
                        val = (
                                dot_skim_stack_wrapper[f"{mode}_{metric}"].loc[mask]
                                / multiplier
                        )
                except AssertionError:
                    val = odt_skim_stack_wrapper["SOV_TIME"].loc[mask] * 0.0
                times.append(val)
            look = pd.concat(times, axis=1)
            trips.loc[mask, "totalTime"] = look.sum(axis=1)
        elif mode in mode_to_spd:
            spd = mode_to_spd[mode]
            dist = od_skim_wrapper[f"DIST{mode}"].loc[mask]
            trips.loc[mask, "totalTime"] = dist / spd
        else:
            fallback_time = odt_skim_stack_wrapper["SOV_TIME"].loc[mask]
            trips.loc[mask, "totalTime"] = fallback_time * 1.1
    return trips


@inject.step()
def generate_beam_plans(trips, tours, persons, skim_dict, skim_stack, chunk_size, trace_hh_id, locutor):
    # Convert to frames only once and work in-place where possible
    trips = trips.to_frame()
    tours = tours.to_frame()
    persons = persons.to_frame()

    # Load configurations
    model_settings = config.read_model_settings('generate_beam_plans.yaml')
    constants = config.get_model_constants(model_settings)
    # - run preprocessor to annotate choosers
    preprocessor_settings = model_settings.get('preprocessor', None)
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

    if orca.is_table("beam_geoms"):
        zones = _process_beam_geoms(pipeline.get_table("beam_geoms"))
    else:
        zones = _process_land_use(pipeline.get_table("land_use"))

    # Setup skims
    trips["trip_period"] = skim_time_period_label(trips.depart)
    skims = _setup_skims(skim_stack, skim_dict)
    set_skim_wrapper_targets(trips, skims)

    constants = config.get_model_constants(model_settings)

    expressions.annotate_preprocessors(
        trips, constants, skims,
        model_settings, None)

    # Modify trips dataframe in-place where possible
    _annotate_trips(trips, tours)

    # Sort trips and fix sequences
    trips = _sort_and_fix_sequences(trips)

    # Get coordinates and times
    trips = get_trip_coords(trips, zones, persons)
    trips["departure_time"] = generate_departure_times(trips, tours)

    # Add tour information efficiently using map
    trips["number_of_participants"] = trips["tour_id"].map(tours["number_of_participants"])
    trips["tour_mode"] = trips["tour_id"].map(tours["tour_mode"])
    trips.rename(columns={
        "TOTAL_TIME_MINS": "trip_dur_min",
        "TOTAL_COST_DOLLARS": "trip_cost_dollars"
    }, inplace=True)

    # Create final plans more efficiently
    return _create_final_plans(trips)


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


def _setup_skims(skim_stack, skim_dict):
    orig_col, dest_col = "origin", "destination"
    return {
        "odt_skims": skim_stack.wrap(left_key=orig_col, right_key=dest_col, skim_key="trip_period"),
        "dot_skims": skim_stack.wrap(left_key=dest_col, right_key=orig_col, skim_key="trip_period"),
        "od_skims": skim_dict.wrap("origin", "destination")
    }


def _annotate_trips(trips, tours):
    trips["inbound"] = ~trips.outbound
    trips["tour_start"] = trips.tour_id.map(tours.start)
    trips["tour_end"] = trips.tour_id.map(tours.end)
    trips["isAtWork"] = trips.purpose == "atwork"

    # Handle actuallyInbound calculation
    trips["actuallyInbound"] = trips["inbound"].copy()
    mask_work = (trips.primary_purpose == "work") & (trips.purpose != "Home")
    trips.loc[mask_work, "actuallyInbound"] = ~trips.loc[mask_work, "inbound"]
    mask_atwork = (trips.purpose == "atwork")
    trips.loc[mask_atwork, "actuallyInbound"] = ~trips.loc[mask_atwork, "inbound"]


def _fix_trip_sequence(df):
    bad_indices = np.nonzero(df.is_bad.values)[0]
    if len(bad_indices) == 0:
        return df

    first_bad_index = bad_indices[0]
    dest_last_good = df.iloc[first_bad_index - 1]["destination"]
    time_period = df.iloc[first_bad_index]["depart"]

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
    df2["is_bad"].iloc[0] = False

    return df2 if df2["original_order"].is_unique else df


def _shuffle_trips(df, time_period):
    trips_to_shuffle = df[df["depart"] == time_period]
    df2 = df.copy()
    df2.loc[trips_to_shuffle.index] = trips_to_shuffle.sample(frac=1).values
    df2["is_bad"] = ~(df2["origin"] == df2["destination"].shift())
    df2["is_bad"].iloc[0] = False
    return df2 if df2["original_order"].is_unique else df


def _sort_and_fix_sequences(trips):
    trips.reset_index(inplace=True)
    trips["original_order"] = np.arange(len(trips))

    # Initial sorting
    trips.sort_values(
        by=['person_id', 'depart', 'tour_start', 'tour_end', 'tour_id', 'inbound', 'trip_num'],
        inplace=True
    )

    # Fix sequences
    topo_sort_mask = ((trips["destination"].shift() == trips["origin"]) |
                      (trips["person_id"].shift() != trips["person_id"]))
    trips["is_bad"] = ~topo_sort_mask

    iteration = 0
    while (trips["is_bad"].sum() > 0) and (iteration < 50):
        logger.info(f"Before rearranging: {trips.is_bad.sum()} trips")
        bad_person_ids = trips.loc[~topo_sort_mask, "person_id"]
        bad_plans = trips.loc[trips.person_id.isin(bad_person_ids)]

        fixed_plans = (bad_plans.groupby("person_id")
                       .apply(_fix_trip_sequence)
                       .reset_index(drop=True))

        fixed_plans.index = bad_plans.index
        trips.loc[fixed_plans.index] = fixed_plans

        topo_sort_mask = ((trips["destination"].shift() == trips["origin"]) |
                          (trips["person_id"].shift() != trips["person_id"]))
        trips["is_bad"] = ~topo_sort_mask
        logger.info(f"After: {trips.is_bad.sum()} trips")
        iteration += 1

    trips.set_index("trip_id", inplace=True)
    return trips


def _create_final_plans(trips):
    # Select necessary columns
    cols = ["person_id", "tour_id", "departure_time", "purpose", "origin",
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
    plans["ActivityType"] = plans.groupby("person_id")["purpose"].shift(1).fillna("Home")
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

    # save back to pipeline
    pipeline.replace_table("plans", final_plans)

#     # summary stats
#     input_cars_per_hh = np.round(
#         households['VEHICL'].sum() / len(households), 2)
#     simulated_cars_per_hh = np.round(
#         households['auto_ownership'].sum() / len(households), 2)
#     logger.warning(
#         "AUTO OWNERSHIP -- input: {0} cars/hh // output: {1} cars/hh".format(
#             input_cars_per_hh, simulated_cars_per_hh))

#     trips['number_of_participants'] = trips['tour_id'].map(
#         tours['number_of_participants'])
#     trips['mode_type'] = 'drive'
#     transit_modes = ['COM', 'EXP', 'HVY', 'LOC', 'LRF', 'TRN']
#     active_modes = ['WALK', 'BIKE']
#     trips.loc[
#         trips['trip_mode'].str.contains('|'.join(transit_modes)),
#         'mode_type'] = 'transit'
#     trips.loc[trips['trip_mode'].isin(active_modes), 'mode_type'] = 'active'
#     expanded_trips = trips.loc[
#         trips.index.repeat(trips['number_of_participants'])]
#     mode_shares = expanded_trips[
#         'mode_type'].value_counts() / len(expanded_trips)
#     mode_shares = np.round(mode_shares * 100, 1)
#     mode_shares.keys()
#     logger.warning(
#         "MODE SHARES -- drive: {0}% // transit: {1}% // active: {2}%".format(
#             mode_shares['drive'], mode_shares['transit'],
#             mode_shares['active']))
