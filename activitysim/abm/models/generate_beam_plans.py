import numpy as np
import pandas as pd
import random
from shapely import wkt
from shapely.geometry import Point, MultiPoint
import geopandas as gpd
import logging
import warnings

from activitysim.abm.models.util.expressions import skim_time_period_label
from activitysim.core import pipeline, orca
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
        ["DTIM", "TOTIVT", "IWAIT", "XWAIT", "WACC", "WAUX", "WEGR"],
        100.0,
    ),
    "WLK_HVY_DRV": (
        ["DTIM", "TOTIVT", "IWAIT", "XWAIT", "WACC", "WAUX", "WEGR"],
        100.0,
    ),
    "WLK_LRF_DRV": (
        ["DTIM", "TOTIVT", "IWAIT", "XWAIT", "WACC", "WAUX", "WEGR"],
        100.0,
    ),
    "DRV_LOC_WLK": (
        ["DTIM", "TOTIVT", "IWAIT", "XWAIT", "WACC", "WAUX", "WEGR"],
        100.0,
    ),
    "DRV_HVY_WLK": (
        ["DTIM", "TOTIVT", "IWAIT", "XWAIT", "WACC", "WAUX", "WEGR"],
        100.0,
    ),
    "DRV_LRF_WLK": (
        ["DTIM", "TOTIVT", "IWAIT", "XWAIT", "WACC", "WAUX", "WEGR"],
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
    trips['purpose'] = trips['purpose'].str.lower()
    rand_point_zones = {}
    for zone in zones[~(zones["geometry"].is_empty | zones["geometry"].isna())].TAZ:
        size = 500
        polygon = zones[zones.TAZ == zone].geometry
        points = sample_geoseries(polygon, size, overestimate=2)
        rand_point_zones[zone] = points

    def assignLoc(grp):
        zs = rand_point_zones[grp.iloc[0]['origin']]
        z = random.choice(zs)
        grp["origin_x"] = z[0]
        grp["origin_y"] = z[1]
        return grp

    trips = trips.groupby(['household_id', 'origin', 'purpose']).apply(assignLoc)

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


def generate_departure_times(trips, tours):
    trips["inbound"] = ~trips.outbound
    trips["tour_start"] = trips.tour_id.map(tours.start)
    trips["tour_end"] = trips.tour_id.map(tours.end)

    # TO DO: fractional times must respect the original order of trips!!!!
    df = (
        trips[
            [
                "person_id",
                "depart",
                "tour_start",
                "tour_end",
                "tour_id",
                "inbound",
                "trip_num",
            ]
        ]
        .reset_index()
        .drop_duplicates("trip_id")
    )
    df["frac"] = np.random.rand(
        len(df),
    )
    df.index.name = "og_df_idx"

    # Making sure trips within the hour are sequential
    ordered_trips = df.sort_values(
        by=[
            "person_id",
            "depart",
            "frac",
            "tour_start",
            "tour_end",
            "tour_id",
            "inbound",
            "trip_num",
        ]
    ).reset_index()
    df2 = df.sort_values(
        by=[
            "person_id",
            "depart",
            "tour_start",
            "tour_end",
            "tour_id",
            "inbound",
            "trip_num",
        ]
    ).reset_index()
    df2["fractional"] = ordered_trips.frac

    # Adding fractional to int hour
    df2["depart"] = np.round(df2["depart"] + df2["fractional"], 3)
    df2.set_index("og_df_idx", inplace=True)
    df2 = df2.reindex(df.index)
    df2.set_index("trip_id", inplace=True)
    df2 = df2.reindex(trips.index)
    return df2.depart


@inject.step()
def generate_beam_plans(trips, tours, persons, skim_dict, skim_stack):
    # Importing ActivitySim results
    trips = trips.to_frame()
    tours = tours.to_frame()
    persons = persons.to_frame()

    if orca.is_table("beam_geoms"):
        beam_geoms = pipeline.get_table("beam_geoms")
        beam_geoms["geometry"] = gpd.GeoSeries.from_wkt(beam_geoms["geometry"])
        zones = gpd.GeoDataFrame(beam_geoms, geometry="geometry", crs="EPSG:4326")
        zones.geometry = zones.geometry.buffer(0)
    else:
        land_use = pipeline.get_table("land_use").reset_index()

        # re-create zones shapefile
        land_use["geometry"] = land_use["geometry"].apply(wkt.loads)
        zones = gpd.GeoDataFrame(land_use, geometry="geometry", crs="EPSG:4326")
        zones.geometry = zones.geometry.buffer(0)

    orig_col = "origin"
    dest_col = "destination"

    odt_skim_stack_wrapper = skim_stack.wrap(
        left_key=orig_col, right_key=dest_col, skim_key="trip_period"
    )
    dot_skim_stack_wrapper = skim_stack.wrap(
        left_key=dest_col, right_key=orig_col, skim_key="trip_period"
    )
    od_skim_wrapper = skim_dict.wrap("origin", "destination")

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_wrapper,
    }
    trips["trip_period"] = skim_time_period_label(trips.depart)

    set_skim_wrapper_targets(trips, skims)

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
        mask = (trips["trip_mode_full"] == mode) & (trips['outbound'] == isOutbound)
        if mode in mode_to_time_columns:
            metrics, multiplier = mode_to_time_columns[mode]
            times = []
            for metric in metrics:
                if isOutbound:
                    val = odt_skim_stack_wrapper[f"{mode}_{metric}"].loc[mask] / multiplier
                else:
                    val = dot_skim_stack_wrapper[f"{mode}_{metric}"].loc[mask] / multiplier
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

    # augment trips table with attrs we need to generate plans
    trips = get_trip_coords(trips, zones, persons)
    trips["departure_time"] = generate_departure_times(trips, tours)
    trips["number_of_participants"] = trips["tour_id"].map(
        tours["number_of_participants"]
    )
    trips["tour_mode"] = trips["tour_id"].map(tours["tour_mode"])

    # trim trips table
    cols = [
        "person_id",
        "tour_id",
        "departure_time",
        "purpose",
        "origin",
        "destination",
        "number_of_participants",
        "tour_mode",
        "trip_mode",
        "x",
        "y",
    ]
    sorted_trips = (
        trips[cols].sort_values(["person_id", "departure_time"]).reset_index()
    )

    topo_sort_mask = (sorted_trips["destination"].shift() == sorted_trips["origin"]) | (
            sorted_trips["person_id"].shift() != sorted_trips["person_id"]
    )
    num_true, num_false = topo_sort_mask.value_counts().values

    if num_false > 0:
        num_trips = len(sorted_trips)
        pct_discontinuous_trips = np.round((num_false / num_trips) * 100, 1)
        logger.warning(
            "{0} of {1} ({2}%) of trips are topologically inconsistent "
            "after assigning departure times.".format(
                num_false, num_trips, pct_discontinuous_trips
            )
        )

    # Adding a new row for each unique person_id
    # this row will represent the returning trip
    return_trip = pd.DataFrame(
        sorted_trips.groupby("person_id").agg({"x": "first", "y": "first"}),
        index=sorted_trips.person_id.unique(),
    )

    plans = sorted_trips.append(return_trip)
    plans.reset_index(inplace=True)
    plans.person_id.fillna(plans["index"], inplace=True)

    # Creating the Plan Element activity Index
    # Activities have odd indices and legs (actual trips) will be even
    plans["PlanElementIndex"] = plans.groupby("person_id").cumcount() * 2 + 1
    plans = plans.sort_values(["person_id", "departure_time"]).reset_index(drop=True)

    # Shifting type one row down
    plans["ActivityType"] = (
        plans.groupby("person_id")["purpose"].shift(periods=1).fillna("Home")
    )
    plans["ActivityElement"] = "activity"

    # Creating legs (trips between activities)
    legs = pd.DataFrame(
        {"PlanElementIndex": plans.PlanElementIndex - 1, "person_id": plans.person_id}
    )
    legs = legs[legs.PlanElementIndex != 0]

    # Adding the legs to the main table
    final_plans = plans.append(legs).sort_values(["person_id", "PlanElementIndex"])
    final_plans.ActivityElement.fillna("leg", inplace=True)

    final_plans["trip_id"] = final_plans["trip_id"].shift()
    final_plans["trip_mode"] = final_plans["trip_mode"].shift()
    final_plans["tour_id"] = final_plans["tour_id"].shift()
    final_plans["tour_mode"] = final_plans["tour_mode"].shift()
    final_plans["number_of_participants"] = final_plans[
        "number_of_participants"
    ].shift()

    final_plans = final_plans[
        [
            "tour_id",
            "trip_id",
            "person_id",
            "number_of_participants",
            "tour_mode",
            "trip_mode",
            "PlanElementIndex",
            "ActivityElement",
            "ActivityType",
            "x",
            "y",
            "departure_time",
        ]
    ]

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
