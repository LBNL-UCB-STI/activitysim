import os
import pandas as pd
import numpy as np
import geopandas as gpd
import orca
from shapely.geometry import Polygon
from shapely import wkt
from h3 import h3
from urbansim.utils import misc
import requests
import logging
from tqdm import tqdm
import time
import boto3
from botocore.exceptions import ClientError

from activitysim.core import config
from activitysim.core import inject


logger = logging.getLogger(__name__)


def get_zone_geoms_from_h3(h3_ids):
    polygon_shapes = []
    for zone in h3_ids:
        boundary_points = h3.h3_to_geo_boundary(h3_address=zone, geo_json=True)
        shape = Polygon(boundary_points)
        polygon_shapes.append(shape)

    return polygon_shapes


def get_county_block_geoms(state_fips, county_fips):

    base_url = (
        'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/'
        'Tracts_Blocks/MapServer/12/query?where=STATE%3D{0}+and+COUNTY%3D{1}'
        '&outFields=GEOID%2CSTATE%2CCOUNTY%2CTRACT%2CBLKGRP%2CBLOCK%2CCENTLAT'
        '%2CCENTLON&outSR=%7B"wkid"+%3A+4326%7D&f=pjson')
    url = base_url.format(state_fips, county_fips)
    result = requests.get(url)
    features = result.json()['features']
    if len(features) >= 100000:
        raise RuntimeError("too many blocks in county to query at once!")
    else:
        df = pd.DataFrame()
        for feature in features:
            tmp = pd.DataFrame([feature['attributes']])
            tmp['geometry'] = Polygon(
                feature['geometry']['rings'][0],
                feature['geometry']['rings'][1:])
            df = pd.concat((df, tmp))
        gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
        return gdf


def get_taz_from_block_geoms(blocks_gdf, zones_gdf, local_crs):

    logger.info("Assigning blocks to TAZs!")

    # df to store GEOID to TAZ results
    block_to_taz_results = pd.DataFrame()

    # ignore empty geoms
    zones_gdf = zones_gdf[~zones_gdf['geometry'].is_empty]

    # convert to meter-based proj
    zones_gdf = zones_gdf.to_crs(local_crs)
    blocks_gdf = blocks_gdf.to_crs(local_crs)

    zones_gdf['zone_area'] = zones_gdf.geometry.area

    # assign blocks to zone with a spatial within query
    within = gpd.sjoin(
        blocks_gdf, zones_gdf.reset_index(), how='inner', op='within')

    # when a block falls within multiple (overlapping) zones,
    # assign it to the zone with the smallest area
    within = within.sort_values(['GEOID', 'zone_area'])
    within = within.drop_duplicates('GEOID', keep='first')

    # add to results df
    block_to_taz_results = pd.concat((
        block_to_taz_results, within[['GEOID', 'TAZ']]))

    # assign remaining blocks based on a spatial intersection
    unassigned_mask = ~blocks_gdf['GEOID'].isin(block_to_taz_results['GEOID'])
    intx = gpd.overlay(
        blocks_gdf[unassigned_mask], zones_gdf.reset_index(),
        how='intersection')

    # assign zone ID's to blocks based on max area of intersection
    intx['intx_area'] = intx['geometry'].area
    intx = intx.sort_values(['GEOID', 'intx_area'], ascending=False)
    intx = intx.drop_duplicates('GEOID', keep='first')

    # add to results df
    block_to_taz_results = pd.concat((
        block_to_taz_results, intx[['GEOID', 'TAZ']]))

    # assign zone ID's to remaining blocks based on shortest
    # distance between block and zone centroids
    unassigned_mask = ~blocks_gdf['GEOID'].isin(block_to_taz_results['GEOID'])

    if any(unassigned_mask):

        blocks_gdf['geometry'] = blocks_gdf['geometry'].centroid
        zones_gdf['geometry'] = zones_gdf['geometry'].centroid

        all_dists = blocks_gdf.loc[unassigned_mask, 'geometry'].apply(
            lambda x: zones_gdf['geometry'].distance(x))

        nearest = all_dists.idxmin(axis=1).reset_index()
        nearest.columns = ['blocks_idx', 'TAZ']
        nearest.set_index('blocks_idx', inplace=True)
        nearest['GEOID'] = blocks_gdf.reindex(nearest.index)['GEOID']

        block_to_taz_results = pd.concat((
            block_to_taz_results, nearest[['GEOID', 'TAZ']]))

    return block_to_taz_results.set_index('GEOID')['TAZ']


def get_taz_from_points(df, zones_gdf, local_crs):
    '''
    Assigns the gdf index (TAZ ID) for each index in df
    Input:
    - df columns names x, and y. The index is the ID of the point feature.
    - zones_gdf: GeoPandas GeoDataFrame with TAZ as index, geometry, area.

    Output:
        A series with df index and corresponding gdf id
    '''
    logger.info("Assigning TAZs to {0}".format(df.index.name))
    df = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326")
    assert df.is_valid.all()
    zones_gdf.geometry.crs = "EPSG:4326"

#     # convert to meters-based local crs (#Not sure why I would need to conver to meters)
#     df = df.to_crs(local_crs)
#     zones_gdf = zones_gdf.to_crs(local_crs)

    # Spatial join
#     intx = gpd.sjoin(
#         df.reset_index(), zones_gdf.reset_index(), how='left', op='intersects')

    intx = gpd.sjoin(
        df, zones_gdf.reset_index(), how='left', op='intersects')

    ## THIS MAY NO LONGER BE NEEDED BECAUSE WE ARE NOT DOING H3 AREAS
    # Drop duplicates and keep the one with the smallest H3 area
#     intx['intx_area'] = intx['geometry'].area
#     intx = intx.sort_values('intx_area')
#     intx.drop_duplicates(subset=[df.index.name], keep='first', inplace=True)
#     intx.set_index(df.index.name, inplace=True)
    df['TAZ'] = intx['TAZ'].reindex(df.index)

#     # Check if there is any unassigined object
#     unassigned_mask = pd.isnull(df['TAZ'])
#     if any(unassigned_mask):

#         zones_gdf['geometry'] = zones_gdf['geometry'].centroid
#         all_dists = df.loc[unassigned_mask, 'geometry'].apply(
#             lambda x: zones_gdf['geometry'].distance(x))

#         df.loc[unassigned_mask, 'TAZ'] = all_dists.idxmin(
#             axis=1).values

    return df['TAZ']


def get_full_time_enrollment(state_fips):
    base_url = (
        'https://educationdata.urban.org/api/v1/'
        '{t}/{so}/{e}/{y}/{l}/?{f}&{s}&{r}&{cl}&{ds}&{fips}')
    levels = ['undergraduate', 'graduate']
    enroll_list = []
    for level in levels:
        level_url = base_url.format(
            t='college-university', so='ipeds', e='fall-enrollment',
            y='2015', l=level, f='ftpt=1', s='sex=99',
            r='race=99', cl='class_level=99', ds='degree_seeking=99',
            fips='fips={0}'.format(state_fips))

        enroll_result = requests.get(level_url)
        enroll = pd.DataFrame(enroll_result.json()['results'])
        enroll = enroll[['unitid', 'enrollment_fall']].rename(
            columns={'enrollment_fall': level})
        enroll.set_index('unitid', inplace=True)
        enroll_list.append(enroll)

    full_time = pd.concat(enroll_list, axis=1)
    full_time['full_time'] = full_time['undergraduate'] + full_time['graduate']
    s = full_time.full_time
    assert s.index.name == 'unitid'

    return s


def get_part_time_enrollment(state_fips):
    base_url = (
        'https://educationdata.urban.org/api/v1/'
        '{t}/{so}/{e}/{y}/{l}/?{f}&{s}&{r}&{cl}&{ds}&{fips}')
    levels = ['undergraduate', 'graduate']
    enroll_list = []
    for level in levels:
        level_url = base_url.format(
            t='college-university', so='ipeds', e='fall-enrollment',
            y='2015', l=level, f='ftpt=2', s='sex=99',
            r='race=99', cl='class_level=99', ds='degree_seeking=99',
            fips='fips={0}'.format(state_fips))

        enroll_result = requests.get(level_url)
        enroll = pd.DataFrame(enroll_result.json()['results'])
        enroll = enroll[['unitid', 'enrollment_fall']].rename(
            columns={'enrollment_fall': level})
        enroll.set_index('unitid', inplace=True)
        enroll_list.append(enroll)

    part_time = pd.concat(enroll_list, axis=1)
    part_time['part_time'] = part_time['undergraduate'] + part_time['graduate']
    s = part_time.part_time
    assert s.index.name == 'unitid'

    return s


def exists_on_s3(s3_client, bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        return int(e.response['Error']['Code']) != 404
    return True


# ** define injectables we'll use over and over **
@orca.injectable()
def county_codes(blocks):
    county_codes = blocks.index.str.slice(2, 5).unique().values
    return county_codes


@orca.injectable()
def state_fips():
    return config.setting('state_fips')


@orca.injectable()
def data_dir():
    return inject.get_injectable('data_dir')


@orca.injectable()
def settings():
    return inject.get_injectable('settings')


@orca.injectable()
def local_crs():
    return config.setting('local_crs')


@orca.injectable(cache=True)
def store(data_dir, settings):

    data_store_path = os.path.join(data_dir, settings['usim_data_store'])
    if not os.path.exists(data_store_path):

        if not settings.get('remote_data_full_path', False):
            logger.info("Creating remote data path from default parameters.")
            bucket = settings.get('bucket_name')
            scenario = settings.get('scenario')
            year = settings.get('year')
            usim_data_store = settings.get('usim_data_store')
            if not isinstance(year, str):
                year = str(year)

            remote_data_full_path = os.path.join(
                bucket, 'input', scenario, year, usim_data_store)

        else:
            remote_data_full_path = settings.get('remote_data_full_path')
        print(remote_data_full_path)

        logger.info("Downloading UrbanSim data from s3 at {0}!".format(
            remote_data_full_path))
        s3 = boto3.client('s3')
        bucket = remote_data_full_path.split('/')[0]
        key = os.path.join(*remote_data_full_path.split('/')[1:])
        if not exists_on_s3(s3, bucket, key):
            raise KeyError(
                "No remote model data found using default path. See "
                "simuation.py --help or configs/settings.yaml "
                "for more ideas.")
        with open(data_store_path, 'wb') as f:
            s3.download_fileobj(bucket, key, f)

    logger.info("Loading UrbanSim input data from disk!")
    store = pd.HDFStore(data_store_path)

    return store


# ** 1. CREATE NEW TABLES **
@orca.table(cache=True)
def usim_households(store):
    households = store['/households']
    return households


@orca.table(cache=True)
def blocks(store):
    blocks = store['/blocks']
    return blocks


@orca.table(cache=True)
def usim_persons(store, blocks, usim_households):
    persons = store['/persons']
    persons_w_res_blk = pd.merge(
        persons, usim_households.to_frame(columns=['block_id']),
        left_on='household_id', right_index=True)
    persons_w_xy = pd.merge(
        persons_w_res_blk, blocks.to_frame(columns=['x', 'y']),
        left_on='block_id', right_index=True)
    persons['home_x'] = persons_w_xy['x']
    persons['home_y'] = persons_w_xy['y']

    del persons_w_res_blk
    del persons_w_xy

    return persons


@orca.table(cache=True)
def jobs(store, blocks, local_crs):

    jobs_df = store['/jobs']
    jobs_cols = jobs_df.columns

    # make sure jobs are only assigned to blocks with land area > 0
    # so that employment density distributions don't contain Inf/NaN
    blocks_df = blocks.to_frame(columns=['square_meters_land'])
    jobs_df['square_meters_land'] = blocks_df.reindex(
        jobs_df['block_id'])['square_meters_land'].values
    jobs_w_no_land = jobs_df[jobs_df['square_meters_land'] == 0]
    blocks_to_reassign = jobs_w_no_land['block_id'].unique()

    if len(blocks_to_reassign) > 0:

        logger.info("Reassigning jobs out of blocks with no land area!")
        blocks_gdf = orca.get_table(
            'block_geoms').to_frame().set_index('GEOID')
        blocks_gdf['square_meters_land'] = blocks[
            'square_meters_land'].reindex(blocks_gdf.index)
        blocks_gdf = blocks_gdf.to_crs(local_crs)

        for block_id in tqdm(
                blocks_to_reassign,
                desc="Redistributing jobs from blocks:"):

            candidate_mask = (
                blocks_gdf.index.values != block_id) & (
                blocks_gdf['square_meters_land'] > 0)
            new_block_id = blocks_gdf[candidate_mask].distance(
                blocks_gdf.loc[block_id, 'geometry']).idxmin()

            jobs_df.loc[
                jobs_df['block_id'] == block_id, 'block_id'] = new_block_id

        # update data store with new block_id's
        logger.info(
            "Storing jobs table with updated block IDs to disk "
            "in .h5 datastore!")
        store['jobs'] = jobs_df[jobs_cols]

    else:
        logger.info("No block IDs to reassign in the jobs table!")

    return jobs_df


@orca.table(cache=True)
def block_geoms(data_dir, state_fips, county_codes):

    all_block_geoms = []

    if os.path.exists(os.path.join(data_dir, "blocks.shp")):
        logger.info("Loading block geoms from disk!")
        blocks_gdf = gpd.read_file(os.path.join(data_dir, "blocks.shp"))

    else:
        logger.info("Downloading block geoms from Census TIGERweb API!")

        # get block geoms from census tigerweb API
        for county in tqdm(
                county_codes, total=len(county_codes),
                desc='Getting block geoms for {0} counties'.format(
                    len(county_codes))):
            county_gdf = get_county_block_geoms(state_fips, county)
            all_block_geoms.append(county_gdf)

        blocks_gdf = gpd.GeoDataFrame(
            pd.concat(all_block_geoms, ignore_index=True), crs="EPSG:4326")

        # save to disk
        logger.info("Saving block geoms to disk!")
        blocks_gdf.to_file(os.path.join(data_dir, "blocks.shp"))

    return blocks_gdf


# Zones
@orca.table('zones', cache=True)
def zones(store, local_crs, data_dir):
    """
    if loading zones from shapefile, coordinates must be
    referenced to WGS84 (EPSG:4326) projection.
    """
    usim_zone_geoms = config.setting('usim_zone_geoms')
    zone_index = config.setting('usim_zone_index')

    # load from the h5 datastore if its there
    if '/zone_geoms' in store.keys():

        logger.info("Loading zone geometries from .h5 datastore!")
        zones = store['zone_geoms']
        if 'geometry' in zones.columns:
            zones['geometry'] = zones['geometry'].apply(wkt.loads)
            zones = gpd.GeoDataFrame(
                zones, geometry='geometry', crs='EPSG:4326') 
            zones.set_index(zone_index, inplace = True)
            zones.index.name = 'TAZ'
        else:
            raise KeyError(
                "Table 'zone_geoms' exists in the .h5 datastore but "
                "no geometry column was found!")

    # else try to load from list of h3 zone IDs
    elif usim_zone_geoms == 'h3':

        try:

            logger.info("Creating zone geometries from skim-based H3 IDs!")
            h3_zone_ids = inject.get_injectable('h3_zone_ids')
            zone_geoms = get_zone_geoms_from_h3(h3_zone_ids)
            zones = gpd.GeoDataFrame(
                h3_zone_ids, geometry=zone_geoms, crs="EPSG:4326")
            zones.columns = ['h3_id', 'geometry']
            zones['TAZ'] = list(range(1, len(h3_zone_ids) + 1))
            zones = zones.set_index('TAZ')

            # if using h3 zones, must clip geoms to block bounds
            # using local CRS
            logger.info("Clipping zone geoms to block boundaries!")
            block_bounds = orca.get_table('block_geoms').to_frame().to_crs(
                local_crs).unary_union
            zones = zones.to_crs(local_crs)
            zones['geometry'] = zones['geometry'].intersection(
                block_bounds)

            # convert back to epsg:4326 for storage in memory
            zones = zones.to_crs('EPSG:4326')

            # save zone geoms in .h5 datastore so we don't
            # have to do this again
            out_zones = pd.DataFrame(zones.copy())
            out_zones['geometry'] = out_zones['geometry'].apply(
                lambda x: x.wkt)

            logger.info("Storing zone geometries to .h5 datastore!")
            store['zone_geoms'] = out_zones

        except KeyError:
            raise RuntimeError(
                "Trying to create intermediate zones table from h3 IDs "
                "but the 'h3_zone_ids' injectable is not defined")

    elif ".shp" in usim_zone_geoms:
        fname = usim_zone_geoms
        filepath = config.data_file_path(fname)
        
        
        zones = gpd.read_file(filepath, crs="EPSG:4326")
#         zones.reset_index(inplace=True, drop=True) ## Problematic code (Assumes .shp file is in order) 
        
        zones.set_index(zone_index, inplace = True)
        zones.index.name = 'TAZ'
        # save zone geoms in .h5 datastore so we don't
        # have to do this again
        out_zones = pd.DataFrame(zones.copy())
        out_zones['geometry'] = out_zones['geometry'].apply(
            lambda x: x.wkt)

        logger.info("Storing zone geometries to .h5 datastore!")
        store['zone_geoms'] = out_zones

    else:
        raise RuntimeError(
            "Zone geometries incorrectly specified in settings.yaml")

    return zones


# Schools
@orca.table(cache=True)
def schools(store, state_fips, county_codes):

    if '/schools' in store.keys():
        logger.info("Loading school enrollment data from .h5 datastore!")
        enrollment = store['schools']
        
        #TO DO: ADD ASSERT TO CHECK SCHOOLS ARE IN THE STATE

    else:
        logger.info(
            "Downloading school enrollment data from educationdata.urban.org!")
        base_url = 'https://educationdata.urban.org/api/v1/' + \
            '{topic}/{source}/{endpoint}/{year}/?{filters}'

#         school_tables = []
#         for county in county_codes:
#             county_fips = str(state_fips) + str(county)
# #             enroll_filters = 'county_code={0}'.format(county_fips) #This not working anymore. 
#             enroll_filters = 'fips={}'.format(int(state_fips))
#             enroll_url = base_url.format(
#                 topic='schools', source='ccd', endpoint='directory',
#                 year='2015', filters=enroll_filters)

#             enroll_result = requests.get(enroll_url)
#             enroll = pd.DataFrame(enroll_result.json()['results'])
#             school_tables.append(enroll)
#             time.sleep(2)
        fips = int(state_fips)
        enroll_filters = 'fips={}'.format(fips) #All schools in state (can't do filter by county)
        enroll_url = base_url.format(topic='schools', source='ccd',
                                     endpoint='directory',year='2015', 
                                     filters=enroll_filters)
        enroll_result = requests.get(enroll_url)
        enrollment = pd.DataFrame(enroll_result.json()['results'])

#         enrollment = pd.concat(school_tables, axis=0)
        enrollment = enrollment[[
            'ncessch', 'county_code', 'latitude',
            'longitude', 'enrollment']].set_index('ncessch')
        enrollment.rename(
            columns={'longitude': 'x', 'latitude': 'y'}, inplace=True)
        
        #Filter schools in coutnies 
        enrollment = enrollment[enrollment['county_code'].str[1:].isin(county_codes)]
        
        logger.info("Saving school enrollment data to .h5 datastore!")
        store['schools'] = enrollment
    enrollment['enrollment'] = enrollment['enrollment'].clip(lower = 0) #Missing values (-1, and -2 as zeroes)
    return enrollment.dropna()


# Colleges
@orca.table(cache=True)
def colleges(store, state_fips, county_codes):

    if '/colleges' in store.keys():
        logger.info('Loading college data from .h5 datastore!')
        colleges = store['colleges']

    else:
        logger.info("Downloading college data from educationdata.urban.org!")
        base_url = 'https://educationdata.urban.org/api/v1/' + \
            '{topic}/{source}/{endpoint}/{year}/?{filters}'

        colleges_list = []
        for county in county_codes:
            county_fips = str(state_fips) + str(county)
            college_filters = 'county_fips={0}'.format(county_fips)
            college_url = base_url.format(
                topic='college-university', source='ipeds',
                endpoint='directory', year='2015', filters=college_filters)

            college_result = requests.get(college_url)
            college = pd.DataFrame(college_result.json()['results'])
            colleges_list.append(college)
            time.sleep(2)

        colleges = pd.concat(colleges_list)
        colleges = colleges[[
            'unitid', 'inst_name', 'longitude',
            'latitude']].set_index('unitid')
        colleges.rename(
            columns={'longitude': 'x', 'latitude': 'y'}, inplace=True)

        logger.info(
            "Downloading college full-time enrollment data from "
            "educationdata.urban.org!")
        colleges['full_time_enrollment'] = get_full_time_enrollment(state_fips)

        logger.info(
            "Downloading college part-time enrollment data from "
            "educationdata.urban.org!")
        colleges['part_time_enrollment'] = get_part_time_enrollment(state_fips)

        logger.info("Saving college data to .h5 datastore!")
        store['colleges'] = colleges
    
    colleges = colleges[~colleges.x.isnull()]
    return colleges


# ** 2. ASSIGN TAZ's to GEOMS **

@orca.column('schools', cache=True)
def TAZ(schools, zones, local_crs):
    if TAZ in schools.columns:
        return schools.TAZ
    zones_gdf = zones.to_frame(columns=['geometry'])
    schools_df = schools.to_frame(columns=['x', 'y'])
    schools_df.index.name = 'school_id'
    return get_taz_from_points(schools_df, zones_gdf, local_crs)


@orca.column('colleges', cache=True)
def TAZ(colleges, zones, local_crs):
    if TAZ in colleges.columns:
        return schools.TAZ
    colleges_df = colleges.to_frame(columns=['x', 'y'])
    colleges_df.index.name = 'college_id'
    zones_gdf = zones.to_frame(columns=['geometry'])
    return get_taz_from_points(colleges_df, zones_gdf, local_crs)

@orca.column('blocks', cache=True)
def TAZ(blocks, zones, local_crs):
    zones_gdf = zones.to_frame(columns=['geometry'])
    blocks_df = blocks.to_frame(columns=['x', 'y'])
    blocks_df.index.name = 'block_id'
    return get_taz_from_points(blocks_df, zones_gdf, local_crs)


@orca.column('usim_households')
def TAZ(blocks, usim_households):
    return misc.reindex(blocks.TAZ, usim_households.block_id)


@orca.column('usim_persons')
def TAZ(usim_households, usim_persons):
    return misc.reindex(usim_households.TAZ, usim_persons.household_id)


@orca.column('jobs')
def TAZ(blocks, jobs):
    return misc.reindex(blocks.TAZ, jobs.block_id)


# ** 3. CREATE NEW VARIABLES/COLUMNS **

# Block Variables

@orca.column('blocks')
def TOTEMP(blocks, jobs):
    jobs_df = jobs.to_frame(columns=['block_id', 'sector_id'])
    return jobs_df.groupby('block_id')['sector_id'].count().reindex(
        blocks.index).fillna(0)


@orca.column('blocks')
def TOTPOP(blocks, usim_households):
    hh = usim_households.to_frame(columns=['block_id', 'persons'])
    return hh.groupby('block_id')['persons'].sum().reindex(
        blocks.index).fillna(0)


@orca.column('blocks')
def TOTACRE(blocks):
    return blocks['square_meters_land'] / 4046.86


# Households Variables

@orca.column('usim_households')
def HHT(usim_households):
    s = usim_households.persons
    return s.where(s == 1, 4)

# @orca.column('usim_households')
# def TOTPOP(usim_households):
#     hh = usim_households.to_frame(columns=['TAZ', 'persons'])
#     return hh.groupby('TAZ')['persons'].sum().reindex(
#         blocks.index).fillna(0)


# Persons Variables

@orca.column('usim_persons')
def ptype(usim_persons):

    # Filters for person type segmentation
    # https://activitysim.github.io/activitysim/abmexample.html#setup
    age_mask_1 = usim_persons.age >= 18
    age_mask_2 = usim_persons.age.between(18, 64, inclusive=True)
    age_mask_3 = usim_persons.age >= 65
    work_mask = usim_persons.worker == 1
    student_mask = usim_persons.student == 1

    # Series for each person segmentation
    type_1 = ((age_mask_1) & (work_mask) & (~student_mask)) * 1  # Full time
    type_4 = ((age_mask_2) & (~work_mask) & (~student_mask)) * 4
    type_5 = ((age_mask_3) & (~work_mask) & (~student_mask)) * 5
    type_3 = ((age_mask_1) & (student_mask)) * 3
    type_6 = (usim_persons.age.between(16, 17, inclusive=True)) * 6
    type_7 = (usim_persons.age.between(6, 16, inclusive=True)) * 7
    type_8 = (usim_persons.age.between(0, 5, inclusive=True)) * 8
    type_list = [
        type_1, type_3, type_4, type_5, type_6, type_7, type_8]

    # Colapsing all series into one series
    for x in type_list:
        type_1.where(type_1 != 0, x, inplace=True)

    return type_1


@orca.column('usim_persons')
def pemploy(usim_persons):
    pemploy_1 = ((usim_persons.worker == 1) & (usim_persons.age >= 16)) * 1
    pemploy_3 = ((usim_persons.worker == 0) & (usim_persons.age >= 16)) * 3
    pemploy_4 = (usim_persons.age < 16) * 4

    # Colapsing all series into one series
    type_list = [pemploy_1, pemploy_3, pemploy_4]
    for x in type_list:
        pemploy_1.where(pemploy_1 != 0, x, inplace=True)

    return pemploy_1


@orca.column('usim_persons')
def pstudent(usim_persons):
    pstudent_1 = (usim_persons.age <= 18) * 1
    pstudent_2 = ((usim_persons.student == 1) & (usim_persons.age > 18)) * 2
    pstudent_3 = (usim_persons.student == 0) * 3

    # Colapsing all series into one series
    type_list = [pstudent_1, pstudent_2, pstudent_3]
    for x in type_list:
        pstudent_1.where(pstudent_1 != 0, x, inplace=True)

    return pstudent_1


# Zones variables

@orca.column('zones', cache=True)
def TOTHH(usim_households, zones):
    s = usim_households.TAZ.groupby(usim_households.TAZ).count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def TOTPOP(usim_persons, zones):
    s = usim_persons.TAZ.groupby(usim_persons.TAZ).count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def EMPRES(usim_households, zones):
    s = usim_households.to_frame(
        columns=['TAZ', 'workers']).groupby('TAZ')['workers'].sum()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HHINCQ1(usim_households, zones):
    df = usim_households.to_frame(columns=['income', 'TAZ'])
    df = df[df.income < 30000]
    s = df.groupby('TAZ')['income'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HHINCQ2(usim_households, zones):
    df = usim_households.to_frame(columns=['income', 'TAZ'])
    df = df[df.income.between(30000, 59999)]
    s = df.groupby('TAZ')['income'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HHINCQ3(usim_households, zones):
    df = usim_households.to_frame(columns=['income', 'TAZ'])
    df = df[df.income .between(60000, 99999)]
    s = df.groupby('TAZ')['income'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HHINCQ4(usim_households, zones):
    df = usim_households.to_frame(columns=['income', 'TAZ'])
    df = df[df.income >= 100000]
    s = df.groupby('TAZ')['income'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGE0004(usim_persons, zones):
    df = usim_persons.to_frame(columns=['TAZ', 'age'])
    df = df[df.age.between(0, 4)]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGE0519(usim_persons, zones):
    df = usim_persons.to_frame(columns=['TAZ', 'age'])
    df = df[df.age.between(5, 19)]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGE2044(usim_persons, zones):
    df = usim_persons.to_frame(columns=['TAZ', 'age'])
    df = df[df.age.between(20, 44)]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGE4564(usim_persons, zones):
    df = usim_persons.to_frame(columns=['TAZ', 'age'])
    df = df[df.age.between(45, 64)]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGE65P(usim_persons, zones):
    df = usim_persons.to_frame(columns=['TAZ', 'age'])
    df = df[df.age >= 65]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGE62P(usim_persons, zones):
    df = usim_persons.to_frame(columns=['TAZ', 'age'])
    df = df[df.age >= 62]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def SHPOP62P(zones):
    return (zones.AGE62P / zones.TOTPOP).reindex(zones.index).fillna(0)



@orca.column('zones', cache=True)
def TOTEMP(jobs, zones):
    s = jobs.TAZ.groupby(jobs.TAZ).count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def RETEMPN(jobs, zones):
    df = jobs.to_frame(columns=['sector_id', 'TAZ'])

    # difference is here (44, 45 vs 4445)
    # sector ids don't match
    df = df[df.sector_id.isin([4445])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def FPSEMPN(jobs, zones):
    df = jobs.to_frame(columns=['sector_id', 'TAZ'])
    df = df[df.sector_id.isin([52, 54])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HEREMPN(jobs, zones):
    df = jobs.to_frame(columns=['sector_id', 'TAZ'])
    df = df[df.sector_id.isin([61, 62, 71])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGREMPN(jobs, zones):
    df = jobs.to_frame(columns=['sector_id', 'TAZ'])
    df = df[df.sector_id.isin([11])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def MWTEMPN(jobs, zones):
    df = jobs.to_frame(columns=['sector_id', 'TAZ'])

    # sector ids don't match
    df = df[df.sector_id.isin([42, 3133, 32, 4849])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def OTHEMPN(jobs, zones):
    df = jobs.to_frame(columns=['sector_id', 'TAZ'])

    # sector ids don't match
    df = df[~df.sector_id.isin([
        4445, 52, 54, 61, 62, 71, 11, 42, 3133, 32, 4849])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def TOTACRE(blocks, zones, local_crs):

    # aggregate acreage from blocks
    blocks_df = blocks.to_frame(columns=['TOTACRE', 'TAZ'])
    s = blocks_df.groupby('TAZ')['TOTACRE'].sum()

    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HSENROLL(schools, zones):
    s = schools.to_frame().groupby(
        'TAZ')['enrollment'].sum()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones')
def TOPOLOGY():
    # assumes everything is flat
    return 1


@orca.column('zones')
def employment_density(zones):
    return zones.TOTEMP / zones.TOTACRE


@orca.column('zones')
def pop_density(zones):
    return zones.TOTPOP / zones.TOTACRE


@orca.column('zones')
def hh_density(zones):
    return zones.TOTHH / zones.TOTACRE


@orca.column('zones')
def hq1_density(zones):
    return zones.HHINCQ1 / zones.TOTACRE


@orca.column('zones')
def PRKCST(zones):
    params = pd.Series(
        [-1.92168743, 4.89511403, 4.2772001, 0.65784643], index=[
            'pop_density', 'hh_density', 'hq1_density',
            'employment_density'])

    cols = zones.to_frame(columns=[
        'employment_density', 'pop_density', 'hh_density', 'hq1_density'])

    s = cols @ params
    return s.where(s > 0, 0)


@orca.column('zones')
def OPRKCST(zones):
    params = pd.Series(
        [-6.17833544, 17.55155703, 2.0786466],
        index=['pop_density', 'hh_density', 'employment_density'])

    cols = zones.to_frame(
        columns=['employment_density', 'pop_density', 'hh_density'])

    s = cols @ params
    return s.where(s > 0, 0)


@orca.column('zones')  # College enrollment
def COLLFTE(colleges, zones):
    s = colleges.to_frame(columns=['TAZ', 'full_time_enrollment']).groupby(
        'TAZ')['full_time_enrollment'].sum()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones')  # College enrollment
def COLLPTE(colleges, zones):
    s = colleges.to_frame(columns=['TAZ', 'part_time_enrollment']).groupby(
        'TAZ')['part_time_enrollment'].sum()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones')
def area_type_metric(zones):
    """
    Because of the modifiable areal unit problem, it is probably a good
    idea to visually assess the accuracy of this metric when implementing
    in a new region. The metric was designed using SF Bay Area data and TAZ
    geometries. So what is considered "suburban" by SFMTC standard might be
    "urban" or "urban fringe" in less densesly developed regions, which
    can impact the results of the auto ownership and mode choice models.

    This issue should eventually resolve itself once we are able to re-
    estimate these two models for every new region/implementation. In the
    meantime, we expect that for regions less dense than the SF Bay Area,
    the area types classifications will be overly conservative. If anything,
    this bias results towards higher auto-ownership and larger auto-oriented
    mode shares. However, we haven't found this to be the case.
    """

    zones_df = zones.to_frame(columns=['TOTPOP', 'TOTEMP', 'TOTACRE'])

    metric_vals = ((
        1 * zones_df['TOTPOP']) + (
        2.5 * zones_df['TOTEMP'])) / zones_df['TOTACRE']

    return metric_vals.fillna(0)


@orca.column('zones')
def area_type(zones):
    # Integer, 0=regional core, 1=central business district,
    # 2=urban business, 3=urban, 4=suburban, 5=rural
    area_types = pd.cut(
        zones['area_type_metric'],
        [0, 6, 30, 55, 100, 300, float("inf")],
        labels=['5', '4', '3', '2', '1', '0'],
        include_lowest=True).astype(str)
    return area_types


@orca.column('zones')
def TERMINAL():
    # TO DO:
    # Improve the imputation of this variable
    # Average time to travel from automobile storage location to
    # origin/destination. We assume zero for now
    return 0  # Assuming O


@orca.column('zones')
def COUNTY():
    return 1  # Assuming 1 all San Francisco County


# ** 4. Define Orca Steps **
@inject.step()
def create_inputs_from_usim_data(data_dir, settings):

    persons_table = os.path.exists(os.path.join(data_dir, "persons.csv"))
    households_table = os.path.exists(os.path.join(data_dir, "households.csv"))
    land_use_table = os.path.exists(os.path.join(data_dir, "land_use.csv"))

    # if the input tables don't exist yet, create them from urbansim data
    if not (persons_table & households_table & land_use_table):

        logger.info("Creating inputs from UrbanSim data!")
        store = orca.get_injectable('store')

        # assign TAZ's to blocks if not already done. we only want to have to
        # do this once in an simulation workflow. The TAZ ID's will be
        # preserved in the land_use table
        assign_taz_to_blocks = False
        if 'TAZ' not in orca.get_table('blocks').local_columns:

            assign_taz_to_blocks = True

        elif orca.get_table('blocks')['TAZ'].isnull().all():

            assign_taz_to_blocks = True
            
#         elif orca.get_table('blocks')['TAZ'].min() == 0:
#             ## Assumes zones numbering starts in 1. 
#             assign_taz_to_blocks = True

        if assign_taz_to_blocks:
            
            zones_gdf = orca.get_table('zones').to_frame(columns=['geometry'])
            blocks_gdf = orca.get_table('block_geoms').to_frame()
            blocks_gdf.crs = 'EPSG:4326'

            blocks_to_taz = get_taz_from_block_geoms(
                blocks_gdf, zones_gdf, settings['local_crs'])

            orca.add_column('blocks', 'TAZ', blocks_to_taz)

            # save new column back to disk
            logger.info(
                "Storing blocks table with TAZ IDs to disk in .h5 datastore!")
            blocks = orca.get_table('blocks')
            blocks_output_cols = blocks.local_columns + ['TAZ']
            store['blocks'] = blocks.to_frame(columns=blocks_output_cols)

        else:

            logger.info(
                "Blocks already have TAZ assignments. Make sure the TAZ "
                "IDs have not changed!")

        # create households input table
        logger.info("Creating households table!")

        hh_names_dict = {
            'persons': 'PERSONS',
            'cars': 'VEHICL',
            'member_id': 'PNUM'}

        usim_households = orca.get_table('usim_households').to_frame()
        hh_df = usim_households.rename(columns=hh_names_dict)
        if 'household_id' in hh_df.columns:
            hh_df.set_index('household_id', inplace=True)
        else:
            hh_df.index.name = 'household_id'
        hh_null_taz = hh_df.TAZ.isnull()
        logger.info('Dropping {0} households without TAZs'.format(
            hh_null_taz.sum()))
        hh_df = hh_df[~hh_null_taz]
        hh_df.to_csv(os.path.join(data_dir, 'households.csv'))
        del hh_df

        # create persons input table
        logger.info("Creating persons table!")
        usim_persons = orca.get_table('usim_persons').to_frame()
        p_names_dict = {'member_id': 'PNUM'}
        p_df = usim_persons.rename(columns=p_names_dict)
        p_null_taz = p_df.TAZ.isnull()
        logger.info("Dropping {0} persons bc they have no TAZs".format(
            p_null_taz.sum()))
        p_df = p_df[~p_null_taz]

        # TO DO: come up with a different fix so we dont reset
        # person ids in between runs
        p_df.sort_values('household_id', inplace=True)
        p_df.reset_index(drop=True, inplace=True)
        p_df.index.name = 'person_id'
        p_df.to_csv(os.path.join(data_dir, 'persons.csv'))
        del p_df

        # create land use input table
        logger.info("Creating land use table!")
        lu_df = orca.get_table('zones').to_frame()
        lu_df.to_csv(os.path.join(data_dir, 'land_use.csv'))
        del lu_df

        # close the datastore
        store.close()

    else:
        logger.info("Found existing input tables, no need to re-create.")
