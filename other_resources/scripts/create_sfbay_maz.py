import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

folder = "prototype_mtc_bg_2_zone"
zone = "bg"
counties = ['001', '013', '041', '055', '081', '085', '075', '095', '097']

if zone == "bl":
    bl = gpd.read_file("/Users/zaneedell/Downloads/tl_2010_06_tabblock10/tl_2010_06_tabblock10.shp")
    bl_sub = bl.loc[bl.COUNTYFP10.isin(counties)]

    bl_sub['GISJOIN'] = "G" + bl_sub['STATEFP10'] + "0" + bl_sub['COUNTYFP10'] + "0" + bl_sub['TRACTCE10'] + bl_sub[
        'BLOCKCE10']
    pops = pd.read_csv('/Users/zaneedell/Downloads/nhgis0001_csv/nhgis0001_ds172_2010_block.csv')
    bl_sub = pd.merge(bl_sub, pops[['GISJOIN', 'H7V001']], on="GISJOIN")
if zone == "bg":
    bl_sub = gpd.read_file("/Users/zaneedell/Downloads/region_blockgroup/region_blockgroup.shp")
    bl_sub['GISJOIN'] = "G" + bl_sub['fipst'] + "0" + bl_sub['fipco'] + "0" + bl_sub['tractn'] + bl_sub['blkgrpn']
    pops = pd.read_csv('/Users/zaneedell/Downloads/nhgis0002_csv/nhgis0002_ds172_2010_blck_grp.csv')
    bl_sub = pd.merge(bl_sub, pops[['GISJOIN', 'H7V001']], on="GISJOIN")

bl_sub.to_crs('epsg:26910', inplace=True)
bl_sub['area_acres'] = bl_sub.geometry.area * 0.000247105
# bl_sub.to_file("../../activitysim/examples/shp/block_groups.shp")
bl_sub['full_geometry'] = bl_sub['geometry'].copy()
bl_sub['geometry'] = bl_sub.geometry.centroid
taz = gpd.read_file("/Users/zaneedell/Desktop/git/beam/production/sfbay/shape/sfbay-tazs-epsg-26910.shp")

lu_orig = pd.read_csv("/Users/zaneedell/Desktop/git/activitysim/bay_area/data/land_use.csv")

joined = gpd.sjoin(bl_sub[['geometry', 'H7V001', 'area_acres','full_geometry']], taz[['taz1454', 'geometry']])
joined = joined.reset_index().drop(columns=['index_right', 'index']).rename(columns={'H7V001': 'Population'})
joined_sub = joined.loc[joined.Population > 0]

taz_missing = taz.loc[~taz.taz1454.isin(joined_sub.taz1454.unique()), ['taz1454', 'geometry']]
taz_missing['area_acres'] = taz_missing.geometry.area * 0.000247105
taz_missing['full_geometry'] = taz_missing['geometry'].copy()
taz_missing.geometry = taz_missing.geometry.centroid

joined_sub_extended = pd.concat([joined_sub, taz_missing], axis=0).fillna(0.0).reset_index(drop=True)
joined_sub_extended.rename_axis('MAZ', inplace=True)
joined_sub_extended.drop('full_geometry', axis=1).to_csv("../../activitysim/examples/{0}/data/maz.csv".format(folder), float_format='%.4f')
gdf = gpd.GeoDataFrame(joined_sub_extended, crs="epsg:26910")

gdf.set_geometry('full_geometry').to_file("../../activitysim/examples/{0}/data/maz.geojson")


required_columns = ["TAZ", "DISTRICT", "SD", "COUNTY", "TOTHH", "TOTPOP", "TOTACRE", "RESACRE", "CIACRE", "TOTEMP",
                    "AGE0519", "RETEMPN", "FPSEMPN", "HEREMPN", "OTHEMPN", "AGREMPN", "MWTEMPN", "PRKCST", "OPRKCST",
                    "area_type", "HSENROLL", "COLLFTE", "COLLPTE", "TOPOLOGY", "TERMINAL"]


def balltree_distance_matrix(a, b, max_distance, output_type="dok_matrix"):
    ac = BallTree(a)
    ind, dist = ac.query_radius(b, max_distance, return_distance=True, count_only=False)

    return ind, dist


out = joined_sub_extended.geometry.apply(lambda x: (x.x, x.y)).to_list()
inds, dists = balltree_distance_matrix(out, out, max_distance=1600)

maz_to_maz = []
for from_idx, (ind, dist) in enumerate(zip(inds, dists)):
    maz_to_x = joined_sub_extended.iloc[ind].index.rename('DMAZ').to_frame().reset_index(drop=True)
    maz_to_x['OMAZ'] = from_idx
    maz_to_x['DISTWALK'] = dist.clip(200, None) / 1609.34
    maz_to_maz.append(maz_to_x)

maz_to_maz_df = pd.concat(maz_to_maz, axis=0)
maz_to_maz_df.to_csv("../../activitysim/examples/{0}/data/maz_to_maz_walk.csv.gz".format(folder), float_format='%.3f', index=False)
maz_to_maz_df.rename(columns={"DISTWALK": "DISTBIKE"}).to_csv(
    "../../activitysim/examples/{0}/data/maz_to_maz_bike.csv.gz".format(folder), float_format='%.3f', index=False)
print('DONE')

pop_columns = ['TOTHH', 'TOTPOP', 'EMPRES', 'HHINCQ1',
               'HHINCQ2', 'HHINCQ3', 'HHINCQ4', 'AGE0004', 'AGE0519', 'AGE2044',
               'AGE4564', 'AGE64P', 'AGE62P', 'SHPOP62P', 'TOTEMP', 'RETEMPN',
               'FPSEMPN', 'HEREMPN', 'AGREMPN', 'MWTEMPN', 'OTHEMPN',
               'HSENROLL', 'COLLFTE', 'COLLPTE']

copy_cols = ['TOPOLOGY', 'PRKCST', 'OPRKCST', 'area_type', 'COUNTY', 'TERMINAL', 'area_type_metric']

density_cols = {'TOTEMP': 'employment_density', 'TOTPOP': 'pop_density',
                'TOTHH': 'hh_density', 'HHINCQ1': 'hq1_density'}

area_columns = ['TOTACRE']


def divideTaz(tazIdx, mazIdxs):
    taz_sub = lu_orig.loc[lu_orig.TAZ == tazIdx]
    maz_sub = joined_sub_extended.loc[mazIdxs].copy()
    if maz_sub.shape[0] > 0:
        maz_sub["pop_probs"] = (maz_sub['Population'] / maz_sub['Population'].sum()).fillna(1.0)
        maz_sub["area_probs"] = (maz_sub['area_acres'] / maz_sub['area_acres'].sum()).fillna(1.0)
        for col in pop_columns:
            maz_sub[col] = np.round(taz_sub[col].values[0] * maz_sub['pop_probs']).astype(int)
        for col in area_columns:
            maz_sub[col] = taz_sub[col].values[0] * maz_sub['area_probs']
        for col in copy_cols:
            maz_sub[col] = taz_sub[col].values[0]
        for num, col in density_cols.items():
            maz_sub[col] = maz_sub[num] / maz_sub["TOTACRE"].values[0]
        return maz_sub
    else:
        return taz_sub


out = []
for tazId, mazIds in joined_sub_extended.groupby('taz1454').groups.items():
    out.append(divideTaz(tazId, mazIds))



new_lu = pd.concat(out, axis=0)
new_lu['TAZ'] = new_lu['taz1454'].astype(int)
new_lu.drop(columns=['taz1454'], inplace=True)


lu_missing = lu_orig.loc[~lu_orig.TAZ.isin(joined_sub.taz1454.unique()), :]
lu_missing.reset_index(inplace=True)
lu_missing.index = lu_missing.index + new_lu.index.max()
lu_missing.rename_axis('MAZ', inplace=True)
lu_full = pd.concat([new_lu.sort_index(), lu_missing])



stops = gpd.read_file("../../activitysim/examples/shp/transitstops_2021_existing_planned.shp")
stops.to_crs('epsg:26910', inplace=True)

maz_centroids = joined_sub_extended.geometry.apply(lambda x: (x.x, x.y)).to_list()
stop_ball_tree = BallTree(stops.geometry.apply(lambda x: (x.x, x.y)).to_list())
dist_to_nearest_stop, nearest_stop_ind = stop_ball_tree.query(maz_centroids, k=1, return_distance=True)
too_far = dist_to_nearest_stop > (1609.34 * 1.5)
dist_to_nearest_stop /= 1609.34
dist_to_nearest_stop[too_far] = -1
# TODO: Make columns access_dist_transit with distance in miles to nearest transit station
lu_full['access_dist_transit'] = -1.0
lu_full['access_dist_transit'].values[:len(dist_to_nearest_stop)] = dist_to_nearest_stop.flatten()
for col in pop_columns:
    lu_full[col] = lu_full[col].astype(int)
lu_full = lu_full.loc[~lu_full.index.duplicated()].loc[joined_sub_extended.index]
lu_full.to_csv("../../activitysim/examples/{0}/data/land_use.csv".format(folder))


lu_full.TAZ.drop_duplicates().sort_values().to_csv("../../activitysim/examples/{0}/data/taz.csv".format(folder), index=False)
print('DONE')

def sample_maz(df):
    taz = df['TAZ'].values[0]
    nPersons = df.shape[0]
    weights = lu.loc[lu.TAZ == taz, 'Population'] + 1
    if weights.sum() == 0:
        weights = None
        print("Skipping taz {0}".format(taz))
    maz = lu.loc[lu.TAZ == taz, 'MAZ'].sample(n=nPersons, weights=weights, replace=True).astype(int)
    df['MAZ'] = 0
    df['MAZ'] = maz.values
    return df



hh = pd.read_csv("../../activitysim/examples/prototype_mtc/data/households.csv")
lu = pd.read_csv("../../activitysim/examples/prototype_mtc_bg_2_zone/data/land_use.csv")
