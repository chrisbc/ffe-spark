import datetime
import glob
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import networkx as nx
from shapely.geometry import Point
import imageio

pd.options.mode.chained_assignment = None  # default='warn'

path = "G:/Sync/FFE/Mesa"
path_output = "G:\Sync\FFE\FireNetwork"


# path = '/Users/alex/Google Drive/05_Sync/FFE/Mesa'
# path_output = '/Users/alex/Google Drive/05_Sync/FFE/Mesa/output'


# path = '/Users/alex/Google Drive/05_Sync/FFE/Mesa'

def load_data(file_name, minx, miny, maxx, maxy):
    # crop data
    bbox = box(minx, miny, maxx, maxy)
    # building point dataset
    gdf_buildings = gpd.read_file(os.path.join(path, file_name), bbox=bbox)
    # gdf_buildings.IgnProb_bl = 0.02
    # xmin,ymin,xmax,ymax = gdf_buildings.total_bounds
    return gdf_buildings


def wind_scenario():
    wind_data = pd.read_csv(os.path.join(path, 'GD_wind.csv'))
    i = np.random.randint(0, wind_data.shape[0])
    w = wind_data.iloc[i, 2]
    d = wind_data.iloc[i, 1]
    b = wind_data.iloc[i, 3]
    return w, d, b


def eudistance(v1, v2):
    return np.linalg.norm(v1 - v2)


def calculate_azimuth(x1, y1, x2, y2):
    azimuth = math.degrees(math.atan2((x2 - x1), (y2 - y1)))
    return 360 + azimuth


def plot(df, column_df):
    fig, ax = plt.subplots(1, 1)
    df.plot(column=column_df, ax=ax, legend=True)
    plt.show()


def build_edge_list(geodataframe, maximum_distance, polygon_file):
    # create arrays for different id combination
    n = np.arange(0, len(geodataframe))
    target = [n] * len(geodataframe)
    target = np.hstack(target)
    source = np.repeat(n, len(geodataframe))
    # put arrays in dataframe
    df = pd.DataFrame()
    df['source_id'] = source
    df['target_id'] = target
    # merge source attributes with source index
    geo_df = geodataframe.copy()
    geo_df['id'] = geo_df.index
    # create source / target gdf from gdf.columns of interest
    geo_df = geo_df[['id', 'TARGET_FID', 'X', 'Y', 'geometry', 'IgnProb_bl']]
    geo_df_TRG = geo_df.copy()
    geo_df_TRG.columns = ['target_' + str(col) for col in geo_df_TRG.columns]
    geo_df_SRC = geo_df.copy()
    geo_df_SRC.columns = ['source_' + str(col) for col in geo_df_SRC.columns]
    # merge data
    merged_data = pd.merge(df, geo_df_SRC, left_on='source_id', right_on='source_id', how='outer')
    merged_data = pd.merge(merged_data, geo_df_TRG, left_on='target_id', right_on='target_id', how='outer')
    merged_data.rename(columns={'source_id': 'source', 'target_id': 'target'}, inplace=True)
    # calculate distance for each source / target pair
    # create a df from polygon shape to get accurate distance
    # print(list(polygon_file))
    polygon = polygon_file[['TARGET_FID', 'geometry']]
    # print(list(polygon))
    source_poly = merged_data[['source_TARGET_FID']]
    target_poly = merged_data[['target_TARGET_FID']]
    # print(list(source_poly))
    src_poly = pd.merge(source_poly, polygon, left_on='source_TARGET_FID', right_on='TARGET_FID', how='left')
    trg_poly = pd.merge(target_poly, polygon, left_on='target_TARGET_FID', right_on='TARGET_FID', how='left')
    src_poly_gdf = gpd.GeoDataFrame(src_poly, geometry='geometry')
    trg_poly_gdf = gpd.GeoDataFrame(trg_poly, geometry='geometry')
    distance_series = src_poly_gdf.distance(trg_poly_gdf)
    # print(distance_series)

    # insert distance in merged data column
    merged_data['v1'] = merged_data.source_X - merged_data.target_X
    merged_data['v2'] = merged_data.source_Y - merged_data.target_Y
    # merged_data['euc_distance'] = np.hypot(merged_data.v1, merged_data.v2)
    merged_data['euc_distance'] = distance_series
    # remove when distance "illegal"
    valid_distance = merged_data['euc_distance'] < maximum_distance
    not_same_node = merged_data['euc_distance'] != 0
    data = merged_data[valid_distance & not_same_node]
    # calculate azimuth
    data['azimuth'] = np.degrees(np.arctan2(merged_data['v2'], merged_data['v1']))
    data['bearing'] = (data.azimuth + 360) % 360
    return data


def create_network(edge_list_dataframe):
    graph = nx.from_pandas_edgelist(edge_list_dataframe, edge_attr=True)
    # options = {'node_color': 'red', 'node_size': 50, 'width': 1, 'alpha': 0.4,
    #            'with_labels': False, 'font_weight': 'bold'}
    # nx.draw_kamada_kawai(graph, **options)
    # plt.show()
    return graph


# run model
def set_initial_fire_to(df):
    """Fine = 0, Fire = 1, Burned = 2"""
    df['RNG'] = np.random.uniform(0, 1, size=len(df))  # add for random suppression per building, df.shape[0])
    onFire = df['source_IgnProb_bl'] > df['RNG']
    ignitions = df[onFire]
    # source nodes ignited
    sources_on_fire = list(ignitions.source)
    sources_on_fire = list(dict.fromkeys(sources_on_fire))
    return sources_on_fire


def set_fire_to(df, existing_fires):
    are_set_on_fire = (df['source'].isin(existing_fires))
    spark = df[are_set_on_fire]
    # source nodes ignited
    sources_on_fire = list(spark.source)
    sources_on_fire = list(dict.fromkeys(sources_on_fire))
    return sources_on_fire


def fire_spreading(list_fires, list_burn, wind_speed, wind_bearing, suppression_threshold, step_value, data):
    # check the fire potential targets
    # print("fire list before spreading : {}, length : {}".format(fire_list, len(fire_list)))
    are_potential_targets = (data['source'].isin(list_fires))
    are_not_already_burned = (~data['target'].isin(list_burn))
    df = data[are_potential_targets & are_not_already_burned]
    if df.empty:
        # print("no fires")
        list_burn.extend(list(list_fires))
        list_burn = list(dict.fromkeys(list_burn))
        return [], list_burn  # to break the step loop
    # set up additional CONDITIONS for fire spreading

    # neighbors selection from buffer
    df['buffer_geometry'] = gdf.geometry.buffer(gdf['d_long'] + wind_speed)

    are_neighbors = df['euc_distance'] < wind_speed
    # print("neighbors affected ? {}".format(list(dict.fromkeys(list(are_neighbors)))))
    df = df[are_neighbors]
    # wind direction
    wind_bearing_max = wind_bearing + 45
    wind_bearing_min = wind_bearing - 45
    if wind_bearing == 360:
        wind_bearing_max = 45
    if wind_bearing <= 0:  # should not be necessary
        wind_bearing_min = 0
    if wind_bearing == 999:
        wind_bearing_max = 999
        wind_bearing_min = 0
    are_under_the_wind = (df['bearing'] < wind_bearing_max) & (df['bearing'] > wind_bearing_min)
    # print("targets under the wind ? {}".format(list(dict.fromkeys(list(are_under_the_wind)))))
    df = df[are_under_the_wind]
    # suppression
    df['random'] = np.random.uniform(0, 1, size=len(df))
    are_not_suppressed = df['random'] > suppression_threshold
    # print("fire suppressed ? {}".format(list(dict.fromkeys(list(are_not_suppressed)))))
    df = df[are_not_suppressed]

    # spread fire based on condition
    fire_df = df
    # fire_df = df[are_neighbors & are_under_the_wind & are_not_suppressed]  # issues with "are_under_the_wind
    # print(len(fire_df.head(5)))
    # print(len(fire_df))
    list_burn.extend(list(list_fires))
    fire_df['step'] = step_value
    fire_df.to_csv(os.path.join(path_output, "step{}_fire.csv".format(step_value)))
    list_fires = list(dict.fromkeys(list(fire_df.target)))
    list_burn.extend(list(fire_df.target))
    list_burn = list(dict.fromkeys(list_burn))
    return list_fires, list_burn


def log_files_concatenate(prefix, scenario_count):
    list_df = []
    files = glob.glob(os.path.join(path_output, prefix))
    if files:
        for file in files:
            # print(file)
            df = pd.read_csv(os.path.join(path_output, file))
            list_df.append(df)
            os.remove(file)
        data = pd.concat(list_df)
        data['scenario'] = scenario_count
        data.to_csv(os.path.join(path_output, "fire_scenario_{}.csv".format(scenario_count)))
    else:
        print("no files to concatenate")


def clean_up_file(prefix, path_path=path_output):
    files = glob.glob(os.path.join(path_path, prefix))
    for file in files:
        # print(file)
        os.remove(file)


def postprocessing(scenarios_recorded, burned_asset, edge_list, gdf_polygons):
    list_of_tuples = list(zip(scenarios_recorded, burned_asset))
    df = pd.DataFrame(list_of_tuples, columns=['scenarios', 'burned_asset_index'])
    # df['count'] = df['burned_asset_index'].value_counts().values
    df['count'] = df.groupby('burned_asset_index')['burned_asset_index'].transform('count')
    print(df.describe())
    df = df[['burned_asset_index', 'count']].drop_duplicates()
    edge = edge_list[
        ['source', 'source_TARGET_FID', 'source_X', 'source_Y', 'source_geometry']]
    df_id = pd.merge(df, edge, left_on='burned_asset_index', right_on='source', how='left')
    # print(list(df_id))
    df_count = pd.merge(gdf_polygons, df_id, left_on='TARGET_FID', right_on='source_TARGET_FID', how='outer')
    df_count = df_count.drop_duplicates()
    dataframe = pd.DataFrame(df_count.drop(columns=['geometry', 'source_geometry']))
    dataframe = dataframe.dropna()
    fig, ax = plt.subplots(1, 1)
    df_count.plot(column='count', cmap='RdYlBu_r', ax=ax, legend=True)
    ax.title.set_text("Burned buildings after {} scenarios".format(max(scenarios_recorded)))
    plt.show()
    df_count = df_count.drop(columns=['source', 'source_TARGET_FID', 'source_X', 'source_Y', 'source_geometry'])
    df_count.to_csv(os.path.join(path_output, "results.csv"))
    # df_count.to_file(os.path.join(path_output, "results.shp"))
    return df_count, dataframe


# set up & load input data
# gdf = load_data("buildings_raw_pts.shp", 1748570, 5426959, 1748841, 5427115)
gdf_polygon = load_data("buildings_raw.shp", 1748000, 5424148, 1750000, 5427600)
gdf_polygon["area"] = gdf_polygon['geometry'].area  # m2
gdf = gdf_polygon.copy()
gdf['geometry'] = gdf['geometry'].centroid
gdf['X'] = gdf.centroid.x
gdf['Y'] = gdf.centroid.y
gdf['d_short'] = gdf_polygon.exterior.distance(gdf)
gdf['d_long'] = gdf['area'] / gdf['d_short']

# create edge list and network
edges = build_edge_list(gdf, 45, gdf_polygon)

# create edges
G = create_network(edges)


#################################
# set number of scenarios
number_of_scenarios = 10
# display of the input data
print("{} assets loaded".format(len(gdf)))
fig, ax = plt.subplots(2, 2)
# gdf.plot(column='area', cmap='hsv', ax=ax[0, 0], legend=True)
gdf_polygon.plot(column='area', cmap='hsv', ax=ax[0, 0], legend=True)
# gdf.plot(column='TARGET_FID', cmap='hsv', ax=ax[1, 0], legend=True)
options = {'node_color': 'red', 'node_size': 50, 'width': 1, 'alpha': 0.4,
               'with_labels': False, 'font_weight': 'bold'}
nx.draw_kamada_kawai(G, **options, ax=ax[1, 1])
ax[0,0].title.set_text("area")
ax[0,1].title.set_text("area")
ax[1,0].title.set_text('FID')
ax[1,1].title.set_text('Network display')
plt.tight_layout()
plt.savefig(os.path.join(path_output, "inputs_{}.png".format(number_of_scenarios)))
plt.show()
plt.close(fig)
################################


# run model
clean_up_file("*csv")
scenarios_list = []
log_burned = []  # no removing duplicate
# --- SCENARIOS
t = datetime.datetime.now()
for scenario in range(number_of_scenarios):
    t0 = datetime.datetime.now()
    burn_list = []
    print("--- SCENARIO : {}".format(scenario))
    # print("initiate fire")
    fire_list = set_initial_fire_to(edges)
    x = fire_list
    # print("fire list : {}, length : {}".format(fire_list, len(fire_list)))
    # print("fires list in scenario loop: {}, length : {}".format(fire_list, len(fire_list)))
    if len(fire_list) == 0:
        print("no fire")
        continue
    w_direction, w_speed, w_bearing = wind_scenario()
    # print(("critical distance : {}, wind bearing : {}".format(w_speed, w_bearing)))
    # --------- STEPS
    for step in range(len(edges)):
        print("--------- STEP : {}".format(step))
        fire_list = set_fire_to(edges, fire_list)
        y = fire_list
        # print("fire datasets are identical with initial fire : {}".format(set(x) == set(y)))
        # print("fire list : {}, length : {}".format(fire_list, len(fire_list)))
        # print("burn list : {}, length : {}".format(burn_list, len(burn_list)))
        # print("spread fire")
        fire_list, burn_list = fire_spreading(fire_list, burn_list, w_speed, w_bearing, 0, step, edges)
        if len(fire_list) == 0:
            # print("no fires")
            break
        # print("fires list : {}, length : {}".format(fire_list, len(fire_list)))
        # print("burn list : {}, length : {}".format(burn_list, len(burn_list)))
    log_burned.extend(burn_list)
    scenarios_list.extend([scenario] * len(burn_list))
    # print("log all burn list : {}, length : {}".format(log_burned, len(log_burned)))
    # print(scenarios_list)

    log_files_concatenate('step*', scenario)
    t1 = datetime.datetime.now()
    print("..... took : {}".format(t1 - t0))
t2 = datetime.datetime.now()
print("total time : {}".format(t2 - t))

count_gdf, count_df = postprocessing(scenarios_list, log_burned, edges, gdf_polygon)