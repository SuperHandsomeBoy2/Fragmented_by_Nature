import sys

print('Python %s on %s' % (sys.version, sys.platform))
import os
import time
import torch
from Detour import (GraphicalIndexWithRoadMap, LoadMultiGeoAndCreateCommutingPointsAndRoadMap)


def global_graphical_index_with_road_map(
        geo_barrier_path='Data/demo_data/geo_barrier_3_cities.shp',
        road_net_path='Data/demo_data/road_net_3_cities.shp',
        save_path='result/global graphical index/global_cities_with_road_map'
):
    t = time.time()

    # save path
    save_all = False
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load geo data
    geo_data = LoadMultiGeoAndCreateCommutingPointsAndRoadMap(
        geo_barrier_path,
        road_net_path,
        0.25)

    # graphical index for all regions
    graphical_indexer = GraphicalIndexWithRoadMap(
        d_max=2 ** 11,
        block_sz=4,
        neighbor_d_max=1,
        degree_threshold=0.49,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    )

    st_id = 0
    # st_id = 8030
    t_f = time.time()
    for i, data in enumerate(geo_data[st_id:]):
        t_load = time.time()
        ii = i + st_id
        id_hdc_g0 = int(geo_data.id_hdc_g0[ii])
        print(f'No. {ii} region')
        barrier_st, barrier_ed, commuting_points, traffic_nodes, road_net, region_bounds, x_factor, y_factor, region_center = data

        # graphical index
        commuting_points = commuting_points.to(dtype=torch.float16)
        traffic_nodes = traffic_nodes.to(dtype=torch.float16)
        road_net = road_net.to(dtype=torch.float16)
        barrier_st = barrier_st.to(dtype=torch.float16)
        barrier_ed = barrier_ed.to(dtype=torch.float16)

        if save_path is None or not save_all:
            save_name = None
        else:
            save_name = save_path + f'/{ii}_{id_hdc_g0}'
            if not os.path.exists(save_name):
                os.makedirs(save_name)
            save_name = save_name + '/'

        gid, d_max, d_mean, d_std, d_mean_direct, n_c, n_r = graphical_indexer.run(
            commuting_points,
            traffic_nodes,
            road_net,
            barrier_st,
            barrier_ed,
            save_name=save_name,
        )
        t_run = time.time()

        print(f'No. {ii} region, id_hdc_g0: {id_hdc_g0}, Graphical index = {gid}, \n'
              f'data load and transform {t_load - t_f} s, calculation {t_run - t_load} s, \n'
              f'total time = {t_run - t}.')
        t_f = t_run

        # save txt
        if save_path is not None:
            file = open(save_path + f'/{ii}_{id_hdc_g0}_index.txt', 'w')
            file.write(f'{ii},{id_hdc_g0},{gid},{d_max},{d_mean},{d_std},{d_mean_direct},{n_c},{n_r}')
            file.close()


def save_csv(summary, filename: str, sort_by: str = None):
    import pandas as pd
    # whether path exist
    print('Save summary to: ' + filename + ' ...')
    (filepath, temp_filename) = os.path.split(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # save the lists to certain text
    if '.csv' not in filename:
        filename += '.csv'
    data = pd.DataFrame(summary)
    if sort_by is not None:
        data = data.sort_values(by=sort_by)
    data.to_csv(filename)
    print('Save finish')


def convert_text_to_csv(save_path: str):
    # scan the txt files
    print(f'Scanning {save_path} ...')
    files = os.listdir(save_path)
    for i, file_name in enumerate(files):
        front, ext = os.path.splitext(file_name)

        if not ext == '.txt':
            del files[i]

    print(f'Got. {len(files)} .txt files')

    # get summary
    summaries = []
    for file_name in files:
        position = save_path + '/' + file_name
        print(f'Open {file_name}')
        with open(position, 'r') as f:
            data = f.read().split(',')
            summary = {'list id': data[0], 'ID_HDC_G0': data[1],
                       'graphical index': data[2],
                       'max distance': data[3], 'mean distance': data[4], 'distance std': data[5],
                       'mean direct distance': data[6],
                       'number of commuting nodes': data[7],
                       'number of traffic nodes': data[8],
                       }
            summaries.append(summary)
    save_csv(summaries, save_path + '/summary.csv', )


if __name__ == '__main__':
    # run demo
    geo_barrier_path = 'Data/demo_data/geo_barrier_3_cities.shp'
    road_net_path = 'Data/demo_data/road_net_3_cities.shp'
    save_path = 'result/3_cities_with_road_map'
    global_graphical_index_with_road_map(geo_barrier_path, road_net_path, save_path)
    convert_text_to_csv(save_path)
