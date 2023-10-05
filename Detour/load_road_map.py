import time
import math
import numpy as np
import torch
import geopandas
import networkx as nx
import osmnx


def geo_data_to_geo_lines(
        geo_data: geopandas.GeoSeries,
        region_center: torch.Tensor,
):
    print(f'Convert geo LineString to torch Tensor .. ')
    region_center = region_center.view(-1, 1)
    points_st = []
    points_ed = []
    if geo_data.boundary.__class__.__name__ == 'LineString':
        boundary = geo_data.boundary
        xy = torch.from_numpy(np.array(boundary.xy))
        if region_center is not None:
            xy = xy - region_center
        xy = xy.to(torch.float32)
        points_st.append(xy[:, :-1])
        points_ed.append(xy[:, 1:])
    else:
        for boundary in geo_data.boundary.geoms:
            if boundary.__class__.__name__ == 'LineString':
                xy = torch.from_numpy(np.array(boundary.xy))
                if region_center is not None:
                    xy = xy - region_center
                xy = xy.to(torch.float32)
                points_st.append(xy[:, :-1])
                points_ed.append(xy[:, 1:])
            elif boundary.__class__.__name__ == 'MultiLineString':
                for ss_boundary in boundary.geoms:
                    if ss_boundary.__class__.__name__ == 'LineString':
                        xy = torch.from_numpy(np.array(ss_boundary.xy))
                        if region_center is not None:
                            xy = xy - region_center
                        xy = xy.to(torch.float32)
                        points_st.append(xy[:, :-1])
                        points_ed.append(xy[:, 1:])
                    else:
                        raise TypeError('ss_boundary should be LineString')
            else:
                raise TypeError('boundary should be LineString or MultiLineString')
    points_st = torch.cat(points_st, dim=1)
    points_ed = torch.cat(points_ed, dim=1)
    print(f'Finish: converted {points_st.size()[-1]} LineStrings')
    return points_st, points_ed


def geo_data_to_points(
        geo_data: geopandas.GeoDataFrame,
        region_center: torch.Tensor,
):
    print(f'Convert geo Points to torch Tensor .. ')
    region_center = region_center.view(-1, 1)
    points = []
    for region_shape in geo_data.geometry:
        if region_shape.__class__.__name__ == 'Point':
            xy = region_shape.xy
            xy = torch.from_numpy(np.array(xy))
            if region_center is not None:
                xy = xy - region_center
            xy = xy.to(torch.float32)
            points.append(xy.view(-1, 1))
    points = torch.cat(points, dim=1)
    print(f'Finish: converted {points.size()[-1]} Points ')
    return points


def create_commuting_points(
        x_factor: float, y_factor: float,
        region_bounds: np.array,
        sampling_interval: float,   # km
        crs='EPSG:4326',  # 指定坐标系为WGS 1984
):
    dx = sampling_interval / x_factor
    dy = sampling_interval / y_factor

    cx = 0.5 * (region_bounds[0] + region_bounds[2])
    cy = 0.5 * (region_bounds[1] + region_bounds[3])

    nx = int((region_bounds[2] - cx) // dx)
    ny = int((region_bounds[3] - cy) // dy)

    px_min = cx - nx * dx
    px_max = cx + nx * dx
    py_min = cy - ny * dy
    py_max = cy + ny * dy

    x_list = torch.linspace(px_min, px_max, nx * 2 + 1)
    y_list = torch.linspace(py_min, py_max, ny * 2 + 1)

    nx = x_list.size(0)
    ny = y_list.size(0)

    px = x_list.view(-1, 1).expand(nx, ny).reshape(-1)
    py = y_list.view(1, -1).expand(nx, ny).reshape(-1)

    points = geopandas.GeoSeries(geopandas.points_from_xy(px.numpy(), py.numpy()),
                                 crs=crs,
                                 index=[str(i) for i in range(len(px))]  # 相关的索引
                                 )
    return geopandas.GeoDataFrame(geometry=points)


def add_road_to_graph(
        road,
        graph: nx.MultiDiGraph,
        scale_factor,
        region_center,
):
    assert road.__class__.__name__ == 'LineString'
    road = (np.array(road.xy) - region_center) * scale_factor
    n = road.shape[-1]
    for i in range(n - 1):
        from_node = road[:, i]
        to_node = road[:, i + 1]
        length = ((from_node - to_node) ** 2).sum() ** 0.5

        from_node = (from_node[0], from_node[1])
        to_node = (to_node[0], to_node[1])

        if not graph.has_node(from_node):
            graph.add_node(from_node, x=from_node[0], y=from_node[1], pos=from_node)
        if not graph.has_node(to_node):
            graph.add_node(to_node, x=to_node[0], y=to_node[1], pos=to_node)

        graph.add_edge(from_node, to_node, length=length)
    return graph


def create_road_map(
        x_factor: float, y_factor: float,
        region_center,
        region_bounds: np.array,
        barrier_data,
        road_map_data,
        id_hdc_g0=None,
        radius_ratio: float = 1.1,
        crs='EPSG:4326',  # 指定坐标系为WGS 1984
        how='intersection',
        simplify_graph=False,
):
    """
    get the road map graph from geopandas GeoDataFrame,
    where each row has a multiline corresponding to a road
    :param x_factor:
    :param y_factor:
    :param region_center:
    :param region_bounds:
    :param barrier_data:
    :param id_hdc_g0:
    :param road_map_data:
    :param radius_ratio:    # radio of road map range compared with commuting nodes
    :param crs:
    :param how:
    :return:
    """
    # clip road map data in current region
    cx = 0.5 * (region_bounds[0] + region_bounds[2])
    cy = 0.5 * (region_bounds[1] + region_bounds[3])

    rx = (region_bounds[2] - cx) * radius_ratio
    ry = (region_bounds[3] - cy) * radius_ratio

    clip_range = (cx - rx, cy - ry, cx + rx, cy + ry)
    if id_hdc_g0 is not None and 'ID_HDC_G0' in road_map_data.columns:
        map_data = road_map_data.loc[road_map_data['ID_HDC_G0'] == id_hdc_g0]
        map_data = geopandas.clip(map_data, clip_range)
    else:
        map_data = geopandas.clip(road_map_data, clip_range)
        map_data = geopandas.overlay(
            map_data,
            geopandas.GeoDataFrame(
                geometry=geopandas.GeoSeries(barrier_data.convex_hull), crs=crs),
            how=how)

    region_center = np.array(region_center).reshape(-1, 1)
    scale_factor = np.array((x_factor, y_factor)).reshape(-1, 1)
    # map_data = map_data.scale(x_factor, y_factor, origin=region_center)

    # convert map data to graph
    graph = nx.MultiDiGraph()
    for road in map_data.geometry:
        if road.__class__.__name__ == 'LineString':
            graph = add_road_to_graph(road, graph, scale_factor, region_center)
        elif road.__class__.__name__ == 'MultiLineString':
            for sub_road in road.geoms:
                graph = add_road_to_graph(sub_road, graph, scale_factor, region_center)
        else:
            raise TypeError(f'road should be LineString or MultiLineString, got {road.__class__.__name__} instead')

    if simplify_graph and not ("simplified" in graph.graph and graph.graph["simplified"]):
        graph = osmnx.simplify_graph(graph, strict=True, remove_rings=True)

    if graph.nodes.__len__() < 32:
        # if the graph contain few roads, try to get the road net from open street map
        for i in range(5):
            try:
                graph = osmnx.graph_from_point(
                    center_point=(region_center[1], region_center[0]),
                    dist=max(rx * x_factor, ry * y_factor) * 1000,
                    network_type="drive",
                    simplify=simplify_graph,
                    retain_all=False,
                    truncate_by_edge=True, )

                # Scale down node values
                for node in graph.nodes:
                    graph.nodes[node]['x'] = (graph.nodes[node]['x'] - region_center[0]) * x_factor
                    graph.nodes[node]['y'] = (graph.nodes[node]['y'] - region_center[1]) * y_factor

                # Scale down edge values
                for u, v, k, data in graph.edges(keys=True, data=True):
                    data['length'] *= 0.001
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retrying in {8} seconds...")
                time.sleep(8)

    return graph


class LoadMultiGeoAndCreateCommutingPointsAndRoadMap(object):
    r"""
    Load multi geo data and create commuting points and road map.

    Args:
        :param barrier_file_name: file name of geo data with barriers (.shp)
        :param road_map_file_name: file name of geo data with road map (.shp)
        :param sampling_interval: sampling interval of commuting points (Km)
        :param how: operation of point set and geo set for creating commuting points (default: 'difference')
    """
    def __init__(self,
                 barrier_file_name: str,
                 road_map_file_name: str,
                 sampling_interval: float = 0.2,
                 how='intersection',
                 filter_road_with_id_hdc_g0=True,
                 simplify_graph=True,
                 ):
        self.sampling_interval = sampling_interval
        self.filter_road_with_id_hdc_g0 = filter_road_with_id_hdc_g0
        self.how = how
        self.simplify_graph = simplify_graph

        print(f'LoadMultiGeoAndCreateCommutingPointsAndRoadMap: Loading geo barrier data from {barrier_file_name} .. ')
        barrier_geo_data = geopandas.read_file(barrier_file_name)

        print(f'LoadMultiGeoAndCreateCommutingPointsAndRoadMap: Loading road map from {road_map_file_name} .. ')
        road_map_geo_data = geopandas.read_file(road_map_file_name)

        print(f'LoadMultiGeoAndCreateCommutingPoints: Loading finish ')

        self.geo_series = barrier_geo_data.geometry
        self.id_hdc_g0 = barrier_geo_data["ID_HDC_G0"].values
        self.crs = barrier_geo_data.crs
        self.num = len(self.geo_series)
        self.start = 0

        self.road_map_data = road_map_geo_data

    def __getitem__(self, item):
        if item.__class__.__name__ == 'slice':
            self.start = item.start
            return self
        if item.__class__.__name__ == 'int':
            item += self.start
            if item >= self.num:
                raise StopIteration
            else:
                return self.get_item(item)

    def get_item(self, item):
        print(f'\nTransform No. {item} Geo data And RoadMap, id_hdc_g0: {self.id_hdc_g0[item]} ...')
        
        # get geo region center and range
        current_geo = self.geo_series[item]
        region_center = (current_geo.centroid.x, current_geo.centroid.y)
        region_bounds = current_geo.bounds

        print(f'region_center = {region_center}')

        x_factor = 111.320 * math.cos(region_center[1] * 0.5 / 180 * math.pi)
        y_factor = 110.574

        # create commuting points
        commuting_points = create_commuting_points(
            x_factor, y_factor,
            region_bounds,
            self.sampling_interval, crs=self.crs)

        commuting_points = geopandas.overlay(
            commuting_points,
            geopandas.GeoDataFrame(geometry=geopandas.GeoSeries(current_geo),
                                   crs=self.crs), how=self.how)

        # get local road map
        if self.filter_road_with_id_hdc_g0:
            id_hdc = self.id_hdc_g0[item]
        else:
            id_hdc = None
        graph = create_road_map(
            x_factor, y_factor,
            region_center, region_bounds,
            current_geo,
            self.road_map_data,
            id_hdc,
            crs=self.crs, how=self.how,
            simplify_graph=self.simplify_graph)
        # get edges and nodes
        self.graph = graph
        traffic_x = nx.get_node_attributes(graph, 'x')
        traffic_x = np.array(list(traffic_x.values())).reshape((1, -1))
        traffic_y = nx.get_node_attributes(graph, 'y')
        traffic_y = np.array(list(traffic_y.values())).reshape((1, -1))
        traffic_nodes = np.vstack((traffic_x, traffic_y))
        traffic_nodes = torch.from_numpy(traffic_nodes.astype(np.float32))
        if traffic_nodes.size(1) > 0:
            road_net = nx.to_scipy_sparse_array(graph, weight='length').toarray().astype(np.float32)
            road_net = torch.from_numpy(road_net)
        else:
            road_net = torch.zeros(0)

        # convert to local coordinate
        region_center = torch.DoubleTensor(region_center)
        line_st, line_ed = geo_data_to_geo_lines(current_geo, region_center)
        commuting_points = geo_data_to_points(commuting_points, region_center)

        line_st[0, :].mul_(x_factor)
        line_st[1, :].mul_(y_factor)
        line_ed[0, :].mul_(x_factor)
        line_ed[1, :].mul_(y_factor)
        commuting_points[0, :].mul_(x_factor)
        commuting_points[1, :].mul_(y_factor)

        print('Geo Transform finish')

        return line_st, line_ed, commuting_points, traffic_nodes, road_net, \
               region_bounds, x_factor, y_factor, region_center


def load_road_map(
):
    import matplotlib.pyplot as plt
    from numpy import vstack
    def plot_line(p_st, p_nd, **args):
        p_st = p_st.numpy()
        p_nd = p_nd.numpy()
        plt.plot(vstack((p_st[0, :], p_nd[0, :])), vstack((p_st[1, :], p_nd[1, :])), **args)
    # load geo data
    # geo_data = LoadMultiGeoAndCreateCommutingPointsAndRoadMap(
    #     '/home/liweipeng/disk_sda2/Dataset/road map/detour测试/地理障碍3个城市.shp',
    #     '/home/liweipeng/disk_sda2/Dataset/road map/detour测试/路网三个城市.shp',
    #     0.25,
    #     simplify_graph=True,
    # )
    # geo_data = LoadMultiGeoAndCreateCommutingPointsAndRoadMap(
    #     '/home/liweipeng/disk_sda2/Dataset/road map/detour测试/地理障碍3个城市.shp',
    #     '/home/liweipeng/disk_sda2/Dataset/road map/detour测试/路网三个城市.shp',
    #     0.25)
    geo_data = LoadMultiGeoAndCreateCommutingPointsAndRoadMap(
        '/home/liweipeng/disk_sda2/Dataset/road map/城市样例-9.4/城市12279_12306_12320.shp',
        '/home/liweipeng/disk_sda2/Dataset/road map/城市样例-9.4/路网12279_12306_12320.shp',
        0.25)

    # graphical index for all regions
    st_id = 0
    for i, data in enumerate(geo_data[st_id:]):
        # if i >= 20:
        #     break
        # get data
        ii = i + st_id
        barrier_st, barrier_ed, commuting_points, traffic_nodes, road_net, region_bounds, x_factor, y_factor, region_center = data
        G = geo_data.graph
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_nodes(G, pos, node_size=10, node_shape=".",)
        nx.draw_networkx_edges(G, pos, arrows=False)
        plot_line(barrier_st, barrier_ed, color='b', linewidth=0.5)
        plt.axis('on')
        plt.show()  # 显示图形
        id_hdc_g0 = int(geo_data.id_hdc_g0[ii])

        # 误差分析
        dist = ((traffic_nodes[0, :].view(-1, 1) - traffic_nodes[0, :].view(1, -1)) ** 2 +
                (traffic_nodes[1, :].view(-1, 1) - traffic_nodes[1, :].view(1, -1)) ** 2) ** 0.5
        dist[road_net == 0] = 0
        er = (dist - road_net).abs()
        # plt.imshow(er)
        # plt.show()
        print(f'Mean error = {er[er > 0].mean()}')
        print(f'No. {ii} region')
    # 误差总体不大，完成测试


if __name__ == '__main__':
    load_road_map()
