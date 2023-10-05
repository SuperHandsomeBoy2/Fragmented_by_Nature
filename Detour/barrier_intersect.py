import torch
from torch import Tensor
from tqdm import tqdm


def cross_product(
        vec1_x: Tensor,
        vec1_y: Tensor,
        vec2_x: Tensor,
        vec2_y: Tensor,
):
    """
    计算叉积
    :param vec1_x:
    :param vec1_y:
    :param vec2_x:
    :param vec2_y:
    :return:
    """
    return vec1_x * vec2_y - vec1_y * vec2_x


def intersect_map_add_road(
        original_points: Tensor,
        new_roads: Tensor,
        barrier_st: Tensor,
        barrier_ed: Tensor,
        b_sz: int = 64,
):
    """
    通过计算线段外积,分析任意点列中所有连接线段与barrier之间的相交关系，
    进而判断任意连接是否有barrier

    方法：使用外积判断跨线，进而判断相交关系
    :param original_points: 点列，所有点之间相连
    :param new_roads: nodes of new road
    :param barrier_st:
    :param barrier_ed:
    :param b_sz: block size
    :return:
    """
    print('Get mask from intersect')
    # nodes
    n_barrier = barrier_st.size()[1]
    n_point = original_points.size()[1]
    n_road = new_roads.size()[1]

    px = original_points[0, :]
    py = original_points[1, :]

    nr_x = new_roads[0, :]
    nr_y = new_roads[1, :]

    # 外积
    # a: barrier start
    # b: barrier end
    # c: original nodes
    # d: new road nodes
    a_d = cross_product(barrier_st[0, :].view(-1, 1), barrier_st[1, :].view(-1, 1),
                        nr_x.view(1, -1), nr_y.view(1, -1)).view(n_barrier, 1, n_road)
    a_c = cross_product(barrier_st[0, :].view(-1, 1), barrier_st[1, :].view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(n_barrier, n_point, 1)
    c_d = cross_product(px.view(-1, 1), py.view(-1, 1),
                        nr_x.view(1, -1), nr_y.view(1, -1)).view(1, n_point, n_road)
    b_d = cross_product(barrier_ed[0, :].view(-1, 1), barrier_ed[1, :].view(-1, 1),
                        nr_x.view(1, -1), nr_y.view(1, -1)).view(n_barrier, 1, n_road)
    b_c = cross_product(barrier_ed[0, :].view(-1, 1), barrier_ed[1, :].view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(n_barrier, n_point, 1)
    a_b = cross_product(barrier_st[0, :], barrier_st[1, :],
                        barrier_ed[0, :], barrier_ed[1, :], ).view(n_barrier, 1, 1)

    # block intersect and >0 intersect map
    output = None
    for i_st in tqdm(range(0, n_barrier, b_sz)):
        i_end = min(i_st + b_sz, n_barrier)

        point_barrier = (a_d[i_st: i_end, :, :] - a_c[i_st: i_end, :, :] - c_d).sgn().to(torch.int8) * \
                        (b_d[i_st: i_end, :, :] - b_c[i_st: i_end, :, :] - c_d).sgn().to(torch.int8) <= 0
        barrier_point = (a_d[i_st: i_end, :, :] - b_d[i_st: i_end, :, :] -
                         a_b[i_st: i_end, :, :]).sgn().to(torch.int8) * \
                        (a_c[i_st: i_end, :, :] - b_c[i_st: i_end, :, :] -
                         a_b[i_st: i_end, :, :]).sgn().to(torch.int8) < 0
        if output is None:
            output = (point_barrier & barrier_point).any(dim=0)
        else:
            output = output | ((point_barrier & barrier_point).any(dim=0))
    print('intersect_map: finish')
    return output


if __name__ == '__main__':
    # test_intersect()
    # test_matrix_intersect()
    # test_matrix_point_intersect()
    pass
