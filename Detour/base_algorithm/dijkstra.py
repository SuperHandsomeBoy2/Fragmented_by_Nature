import torch
from torch import Tensor


def dijkstra_algorithm(
        distance_mat: Tensor,
        replace=True,
        min_k: int = None,
        max_k: int = None,
):
    r"""
    Floyd minimal distance / time algorithm
    :param distance_mat: adjacent distance matrix, distance of i->j, row -> col
    :param replace: replace original distance_mat with minimum distance
    :param min_k: minimum node id for iteration
    :param max_k: maximum node id for iteration
    :return:
    distance_mat: minimum distance of i->j, row -> col
    path: shortest path, where row i is the shortest path from node i to other nodes
    """
    print('Minimum distance with floyd algorithm')
    m, n = distance_mat.size()
    assert m == n

    if ~replace:
        distance_mat = distance_mat.clone()
    min_k = 0 if min_k is None else max(min_k, 0)
    max_k = n if max_k is None else min(max_k, n)

    path = torch.arange(n, device=distance_mat.device, dtype=torch.int16).unsqueeze(
        dim=1).repeat_interleave(n, 1)
    min_dist_mat = distance_mat[start_node, :].clone().to(device)

    for k in range(min_k, max_k, 1):
        if k % 100 == 0:
            print(f'step: {k} / {n}')
        dik = distance_mat[:, k]
        dkj = distance_mat[k, :]
        dij_new = dik.view(-1, 1) + dkj.view(1, -1)
        change_path = dij_new < distance_mat
        distance_mat[change_path] = dij_new[change_path]
        path[change_path] = path[k, :].unsqueeze(dim=0).expand_as(path)[change_path]
    print('Floyd algorithm finish')
    return distance_mat, path