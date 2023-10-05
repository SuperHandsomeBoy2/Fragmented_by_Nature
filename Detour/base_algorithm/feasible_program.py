# 生成架桥/隧道的可选连接方案

import torch
from torch import Tensor


def feasible_program(
        candidate_points: Tensor,
        threshold: float = 0.4,
):
    r"""
    get candidate points
    :param candidate_points: order points
    :param threshold: threshold for distance of points around barrier
    :return:
    """
    n = candidate_points.size()[-1]
    id_n = torch.arange(n, device=candidate_points.device, dtype=torch.float32)
    x = candidate_points[0, :]
    y = candidate_points[1, :]

    dist = ((x.view(-1, 1) - x.view(1, -1)).pow(2) + (y.view(-1, 1) - y.view(1, -1)).pow(2)).sqrt()
    id_dist = (id_n.view(1, -1) - id_n.view(-1, 1)).abs()
    mask = id_dist > 0.5 * n
    id_dist[mask] = n - id_dist[mask]

    threshold = min(max(threshold, 0.01), 0.9)
    mask = (dist < dist.mean() + 3 * dist.std()) & (id_dist > 0.5 * n * threshold)
    mask = mask & (id_n.view(1, -1) > id_n.view(-1, 1))
    mask = mask.view(1, n, n).expand((2, n, n))
    return candidate_points.view(2, -1, 1).expand((2, n, n))[mask].view(2, -1), \
           candidate_points.view(2, 1, -1).expand((2, n, n))[mask].view(2, -1)


def test():
    sita = torch.range(0, 2 * torch.pi, 0.2)
    r = torch.rand(sita.size()) * 2 + 0.5
    p = torch.cat([(r * sita.sin()).view(1, -1), (2 * r * sita.cos()).view(1, -1)], dim=0)
    p_st, p_ed = feasible_program(p, 0.5)
    print(p_st.size()[-1])


if __name__ == '__main__':
    test()
