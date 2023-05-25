import heapq
import numpy as np
import argparse
import time


def solve(args):
    start_time = time.time()
    # 网络类型
    network_type = args.network_type
    # 读取结点数量 n 和边的数量 m
    with open(args.network_file, "r") as f:
        n = int(f.readline().split("=")[1].strip())
        m = int(f.readline().split("=")[1].strip())
    # 读取起始点
    with open(args.origin_file, "r") as f:
        # 跳过第一行
        f.readline()
        origin = int(f.readline().strip())
    # 读取网络文件中的 Forward star representation
    network = np.loadtxt(
        args.network_file,
        skiprows=3,
        delimiter=",",
        dtype=np.int32,
    )
    # 提取所有的 node id
    node_id = np.unique(network[:, 0:2])
    # 按照 network 中的第一列排序，得到 network
    network = network[network[:, 0].argsort()]
    # 定义 point 数组，表示每个 node 在 network from 中首次出现的索引
    point = np.empty(n + 1, dtype=np.int32)
    point[:-1] = np.searchsorted(network[:, 0], node_id)
    point[-1] = m
    # 初始化
    i = origin
    # 最终输出结果，为一个 n * 3 的矩阵，三列分别为：起始点编号，终止点编号，最短路径
    output = np.empty((n, 3), dtype=np.int32)  # 创建一个 n*3 的矩阵
    output[:, 0] = origin  # 将第一列的所有元素设置为 origin
    output[:, 1] = np.arange(1, n + 1)  # 将第二列的所有元素设置为 range(1, n+1)
    output[:, 2] = np.iinfo(np.int32).max  # 将第三列的所有元素设置为 int32 的最大值
    output[i - 1, 2] = 0  # 将第 origin 到 origin 的距离设置为 0
    # Temporarily labeled set，用小顶堆实现
    t = [(0, i)]  # 将 (0, i) 放入小顶堆中
    t_set = set([i])  # 创建一个集合，包含 t 中的 node
    # Permanently labeled set
    p = set()  # 创建一个空集合
    # 对于 one-to-all 问题，当 Permanently labeled set 中的元素个数等于 n 时，结束循环
    while len(p) != n:
        # Node selection
        # 从 t 中取出最小的元素
        d_i, i = heapq.heappop(t)
        # 将 i 从 t_set 中移除
        t_set.remove(i)
        # 将 i 加入到 p 中
        p.add(i)
        # 从 network 中取出 i 的邻接点
        start_point = point[i - 1]
        end_point = point[i]
        # 提取 i 的所有邻接点 j_s
        j_s = network[start_point:end_point, 1]
        # 提取 i 到 所有邻接点 j_s 的距离
        d_ij_s = network[start_point:end_point, 2]
        # 判断 d_i + d_ij_s 是否小于当前的最短距离，若是则需要更新它们的最短距离
        is_less = d_i + d_ij_s < output[j_s - 1, 2]
        # 判断 j_s[less] 中的元素是否在 t_set 中
        is_in_t_set = list(map(lambda x: x in t_set, j_s[is_less]))
        # 如果 j_s[less] 中的元素不在 t_set 中，则将 j 加入到 t_set 中
        t_set.update(j_s[is_less][list(map(lambda x: not x, is_in_t_set))])
        # 如果 j_s[less] 中的元素在 t_set 中，则将原有的 (d_j, j) 移除
        for j in j_s[is_less][is_in_t_set]:
            t.remove((output[j - 1, 2], j))
        # 更新 j 的最短距离：d_j = min(d_j, d_i+d_ij)
        output[(j_s - 1)[is_less], 2] = d_i + d_ij_s[is_less]
        # 将所有 j_s[is_less] 的元素及其最短距离加入到 t 中
        for j in j_s[is_less]:
            heapq.heappush(t, (output[j - 1, 2], j))
    end_time = time.time()
    # 将结果导出到 22210690089_output_{args.network_type}.txt
    np.savetxt(
        f"22210690089_output_{args.network_type}.txt",
        output,
        fmt="%d",
        delimiter=",",
        header="origin,destination,distance",
        comments="",
    )
    # 将运行时间导出到 22210690089_time_{args.network_type}.txt
    with open(f"22210690089_time_{args.network_type}.txt", "w") as f:
        f.write(f"{(end_time - start_time) * 1000:.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("22210690089")
    parser.add_argument(
        "network_type",
        type=str,
        default="grid",
        choices=[
            "grid",
            "random",
            "dense",
        ],
    )
    parser.add_argument(
        "network_file",
        type=str,
        default="grid_150_150.txt",
    )
    parser.add_argument(
        "origin_file",
        type=str,
        default="origin_grid_150_150.txt",
    )
    args = parser.parse_args()
    solve(args)
