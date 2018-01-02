#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-11-3, 09:23

@Description:

@Update Date: 17-11-3, 09:23
"""

from pro.util import *


# 统计部分class的边，点，出入度
def obtain_part_graph(rg, rg_node, classes, region=None):
    assert isinstance(classes, list)
    if region is not None:
        l_b_lon = region[0][0]
        l_b_lat = region[0][1]
        r_u_lon = region[1][0]
        r_u_lat = region[1][1]

    rg = rg[np.in1d(rg["class"], classes)]
    part_nodes = np.union1d(rg.s_id.unique(), rg.e_id.unique())
    in_degree = rg.groupby("s_id").count()["edge_id"].reset_index()
    in_degree.columns = ["node_id", "in_degree"]
    out_degree = rg.groupby("e_id").count()["edge_id"].reset_index()
    out_degree.columns = ["node_id", "out_degree"]
    nodes = pd.DataFrame({"node_id": part_nodes})
    nodes = pd.merge(nodes, in_degree, on="node_id", how="left")
    nodes = pd.merge(nodes, out_degree, on="node_id", how="left")
    nodes = nodes.fillna(0)
    nodes = pd.merge(nodes, rg_node, on="node_id", how="left")
    if region is not None:
        nodes = nodes[(nodes.node_lon > l_b_lon) &
                      (nodes.node_lon < r_u_lon) &
                      (nodes.node_lat > l_b_lat) &
                      (nodes.node_lat < r_u_lat)]

        rg = rg[rg.s_id.isin(nodes.node_id) & rg.e_id.isin(nodes.node_id)]

    print "all edges shape", rg.shape[0]
    print "all node shape", nodes.shape[0]
    print "in degree < 2 and out degree < 2 shape", nodes[(nodes.in_degree < 2) & (nodes.out_degree < 2)].shape[0]
    return rg, nodes


# 在子路网中寻找连通子图
def find_connected_graph(part_rg,
                         part_nodes,
                         start_node_index=0):
    _edges = set()
    _nodes = set()
    _new_nodes = list()
    _new_nodes.append(part_nodes.iloc[start_node_index]["node_id"])

    def add_new_node(x, nodes, new_nodes):
        if x not in nodes:
            nodes.add(x)
            new_nodes.append(x)

    while (len(_new_nodes) != 0):
        _new_node_id = _new_nodes.pop()
        _nodes.add(_new_node_id)
        # 找和新节点相连的边
        part_rg[part_rg.s_id == _new_node_id]["edge_id"].map(lambda x: _edges.add(x))
        part_rg[part_rg.e_id == _new_node_id]["edge_id"].map(lambda x: _edges.add(x))

        # 和新节点相连的点
        part_rg[part_rg.s_id == _new_node_id]["e_id"].map(lambda x: add_new_node(x, _nodes, _new_nodes))
        part_rg[part_rg.e_id == _new_node_id]["s_id"].map(lambda x: add_new_node(x, _nodes, _new_nodes))
    part_nodes = part_nodes[np.in1d(part_nodes.node_id, np.array(list(_nodes)))]
    part_rg = part_rg[np.in1d(part_rg.edge_id, np.array(list(_edges)))]
    return part_rg, part_nodes


def extract_subgraph(rg_all, rg_node_all, classes, region, save_path, suffix="part"):
    part_rg, part_nodes = obtain_part_graph(rg_all,
                                            rg_node_all,
                                            classes,
                                            region)  # ,"0x04","0x06","0x08","0x09","0x0a","0x0b"

    part_rg, part_nodes = find_connected_graph(part_rg, part_nodes, 0)
    print "after find connected graph"
    print "edges shape", part_rg.shape[0]
    print "nodes shape", part_nodes.shape[0]
    if save_path is not None and save_path != "":
        rg_path = os.path.join(save_path, "R_G_{}.csv".format(suffix))
        part_rg.to_csv(rg_path)
        rg_node_path = os.path.join(save_path, "R_G_node_{}.csv".format(suffix))
        part_nodes.to_csv(rg_node_path)
    return part_rg, part_nodes

def main():
    rg = load_all_RG()
    rg_node = load_all_RG_node()
    extract_subgraph(rg,
                     rg_node,
                     ["0x00", "0x01", "0x02", "0x03"],
                     region1,
                     "data",
                     "0123class_region1")


if __name__ == '__main__':
    main()
