import torch
import numpy as np
import torch_geometric as pyg
import torch_geometric.data as data
from utils.coarsening_func import get_coarsened_graph
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import math
import logging

def product_graph_construction(cfg):

    MAX_SPD_DIM = 100
    uL = "uL" in cfg.model.aggs  # internal
    vL = "vL" in cfg.model.aggs  # external
    global_agg = "global" in cfg.model.aggs
    point_agg = "point" in cfg.model.aggs
    loint_agg = "loint" in cfg.model.aggs

    def call(graph):
        n_clusters = cfg.data.preprocess.n_cluster
        if n_clusters >= graph.num_nodes:
            n_clusters = graph.num_nodes
        coarsened_graph = get_coarsened_graph(
            graph=graph, n_clusters=n_clusters, dim_laplacian=cfg.data.preprocess.dim_laplacian)
        graph_product_node_indices = torch.arange(
            n_clusters*graph.num_nodes).view((n_clusters, graph.num_nodes))
        product_graph_features = torch.cat([graph.x] * n_clusters, dim=0)
        
        # ======================== SPD ======================== #
        apsp = get_spd_for_product_graph(
            graph=graph, coarsened_graph=coarsened_graph, n_clusters=n_clusters, INF_VALUE=cfg.data.preprocess.inf_value, pad_value=cfg.data.preprocess.pad_value, MAX_SPD_DIM=MAX_SPD_DIM)  # Nodes x d
        
        # ======================== edges ======================== #
        uL_edge_index = uL_edge_attr = None
        vL_edge_index = vL_edge_attr = None
        global_edge_index = global_edge_attr = None
        point_edge_index = point_edge_attr = None
        loint_edge_index = loint_edge_attr = None
        
        if uL:  # internal
            uL_edge_index, uL_edge_attr = get_uL_edge_index_and_attr(
                edge_index_original_graph=graph.edge_index,
                edge_attr_original_graph=graph.edge_attr,
                num_supernodes=n_clusters, num_nodes_in_original_graph=graph.num_nodes)

        if vL:  # external
            vL_edge_index, vL_edge_attr = get_vL_edge_index_and_attr(
                edge_index_coarsened_graph=coarsened_graph.edge_index, edge_attr_coarsened_graph=coarsened_graph.edge_attr,
                num_nodes_in_original_graph=graph.num_nodes)
            if vL_edge_index == None: # Warning: if vL_edge_index is empty, uses vL_edge_index again!
                vL_edge_index = uL_edge_index
                vL_edge_attr = uL_edge_attr

        if global_agg: # global
            global_edge_index, global_edge_attr = get_global_edge_index_and_attr(
                n_clusters=n_clusters, num_nodes=graph.num_nodes, cluster_to_nodes_map=coarsened_graph.super_nodes)
            global_edge_attr = global_edge_attr.to(torch.int) - 1
            global_edge_attr = torch.clamp(
                global_edge_attr, 0, cfg.data.preprocess.global_attr_max_val)
        else:
            global_edge_index, global_edge_attr = None, None
            
        if point_agg:  # point
            point_edge_index, point_edge_attr = get_point_edge_index_and_attr(
                n_clusters=n_clusters, num_nodes=graph.num_nodes, cluster_to_nodes_map=coarsened_graph.super_nodes)
            point_edge_attr = point_edge_attr.to(torch.int) - 1
            point_edge_attr = torch.clamp(
                point_edge_attr, 0, cfg.data.preprocess.global_attr_max_val)
        else:
            point_edge_index, point_edge_attr = None, None

        if loint_agg:
            loint_edge_index, loint_edge_attr = get_loint_edge_index_and_attr(original_edge_index =graph.edge_index,
                n_clusters=n_clusters, num_nodes=graph.num_nodes, cluster_to_nodes_map=coarsened_graph.super_nodes)
            loint_edge_attr = loint_edge_attr.to(torch.int) - 1
            loint_edge_attr = torch.clamp(
                loint_edge_attr, 0, cfg.data.preprocess.global_attr_max_val)
        else:
            loint_edge_index, loint_edge_attr = None, None

            
        # ======================== pooling ======================== #
        if cfg.model.sum_pooling: # sum pooling
            pool_index = edge_index_for_sum_pool(graph_product_node_indices=graph_product_node_indices)
        else: # mean pooling
            pool_index = edge_index_for_mean_pool(n_clusters=n_clusters, num_nodes=graph.num_nodes)
            
        data_dict = {
            "y": graph.y,
            
            # node indices
            "node_indices": graph_product_node_indices.reshape(-1),
            # node features
            "x": product_graph_features,
            
            # node marking
            "d": apsp,

            # internal edges
            "index_uL": uL_edge_index,
            "attr_uL": uL_edge_attr,

            # external edges
            "index_vL": vL_edge_index,
            "attr_vL": vL_edge_attr,
            
            # global edges
            "index_global": global_edge_index,
            "attr_global": global_edge_attr,
            
            # point edges
            "index_point": point_edge_index,
            "attr_point": point_edge_attr,
            
            # loint edges
            "index_loint": loint_edge_index,
            "attr_loint": loint_edge_attr,
    
            # pool
            "index_pool": pool_index
        }

        return data.Data(**data_dict)

    return call

# ================================== SPD pre-proccess ================================== #


def get_spd_for_product_graph(graph, coarsened_graph, n_clusters, pad_value=-1, INF_VALUE=1001.0, MAX_SPD_DIM=100):
    # Warning: Assumes the value of INF_VALUE (1001) accounts for 2 nodes which are unreachable from each other!
    adj_of_origianl_graph = to_dense_adj(
        graph.edge_index, max_num_nodes=graph.num_nodes).squeeze(0)
    apsp = get_all_pairs_shortest_paths(adj=adj_of_origianl_graph)

    # Replaces all 'inf' values with 1001
    apsp[torch.isinf(apsp)] = INF_VALUE
    apsp = apsp.to(int)

    list_of_spds = []
    for super_node in range(n_clusters):
        super_nodes_internal_nodes = coarsened_graph.super_nodes[super_node]
        for node in range(graph.num_nodes):
            spd_values = apsp[super_nodes_internal_nodes, node]
            spd_values, _ = spd_values.sort()
            spd_values = spd_values[:MAX_SPD_DIM]
            # Warning: (1) 100 is the pad size
            # Warning: (2) stores 100 spd's
            spd_values = pad_tensor(
                spd_values, pad_size=MAX_SPD_DIM, pad_value=pad_value)
            list_of_spds.append(spd_values)

    apsp = torch.cat(list_of_spds, dim=0).reshape(
        n_clusters, graph.num_nodes, MAX_SPD_DIM).reshape(-1, MAX_SPD_DIM)
    apsp = apsp[:, :MAX_SPD_DIM]
    return apsp

# ============================= helpers ============================= #


def get_all_pairs_shortest_paths(adj):
    spd = torch.where(~torch.eye(len(adj), dtype=bool) & (adj == 0),
                      torch.full_like(adj, float("inf")), adj)
    # Floyd-Warshall

    for k in range(len(spd)):
        dist_from_source_to_k = spd[:, [k]]
        dist_from_k_to_target = spd[[k], :]
        dist_from_source_to_target_via_k = dist_from_source_to_k + dist_from_k_to_target
        spd = torch.minimum(spd, dist_from_source_to_target_via_k)
    return spd


def pad_tensor(tensor, pad_size=10, pad_value=0):
    """
    Pads a 1D tensor to a new length with a specified pad value.

    Parameters:
    - tensor (torch.Tensor): The 1D tensor to pad.
    - new_length (int): The desired length of the output tensor.
    - pad_value (numeric, optional): The value to pad with. Default is 0.

    Returns:
    - torch.Tensor: The padded tensor.
    """
    # Calculate the number of padding values needed
    padding_needed = pad_size - tensor.size(0)

    # Assert that the new length is greater than or equal to the tensor's current length
    assert padding_needed >= 0, "Super node is to large - increase pad_size!"

    # Check if padding is needed
    if padding_needed > 0:
        # Create a tensor of padding values
        padding = torch.full((padding_needed,), pad_value,
                             dtype=tensor.dtype, device=tensor.device)
        # Concatenate the original tensor with the padding tensor
        padded_tensor = torch.cat((tensor, padding), dim=0)
        return padded_tensor
    else:
        return tensor

# ================================== uL (internal) edges pre-proccess ================================== #


def get_uL_edge_index_and_attr(edge_index_original_graph, edge_attr_original_graph, num_supernodes, num_nodes_in_original_graph):
    try:
        edge_attr_original_graph = edge_attr_original_graph.reshape(
            edge_attr_original_graph.shape[0], -1)
    except Exception as e:
        logging.info(
            f"The graph has no connectivity! Returning an edge index with 1 self loop on the 0-th indexed node, and a dummpy edge attr! -- This should only be on molesol!")
        big_edge_index = torch.tensor([[0],
                                       [0]])
        big_edge_attr = torch.tensor([[0,0,0]])
        return big_edge_index, big_edge_attr

    edge_indices_per_row = []
    for super_node_idx in range(num_supernodes):
        # Calculate the offset for the current supernode's edge indices
        offset = super_node_idx * num_nodes_in_original_graph
        # Apply the offset to the original graph's edge indices and store the result
        edge_indices_per_row.append(offset + edge_index_original_graph)

    # Concatenate all matrices horizontally (along columns)
    big_edge_index = torch.hstack(edge_indices_per_row)
    big_edge_attr = edge_attr_original_graph.repeat(num_supernodes, 1)
    return big_edge_index, big_edge_attr

# ================================== vL (external) edges pre-proccess ================================== #


def get_vL_edge_index_and_attr(edge_index_coarsened_graph, edge_attr_coarsened_graph, num_nodes_in_original_graph):
    if edge_index_coarsened_graph == None:
        return None, None
    induces_edges_from_coarsen_edges = []
    induces_edges_attrs_from_coarsen_edges = []
    for edge, attr in zip(edge_index_coarsened_graph.T, edge_attr_coarsened_graph):
        induced_edges = edge.reshape(2, 1) * num_nodes_in_original_graph + torch.arange(
            num_nodes_in_original_graph).repeat(2).reshape(2, -1)
        induces_edges_from_coarsen_edges.append(induced_edges)
        induced_edges_attr = attr.repeat(induced_edges.shape[1], 1) 
        induces_edges_attrs_from_coarsen_edges.append(induced_edges_attr)
    # Concatenate all matrices horizontally (along columns)
    if len(induces_edges_from_coarsen_edges) != 0:
        big_edge_index = torch.hstack(induces_edges_from_coarsen_edges)
        big_edge_attr = torch.vstack(induces_edges_attrs_from_coarsen_edges)
    else:
        big_edge_index = None
        big_edge_attr = None
    
    return big_edge_index, big_edge_attr

# ================================== Global edges pre-proccess ================================== #


def get_global_edge_index_and_attr(n_clusters, num_nodes, cluster_to_nodes_map):
    edge_index = []
    edge_attr = []
    for dst_cluster_idx in range(n_clusters):
        for dst_node_idx in range(num_nodes):
            S = set(cluster_to_nodes_map[dst_cluster_idx])
            i = dst_node_idx
            dst_node_index_in_prod_graph = dst_cluster_idx * num_nodes + dst_node_idx
            for src_cluster_idx in range(n_clusters):
                for src_node_idx in range(num_nodes):
                    S_prime = set(cluster_to_nodes_map[src_cluster_idx])
                    i_prime = src_node_idx
                    src_node_index_in_prod_graph = src_cluster_idx * num_nodes + src_node_idx
                    edge_index.append([
                        dst_node_index_in_prod_graph,
                        src_node_index_in_prod_graph
                    ])
                    A1, A2, A3, A4, A5, A6 = get_alpha_indices(
                        S=S, S_prime=S_prime, i=i, i_prime=i_prime)
                    edge_attr.append([A1, A2, A3, A4, A5, A6])
    edge_index = edges_to_tensor(edge_index)
    edge_attr = edge_attrs_to_tensor(edge_attr)
    return edge_index, edge_attr


def get_point_edge_index_and_attr(n_clusters, num_nodes, cluster_to_nodes_map):
    edge_index = []
    edge_attr = []
    for dst_cluster_idx in range(n_clusters):
        for dst_node_idx in range(num_nodes):
            S = set(cluster_to_nodes_map[dst_cluster_idx])
            i = dst_node_idx
            dst_node_index_in_prod_graph = dst_cluster_idx * num_nodes + dst_node_idx
            for src_cluster_idx in range(n_clusters):
                for src_node_idx in range(num_nodes):
                    S_prime = set(cluster_to_nodes_map[src_cluster_idx])
                    i_prime = src_node_idx
                    if not is_root(S=S_prime, i=i_prime):
                        continue
                    if not (i_prime == i):
                        continue
                     
                    src_node_index_in_prod_graph = src_cluster_idx * num_nodes + src_node_idx
                    edge_index.append([
                        dst_node_index_in_prod_graph,
                        src_node_index_in_prod_graph
                    ])
                    A1, A2, A3, A4, A5, A6 = get_alpha_indices(
                        S=S, S_prime=S_prime, i=i, i_prime=i_prime)
                    edge_attr.append([A1, A2, A3, A4, A5, A6])
    edge_index = edges_to_tensor(edge_index)
    edge_attr = edge_attrs_to_tensor(edge_attr)
    return edge_index, edge_attr


def get_loint_edge_index_and_attr(original_edge_index, n_clusters, num_nodes, cluster_to_nodes_map):
    edge_index = []
    edge_attr = []
    for dst_cluster_idx in range(n_clusters):
        for dst_node_idx in range(num_nodes):
            S = set(cluster_to_nodes_map[dst_cluster_idx])
            i = dst_node_idx
            dst_node_index_in_prod_graph = dst_cluster_idx * num_nodes + dst_node_idx
            for src_cluster_idx in range(n_clusters):
                for src_node_idx in range(num_nodes):
                    S_prime = set(cluster_to_nodes_map[src_cluster_idx])
                    i_prime = src_node_idx
                    # if (root is source AND its the same colomn) OR (its a local edge OR)
                    if (is_root(S=S_prime, i=i_prime) and (i_prime == i)) or is_local_edge(i=i, i_prime=i_prime, original_edge_index=original_edge_index):
                        src_node_index_in_prod_graph = src_cluster_idx * num_nodes + src_node_idx
                        edge_index.append([
                            dst_node_index_in_prod_graph,
                            src_node_index_in_prod_graph
                        ])
                        A1, A2, A3, A4, A5, A6 = get_alpha_indices(
                            S=S, S_prime=S_prime, i=i, i_prime=i_prime)
                        edge_attr.append([A1, A2, A3, A4, A5, A6])
                    else: 
                        continue
    edge_index = edges_to_tensor(edge_index)
    edge_attr = edge_attrs_to_tensor(edge_attr)
    return edge_index, edge_attr

def is_local_edge(i, i_prime, original_edge_index):
    edge = torch.tensor([
                        [i],
                        [i_prime]
                    ])
    edge_expanded = edge.repeat(1, original_edge_index.size(1))
    matches = (original_edge_index == edge_expanded).all(dim=0)
    edge_exists = matches.any()
    return edge_exists
    
    
def is_root(S, i):
    return (i in S)

def get_alpha_indices(S, S_prime, i, i_prime):
    A1 = get_A1(i=i, i_prime=i_prime)
    A2 = get_A2_3(S=S)
    A3 = get_A2_3(S=S_prime)
    A4 = get_A4(S=S, S_prime=S_prime)
    A5 = get_A5_6(S=S, S_prime=S_prime, i=i, i_prime=i_prime)
    A6 = get_A5_6(S=S_prime, S_prime=S, i=i, i_prime=i_prime)
    return A1, A2, A3, A4, A5, A6


def get_A1(i, i_prime):
    if i == i_prime:
        return 1
    else:  # i != j
        return 2


def get_A2_3(S):
    return len(S)


def get_A4(S, S_prime):
    return len(S.intersection(S_prime)) + 1


def get_A5_6(S, S_prime, i, i_prime):
    if i in S and i_prime in S_prime:
        return 1
    elif i in S and i_prime not in S_prime:
        return 2
    elif i not in S and i_prime in S_prime:
        return 3
    else:  # i not in S and i_prime not in S_prime:
        return 4

def edges_to_tensor(edges):
    # Convert the list of edges to a tensor of shape (E, 2)
    edge_tensor = torch.tensor(edges, dtype=torch.long)
    
    # Transpose the tensor to shape (2, E)
    edge_tensor = edge_tensor.t()
    
    return edge_tensor

def edge_attrs_to_tensor(edge_attrs):
    # Convert the list of edge attributes to a tensor of shape (E, D)
    edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float)
    
    return edge_attr_tensor


# ================================== Pool pre-proccess ================================== #

def edge_index_for_mean_pool(n_clusters, num_nodes):
    src_nodes = torch.arange(n_clusters*num_nodes)
    target_nodes = torch.repeat_interleave(
        torch.arange(n_clusters), num_nodes)
    
    index_mean_pool = torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)
    return index_mean_pool

def edge_index_for_sum_pool(graph_product_node_indices):
    target_nodes, src_nodes = torch.broadcast_tensors(
        graph_product_node_indices[:, :, None], graph_product_node_indices[:, None, :])
    index_sum_pool = torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)
    return index_sum_pool