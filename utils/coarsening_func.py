import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from torch_geometric.data import Data
import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import logging

def identity_coarsen(graph: Data) -> Data:
    co_edge_attr = graph.edge_attr
    if co_edge_attr.dim() == 1:
        co_edge_attr = co_edge_attr.reshape(-1, 1)

    cluster_to_nodes_map = {i: [i] for i in range(graph.num_nodes)}

    coarsened = Data(
        edge_index=graph.edge_index,
        edge_attr=co_edge_attr,
        num_nodes=graph.num_nodes,
        super_nodes=cluster_to_nodes_map
    )
    return coarsened

# ===================================================================================================================== #
# ========================================           (1) k-means               ======================================== #
# ===================================================================================================================== #


def get_coarsened_graph(graph, n_clusters=2, type='K-mean', dim_laplacian=2):
    if graph.edge_attr.dim() == 1:
        graph.edge_attr = graph.edge_attr.reshape(-1, 1)
    if type.lower() == "k-mean":  # Case-insensitive comparison
        adjacency_matrix = to_dense_adj(
            edge_index=graph.edge_index, max_num_nodes=graph.num_nodes).numpy()[0]
        labels, new_edge_index_tensor, new_edge_attr_tensor, cluster_to_nodes_map = get_coarsened_graph_attributes_K_means(
            edge_index=graph.edge_index, edge_attr=graph.edge_attr, adjacency_matrix=adjacency_matrix, n_clusters=n_clusters, dim_laplacian=dim_laplacian)

        coarsened_edge_index = new_edge_index_tensor
        coarsened_edge_attr = new_edge_attr_tensor

        coarsened_num_nodes = n_clusters

        graph = Data(edge_index=coarsened_edge_index,
                     edge_attr=coarsened_edge_attr,
                     num_nodes=coarsened_num_nodes,
                     super_nodes=cluster_to_nodes_map)
    else:
        raise ValueError("Bad type for graph coarsening")
    return graph

# ================================== helpers ================================== #


def get_coarsened_graph_attributes_K_means(edge_index, edge_attr, adjacency_matrix, n_clusters, dim_laplacian):
    labels = spectral_clustering(adjacency_matrix, n_clusters, dim_laplacian)
    _, new_edge_index_tensor, new_edge_attr_tensor = create_graph_based_on_clustering(
        edge_index, edge_attr, n_clusters, adjacency_matrix, labels)


    # Initialize a dictionary to map new cluster indices to original node indices
    cluster_to_nodes_map = {i: [] for i in range(n_clusters)}

    # Populate the mapping
    for node_index, cluster_label in enumerate(labels):
        cluster_to_nodes_map[cluster_label].append(
            node_index)

    return labels, new_edge_index_tensor, new_edge_attr_tensor, cluster_to_nodes_map



def spectral_clustering(adjacency_matrix, n_clusters=2, dim_laplacian=2):
    if adjacency_matrix.shape[0] == 0: # graph has one node
        labels = np.array([0])
        return labels
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
        

    try:
        eigenvalues, eigenvectors = eigh(
            laplacian_matrix, subset_by_index=[1, dim_laplacian])
        X = eigenvectors
    except Exception as e:
        # logging.info(f"dim laplacian is to big: {e}\n\n computing all eigenvectors. \n\n\n")
        eigenvalues, eigenvectors = eigh(
            laplacian_matrix)
        X = eigenvectors[:, 1:dim_laplacian]



    if X.shape == (1,0):
        logging.info(
            f"This graph has 1 node, can't slice this way: X = eigenvectors[:, 1:dim_laplacian]. \n\n\n")
        eigenvalues, eigenvectors = eigh(
            laplacian_matrix)
        X = eigenvectors
        
    kmeans = KMeans(n_clusters=n_clusters, n_init=1)  # "auto" 지원 X
    kmeans.fit(X)
    labels = kmeans.labels_
    return labels


def create_graph_based_on_clustering(edge_index, edge_attr, n_clusters, adjacency_matrix, labels):
    try:
        edge_attr = edge_attr.reshape(-1, len(edge_attr[0]))
    except Exception as e:
        logging.info(
            f"The graph has one node with no connectivity! Returning an empty edge index/edge attr for the coarsened graph")
        new_edge_index_tensor = None
        new_edge_attr_tensor = None
        return labels, new_edge_index_tensor, new_edge_attr_tensor
    new_edge_index = []
    new_edge_attr = []
    for edge, attr in zip(edge_index.T, edge_attr):
        dst, src = edge[0], edge[1]
        if labels[dst] != labels[src]:
            new_edge_index.append([labels[dst], labels[src]])
            new_edge_attr.append(attr.tolist())

    if len(new_edge_index) != 0:
        new_edge_index_tensor = torch.tensor(new_edge_index).T
        new_edge_attr_tensor = torch.tensor(new_edge_attr)
    else:
        new_edge_index_tensor = None
        new_edge_attr_tensor = None
    
    return labels, new_edge_index_tensor, new_edge_attr_tensor



# ===================================================================================================================== #
# ========================================             preproccess                 ==================================== #
# ===================================================================================================================== #

# ================================== SPD pre-proccess ================================== #
def get_spd_for_product_graph(graph, coarsened_graph, n_clusters, MAX_SPD_DIM, pad_value=-1, INF_VALUE=1001.0):
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
            spd_values = pad_tensor(
                spd_values, MAX_SPD_DIM=MAX_SPD_DIM, pad_value=pad_value)
            list_of_spds.append(spd_values)

    apsp = torch.cat(list_of_spds, dim=0).reshape(
        n_clusters, graph.num_nodes, MAX_SPD_DIM).reshape(-1, MAX_SPD_DIM)
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


def pad_tensor(tensor, MAX_SPD_DIM=10, pad_value=0):
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
    padding_needed = MAX_SPD_DIM - tensor.size(0)

    # Assert that the new length is greater than or equal to the tensor's current length
    assert padding_needed >= 0, "Super node is to large - increase MAX_SPD_DIM!"

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


def get_vL_edge_index_and_attr(edge_index_coarsened_graph, num_nodes_in_original_graph):
    induces_edges_from_coarsen_edges = []
    for edge in edge_index_coarsened_graph.T:
        induced_edges = edge.reshape(2, 1) * num_nodes_in_original_graph + torch.arange(
            num_nodes_in_original_graph).repeat(2).reshape(2, -1)
        induces_edges_from_coarsen_edges.append(induced_edges)
    # Concatenate all matrices horizontally (along columns)
    big_edge_index = torch.hstack(induces_edges_from_coarsen_edges)
    return big_edge_index

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
    return edge_index, edge_attr


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


if __name__ == "__main__":
    ## TEST coarsening function ##
    num_nodes = 6
    # Node features
    node_features = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=torch.float)

    # Adjecency
    adjacency_matrix = np.array([
        [0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ])

    # Find the non-zero entries in the adjacency matrix (edges)
    edges = np.nonzero(adjacency_matrix)

    # Convert to a PyTorch tensor
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.long)
    # Convert the adjacency matrix to edge index
    edge_index, edge_weight = dense_to_sparse(adjacency_matrix)

    def create_tensor(N, K, R):
        """
        Create a tensor of dimensions N x K with integer values ranging from 0 to R.

        Args:
        N (int): Number of rows in the tensor.
        K (int): Number of columns in the tensor.
        R (int): Maximum value in the tensor (inclusive).

        Returns:
        torch.Tensor: An N x K tensor with integers from 0 to R.
        """
        return torch.randint(low=0, high=R+1, size=(N, K))

    edge_attr = create_tensor(N=edge_index.shape[1], K=3, R=5)

    n_clusters = 4

    ## TEST coarsening function ##

    # Nodes
    num_nodes = 6

    # Node features
    node_features = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=torch.float)

    # edge_index
    adjacency_matrix = np.array([
        [0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ])

    # Find the non-zero entries in the adjacency matrix (edges)
    edges = np.nonzero(adjacency_matrix)

    # Convert to a PyTorch tensor
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.long)
    # Convert the adjacency matrix to edge index
    edge_index, edge_weight = dense_to_sparse(adjacency_matrix)

    def create_tensor(N, K, R):
        """
        Create a tensor of dimensions N x K with integer values ranging from 0 to R.

        Args:
        N (int): Number of rows in the tensor.
        K (int): Number of columns in the tensor.
        R (int): Maximum value in the tensor (inclusive).

        Returns:
        torch.Tensor: An N x K tensor with integers from 0 to R.
        """
        return torch.randint(low=0, high=R+1, size=(N, K))
    edge_attr = create_tensor(N=edge_index.shape[1], K=3, R=5)
    # Creating the graph with node features
    graph = Data(x=node_features, edge_index=edge_index,
                 edge_attr=edge_attr, num_nodes=num_nodes)

    coarsened_graph = get_coarsened_graph(
        graph=graph, n_clusters=n_clusters, type='K-mean')