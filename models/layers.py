import torch
import torch.nn as nn
import torch_geometric.nn as gnn 
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import logging
import torch.nn.functional as F
import torch_scatter as pys

# --------------------------------- Atom/Bond EMBEDDING -------------------------------- #
class Atom_coarsen(nn.Module):
    """Atom encoder

    Args:
        dim (int): embedding dimension
        dis (int): maximum encoding distance
        encode (bool): whether to use encoder

    """

    def __init__(self, cfg, dim: int, max_dis: int, encode: bool = True, 
                 use_linear: bool = False,
                 atom_dim: int = 6):
        super().__init__()
        self.cfg = cfg
        self.max_dis = max_dis
        # Warning: To account for infinite distance and for the padding value self.embed_values = max_dis + 2
        self.embed_values = self.max_dis + 2
        self.encode = encode
        if use_linear:
            logging.info("Using an MLP to encode atoms -- uses atom_dim variable.")
        else:
            logging.info("Using a look up table to encode atoms -- not uses atom_dim variable.")
        self.use_linear = use_linear
        self.embed_v = encode and AtomEncoder(dim)

        if use_linear: # linear layer
            self.embed_v = encode and nn.Sequential(nn.Linear(atom_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        else: # look up table
            self.embed_v = encode and AtomEncoder(dim)

        self.embed_d = nn.Embedding(self.embed_values+1, dim) # always self.embed_values + 1

    def forward(self, batch):
        if self.encode:
            if not self.use_linear:
                if batch.x.dtype == torch.float32:
                    batch.x = batch.x.int()
            x = self.embed_v(batch.x)
        else:
            x = 0
        
        # Warning: Assumes the value of 1001 accounts for 2 nodes which are unreachable from each other!
        d = Atom_coarsen.custom_clamp(
            tensor=batch.d, min_val=None, max_val=self.max_dis)
        d = d[:, :self.cfg.data.preprocess.max_spd_elements]
        d = self.embed_d(d)
        d = torch.sum(d, dim=1) # Warning: any permutation inv function along the first dimention is ok
        
        batch.x = x + d

        del batch.d

        return batch
    @staticmethod
    def custom_clamp(tensor, min_val, max_val):
        # First, clamp all values between min_val and max_val
        clamped_tensor = torch.clamp(tensor, min_val, max_val)
        
        # Create a mask for values that are greater than 1000 in the original tensor
        mask = tensor >= 1001
        
        # Update the elements where the condition in 'mask' is True
        clamped_tensor[mask] = max_val + 1
        
        mask = tensor >= 1002
        # Update the elements where the condition in 'mask' is True
        clamped_tensor[mask] = max_val + 2
        
        return clamped_tensor


class Bond(nn.Module):
    """Bond encoder

    Args:
        dim (int): embedding dimension

    """

    def __init__(self, dim: int, 
                linear: bool = False,
                linear_in_dim: int = 4):
        super().__init__()
        self.linear = linear
        if self.linear:
            logging.info("Using an MLP to encode bonds -- uses atom_dim variable.")
            self.bond_encoder = nn.Sequential(nn.Linear(linear_in_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        else:
            logging.info("Using a look up table to encode bonds -- not uses edge_attr_dim variable.")

        self.embed = BondEncoder(dim)
        

    def forward(self, message, attrs, dont_use_message=False, peptides_flag = False):
        if attrs is None:
            return F.relu(message)

        
        if self.linear:
            attr_of_each_edge = attrs
            if peptides_flag:
                return self.bond_encoder(attr_of_each_edge.float())
            else:
                return self.bond_encoder(attr_of_each_edge)


        if dont_use_message:
            attr_of_each_edge = attrs
            if attr_of_each_edge.shape[-1] == 1:
                return F.relu(self.embed(attr_of_each_edge)) # Graphs x Embs
            else: 
                return F.relu(self.embed(attr_of_each_edge.to(torch.int)[:, None]).mean(dim=1)) # Graphs x Embs

        else:
            attr_of_each_edge = attrs
            return F.relu(message + self.embed(attr_of_each_edge))



# --------------------------------- MPNN block -------------------------------- #

class MPNN_block(torch.nn.Module):
    def __init__(self, d, agg, point_encoder, H=1, d_output=64, edge_dim=64, type='Gat', use_linear=False):
        super(MPNN_block, self).__init__()
        self.agg = agg
        self.point_encoder = point_encoder
        self.use_linear = use_linear
        self.type = type
        self.H = H
        self.d_output = d_output
        self.edge_dim = edge_dim
        self.d = d
        assert self.d_output == (
            (self.d_output // self.H) * self.H), f"Invalid self.d_output value. Expected: {(self.d_output // self.H) * self.H}, but got: {self.d_output}."
        if self.type == 'GatV2':
            self.layer = gnn.GATv2Conv(
                in_channels=self.d, out_channels=self.d_output // self.H, heads=self.H, edge_dim=self.edge_dim)
        elif self.type == 'Transformer_conv':
            self.layer = gnn.TransformerConv(
                in_channels=self.d, out_channels=self.d_output // self.H, heads=self.H, edge_dim=self.edge_dim)
        elif self.type == 'Gat':
            self.layer = gnn.GATConv(
                in_channels=self.d, out_channels=self.d_output // self.H, heads=self.H, edge_dim=self.edge_dim)
        elif self.type == 'Gin':
            self.eps = nn.Parameter(torch.zeros(1))
            if use_linear:
                if self.agg == "point":
                    self.coupling_layer = LINEAR(self.d, self.d_output)
                self.layer = LINEAR(self.d, self.d_output)
            else:
                if self.agg == "point":
                    self.coupling_layer = MLP(idim=self.d, odim=self.d_output, hdim=48)
                self.layer = MLP(self.d, self.d_output)
        elif self.type == "Gcn":
            self.layer = gnn.GCNConv(
                in_channels=self.d, out_channels=self.d_output)
        else:
            raise ValueError(f"{type} is not a valid model.")


    def forward(self, x, edge_index, edge_attr):
        if self.type == "Gat" or self.type == "Transformer_conv" or self.type == "GatV2":
            x, _ = self.layer(
                x=x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=False)
        elif self.type == 'Gin':
            # self
            self_element = x * (1.0 + self.eps) # (1 + \epsilon) * x
            # x_message
            dst, src = edge_index
            x_message = torch.index_select(x, dim=0, index=src) # x(src)
            if edge_attr is not None:
                g_message = edge_attr # g(src)
                message = x_message + g_message  # x(src) + g(src)
                if self.agg == "point":
                    if self.point_encoder == "RELU":
                        message = F.relu(message)
                    elif self.point_encoder == "MLP":
                        message = self.coupling_layer(
                            message)  # MLP[x(src) + g(src)]
                    elif self.point_encoder == "NONE":
                        message = message
                    else:
                        raise ValueError(
                            f"Invalid point encoder: {self.point_encoder}")
                    # message = self.coupling_layer(message)  # MLP[x(src) + g(src)]
                else:
                    message = F.relu(message)
                x_agg = pys.scatter(message, dim=0, index=dst, dim_size=len(x)) # sum and put in correct slots - MLP[x(src) + g(src)]
            else:
                message = x_message
                x_agg = pys.scatter(message, dim=0, index=dst, dim_size=len(x))
            x = self_element + x_agg
            x = self.layer(x)
            
            ###############
            # agg_elemet = aggregate(x=x, edge_index=edge_index, edge_attr=edge_attr)
            # x = self_element + agg_elemet
            # x = self.layer(x)
        elif self.type == 'Gcn':
            x = get_x_with_its_neighbours_edge_attr(x=x,
                edge_index=edge_index, edge_attr=edge_attr)
            x  = self.layer(
                x=x, edge_index=edge_index, edge_weight=None)            
        else:
            raise ValueError(f"{self.type} is not a valid model.")
        return x


# --------------------------- MPNN block - helpers -------------------------- #


def aggregate(x, edge_index, edge_attr):
    dst, src = edge_index
    x_message = torch.index_select(x, dim=0, index=src)
    x_recieve = pys.scatter(x_message, dim=0, index=dst, dim_size=len(x))
    x = get_x_with_its_neighbours_edge_attr(
        x=x_recieve, edge_index=edge_index, edge_attr=edge_attr)
    return x
    

def get_x_with_its_neighbours_edge_attr(x, edge_index, edge_attr):
    dst, _ = edge_index
    if edge_attr is not None:
        dst, _ = edge_index
        edge_message = pys.scatter(
            edge_attr, dim=0, index=dst, dim_size=len(x))
        x = x + edge_message
    return x


# --------------------------------- General Layers -------------------------------- #

class NormReLU(nn.Sequential):

    def __init__(self, dim: int):
        super().__init__()

        self.add_module("bn", nn.BatchNorm1d(dim))
        self.add_module("ac", nn.ReLU())


class MLP(nn.Sequential):

    def __init__(self, idim: int, odim: int, hdim: int = None, norm: bool = True):
        super().__init__()
        hdim = hdim or idim

        self.add_module("input", nn.Linear(idim, hdim))
        self.add_module("input_nr", NormReLU(hdim) if norm else nn.ReLU())
        self.add_module("output", nn.Linear(hdim, odim))

class LINEAR(nn.Sequential):

    def __init__(self, idim: int, odim: int):
        super().__init__()

        self.add_module("input", nn.Linear(idim, odim))


class Pooling(nn.Module):
    """Final pooling

    Args:
        idim (int): input dimension
        odim (int): output dimension

    """

    def __init__(self, idim: int, odim: int):
        super().__init__()

        self.predict = MLP(idim, odim, hdim=idim*2, norm=False)

    def forward(self, batch, subgraph_rep = None, efficient=False):
        if efficient: # Warning: this is mean pool
            return self.predict(gnn.global_mean_pool(subgraph_rep, batch.batch))
        else:
            return self.predict(gnn.global_mean_pool(batch.x, batch.batch)) # Warning: this is sum pool


class Equiv_layer_encoder(torch.nn.Module):

    def __init__(self, edge_in_dim, edge_out_dim):
        super(Equiv_layer_encoder, self).__init__()
        in_dims = [
            2,
            edge_in_dim,
            edge_in_dim,
            edge_in_dim,
            4,
            4,
        ]
        self.bond_embedding_list = torch.nn.ModuleList()

        for i in range(6):
            emb = torch.nn.Embedding(in_dims[i], edge_out_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding
