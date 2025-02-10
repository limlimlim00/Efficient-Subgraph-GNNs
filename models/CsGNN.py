import torch
import torch.nn as nn
from models import layers
import logging
import torch_scatter as pys
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class Coarsen_based_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # preprocess
        self.max_dis = cfg.data.preprocess.max_dis
        
        # atom encoder
        self.atom_dim = cfg.model.atom_encoder.in_dim
        self.use_linear_atom_encoder = cfg.model.atom_encoder.linear

        # edge encoder
        self.use_linear_edge_encoder = cfg.model.edge_encoder.linear
        self.use_edge_attr_uL = cfg.model.edge_encoder.use_edge_attr_uL
        self.use_edge_attr_vL = cfg.model.edge_encoder.use_edge_attr_vL
        self.edge_attr_dim = cfg.model.edge_encoder.in_dim

        # model
        self.num_layers = cfg.model.num_layer
        self.aggs = cfg.model.aggs
        self.H = cfg.model.H
        self.final_dim = cfg.model.final_dim
        self.dropout = cfg.model.dropout
        self.use_residual = cfg.model.residual
        self.use_linear = cfg.model.layer_encoder.linear
        self.base_mpnn = cfg.model.base_mpnn
        self.use_sum = True
        

        # pooling
        self.use_sum_pooling = cfg.model.sum_pooling

        # general
        self.dataset = cfg.data.name

        logging.info("Checking dimentions.")
        self.post_concat_dim_embed, self.each_agg_dim_embed = Coarsen_based_model.compute_final_embedding_dimension(
            dim_embed=cfg.model.dim_embed, num_aggs=len(self.aggs), H=self.H)
        
        self.each_agg_dim_embed = self.post_concat_dim_embed

        logging.info("Initializing Atom encoder + NM")
        self.atom_encoder = self.get_preprocess_layer(
            dim=self.post_concat_dim_embed, max_dis=self.max_dis, use_linear=self.use_linear_atom_encoder, atom_dim=self.atom_dim)
                
        logging.info(f"Initializing all {self.num_layers} layers")
        # MPNN (local/global)
        self.MPNNs = nn.ModuleList()
        self.EDGE_ENCODER = nn.ModuleList()
        # LAYER AGG (point || local || global)
        self.CAT_ENCODERs = nn.ModuleList()
        self.BNORM_RELUs = nn.ModuleList()
        # DROPOUT
        self.DROP_OUTs = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            logging.info(f"Initializing layer number {layer_idx}.")
            # MPNN (local/global)
            MPNN_i = {}
            EDGE_ENCODER_i = {}

        
            for agg in self.aggs:
                logging.info(f"Initializing aggregation {agg}.")
                if self.is_point_agg(agg): # point agg
                    edge_in_dim = cfg.data.preprocess.global_attr_max_val + 1
                    edge_out_dim = self.get_edge_encoder_out_dim(layer_idx=layer_idx)
                    # edge_encoder_i = nn.Embedding(edge_in_dim, edge_out_dim) 
                    edge_encoder_i = layers.Equiv_layer_encoder(
                        edge_in_dim, edge_out_dim)
                    EDGE_ENCODER_i[agg] = edge_encoder_i
                else: # main connectivity
                    if self.use_edge_encoder(agg=agg):
                        edge_out_dim = self.get_edge_encoder_out_dim(layer_idx=layer_idx)
                        edge_encoder_i = self.init_edge_encoder(use_linear=self.use_linear_edge_encoder, in_dim=self.edge_attr_dim, out_dim=edge_out_dim)
                        EDGE_ENCODER_i[agg] = edge_encoder_i
                    else:
                        edge_out_dim = None
                        EDGE_ENCODER_i[agg] = None
                mpnn_i = layers.MPNN_block(H = self.H, d=self.post_concat_dim_embed, d_output=self.each_agg_dim_embed, edge_dim=edge_out_dim, type=self.base_mpnn, use_linear=self.use_linear, agg=agg, point_encoder=self.cfg.model.point_encoder)
                MPNN_i[agg] = mpnn_i
                    
            # MPNN (local/global)
            self.MPNNs.append(nn.ModuleDict(MPNN_i))
            self.EDGE_ENCODER.append(nn.ModuleDict(EDGE_ENCODER_i))
            
            self.BNORM_RELUs.append(layers.NormReLU(self.post_concat_dim_embed))
            if self.dropout > 0:
                self.DROP_OUTs.append(nn.Dropout(p=self.dropout))

        logging.info(f"Initializing pooling")
        self.POOLING = layers.Pooling(
            self.post_concat_dim_embed, self.final_dim)

    def forward(self, batch):
        # ATOM ENCODER + NM
        batch = self.atom_encoder(batch)

        # LAYERS
        for layer_idx in range(self.num_layers):
            # AGGS
            all_aggs = []
            for agg in self.aggs:
                encoded_edge_atr = self.get_edge_attr(
                        agg=agg, batch=batch, edge_encoder=self.EDGE_ENCODER[layer_idx][agg])
                edge_index = batch[f"index_{agg}"]
                agg_element = self.MPNNs[layer_idx][agg](
                    x=batch.x, edge_index=edge_index, edge_attr=encoded_edge_atr)
                all_aggs.append(agg_element)
            
            # SUM ALL AGGS
            if len(all_aggs) == 2:
                all_aggs_sum = all_aggs[0] + all_aggs[1]
            else:  # len(all_aggs) = 2
                all_aggs_sum = all_aggs[0] + all_aggs[1] + all_aggs[2]
            # BN_RELU
            batch_x = self.BNORM_RELUs[layer_idx](all_aggs_sum)

            # DROPOUT
            if self.dropout > 0:
                batch_x = self.DROP_OUTs[layer_idx](batch_x)
            # RESIDUAL
            if self.use_residual:
                batch.x = batch_x + batch.x
            else:
                batch.x = batch_x
        # POOL
        pool_value = self.pooling_forward(
            batch=batch, use_sum_pooling=self.use_sum_pooling)
        return pool_value

    # ============================= forward - helpers ============================= #
    def pooling_forward(self, batch, use_sum_pooling):
        if use_sum_pooling:
            batch.x = self.aggregate(graph=batch, agg="pool", encode=None)
            global_pool = self.POOLING(batch)
            return global_pool
        else:
            subgraph_rep = self.aggregate(
                graph=batch, agg="pool", encode=None, pool_efficiently=True)
            global_pool_efficient = self.POOLING(
                batch=batch, subgraph_rep=subgraph_rep, efficient=True)
            return global_pool_efficient
        
    def get_edge_attr(self, agg, batch, edge_encoder):
        if not self.use_edge_encoder(agg=agg):
            edge_attr = None
            return edge_attr
        edge_attr = batch.get(f"attr_{agg}", None)
        if edge_attr != None:
            if ("point" in agg):  # global
                edge_attr = edge_encoder(edge_attr.to(torch.int))
            else: # uL/vL
                edge_attr = edge_encoder(
                    message=-1, attrs=edge_attr, dont_use_message=True)

        return edge_attr
    
    def aggregate(self, graph, agg, encode=None, pool_efficiently=False):
        if pool_efficiently:
            dst, src = graph[f"index_{agg}"]
        else:
            dst, src = graph[f"index_{agg}"]

        message = torch.index_select(graph.x, dim=0, index=src)
        if encode is not None:
            message = encode(message, graph[f"attrs_{agg}"])

        return pys.scatter(message, dim=0, index=dst, dim_size=len(graph.x))
    
    # ============================= init - helpers ============================= #
    def is_point_agg(self, agg):
        return ("point" in agg)
    
    def use_edge_encoder(self, agg):
        if ("uL" in agg):
            if self.use_edge_attr_uL:
                return True
            else:
                return False
        elif ("vL" in agg):
            if self.use_edge_attr_vL:
                return True
            else:
                return False
        elif ("point" in agg):
            return True
        else:
            return False

    def get_edge_encoder_out_dim(self, layer_idx):
        edge_out_dim = self.post_concat_dim_embed
        if layer_idx == 0 and self.dataset == "alchemy":
            edge_out_dim = 6
        return edge_out_dim
    
    def init_edge_encoder(self, use_linear, in_dim, out_dim):
        edge_encoder = layers.Bond(
            dim=out_dim, linear=use_linear, linear_in_dim=in_dim)
        return edge_encoder
    
    def get_cat_encoder(self, use_linear, in_dim, out_dim):
        if use_linear:
            cat_encoder = layers.LINEAR(in_dim, out_dim)
        else:
            cat_encoder = layers.MLP(in_dim, out_dim)
        return cat_encoder
    
    # ============================= general - helpers ============================= #

    def get_preprocess_layer(self, dim, max_dis, use_linear, atom_dim):
        nm_dim = dim
        self.atom_encoder = layers.Atom_coarsen(cfg=self.cfg,
            dim=nm_dim, max_dis=max_dis, encode=True, use_linear=use_linear, atom_dim=atom_dim)

        return self.atom_encoder

    @staticmethod
    def compute_final_embedding_dimension(dim_embed, num_aggs, H=1):
        each_head_dim = (dim_embed // num_aggs) // H
        each_agg_dim_embed = each_head_dim * H
        post_concat_dim_embed = each_agg_dim_embed * num_aggs
        if dim_embed != post_concat_dim_embed:
            logging.info(
                "Modified the embedding dim to fit the concatenation and the heads!\n")

            logging.info(
                f"Original embedding final dimension of layers concatenated: {dim_embed}")
            logging.info(
                f"Modified embedding final dimension of layers concatenated: {post_concat_dim_embed}")
            logging.info(f"Each agg embedding size: {each_agg_dim_embed}")
            logging.info(f"Number of aggs: {num_aggs}")

            logging.info(f"Each head embedding size: {each_head_dim}")
            logging.info(f"Number of heads: {H}")
            # logging.info(
            #     f"Asserting that \n   1)  {H=} * {each_head_dim=} = {each_agg_dim_embed=}  \nand\n   2)  {each_agg_dim_embed=} * {num_aggs=} = {post_concat_dim_embed=}"
            # )

            assert H * each_head_dim == each_agg_dim_embed, (
                f"Dimension mismatch: Expected {H * each_head_dim} but got {each_agg_dim_embed}"
            )

            assert each_agg_dim_embed * num_aggs == post_concat_dim_embed, (
                f"Dimension mismatch: Expected {each_agg_dim_embed * num_aggs} but got {post_concat_dim_embed}"
            )
        return post_concat_dim_embed, each_agg_dim_embed



def get_model_params(model, dim_embed, AtomEncoder=AtomEncoder, BondEncoder=BondEncoder):

    try:
        total_params = sum(param.numel() for param in model.parameters())

        unused_atom_embed_params = sum(
            sum(param.numel() for param in m.parameters()) - 30 * dim_embed
            for m in model.modules() if isinstance(m, AtomEncoder)
        )

        unused_bond_embed_params = sum(
            sum(param.numel() for param in m.parameters()) - 5 * dim_embed
            for m in model.modules() if isinstance(m, BondEncoder)
        )

        unused_params = unused_atom_embed_params + unused_bond_embed_params

        return total_params, unused_params, total_params - unused_params

    except Exception as e:
        print(
            f"An error occurred when caculating the model size: {e}!\n Skipping this part.")
        # Optionally, you can log the error for debugging purposes
        print(f"\n")
        return -1, -1, -1
