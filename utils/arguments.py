import yaml
from easydict import EasyDict as edict
import argparse

# load config path
_config_parser = argparse.ArgumentParser(add_help=False)
_config_parser.add_argument("--config", type=str, default="./config/config.yaml",
                            help="Path to YAML config file")
_CONFIG_ARGS, _ = _config_parser.parse_known_args()
CONFIG_PATH = _CONFIG_ARGS.config

# --------------------------------------------------------------------------------- #
#                               Main functions                                      #
# --------------------------------------------------------------------------------- #

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)


def override_config_with_args(cfg, return_nested=True):

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--general__seed", type=int,
                        default=cfg.general.seed, help="seed")
    parser.add_argument("--general__device", type=int,
                        default=cfg.general.device, help="device")

    # data
    parser.add_argument("--data__name", type=str,
                        default=cfg.data.name, help="name")
    parser.add_argument("--data__bs", type=int, default=cfg.data.bs, help="bs")
    parser.add_argument("--data__num_workers", type=int,
                        default=cfg.data.num_workers, help="num_workers")
    
    # data -> preprocess
    parser.add_argument("--data__preprocess__max_dis", type=int,
                    default=cfg.data.preprocess.max_dis, help="max_dis")
    parser.add_argument("--data__preprocess__inf_value", type=int,
                        default=cfg.data.preprocess.inf_value, help="inf_value - value for 2 nodes which are unreachable from each other")
    parser.add_argument("--data__preprocess__max_spd_elements", type=int,
                        default=cfg.data.preprocess.max_spd_elements, help="max_spd_elements")
    parser.add_argument("--data__preprocess__pad_value", type=int,
                        default=cfg.data.preprocess.pad_value, help="pad_value")
    parser.add_argument("--data__preprocess__global_attr_max_val", type=int,default=cfg.data.preprocess.global_attr_max_val, help="global_attr_max_val")
    parser.add_argument("--data__preprocess__n_cluster", type=int,
                        default=cfg.data.preprocess.n_cluster, help="n_cluster")
    parser.add_argument("--data__preprocess__dim_laplacian", type=int,
                        default=cfg.data.preprocess.dim_laplacian, help="n_cluster")
    



    # model
    parser.add_argument("--model__residual", type=str2bool,
                        default=cfg.model.residual, help="residual")
    parser.add_argument("--model__aggs", type=list,
                        default=cfg.model.aggs, help="aggs")
    parser.add_argument("--model__sum_pooling", type=str2bool,
                        default=cfg.model.sum_pooling, help="sum_pooling")




    parser.add_argument("--model__num_layer", type=int,
                        default=cfg.model.num_layer, help="num_layer")
    parser.add_argument("--model__dim_embed", type=int,
                        default=cfg.model.dim_embed, help="dim_embed")
    parser.add_argument("--model__final_dim", type=int,
                        default=cfg.model.final_dim, help="final_dim")
    parser.add_argument("--model__dropout", type=float,
                        default=cfg.model.dropout, help="dropout")
    parser.add_argument("--model__base_mpnn", type=str,
                        default=cfg.model.base_mpnn, help="base_mpnn")
    parser.add_argument("--model__H", type=int,
                        default=cfg.model.H, help="H")
    parser.add_argument("--model__point_encoder", type=str,
                        default=cfg.model.point_encoder, help="point_encoder")
    
    
    # model -> atom_encoder
    parser.add_argument("--model__atom_encoder__in_dim", type=int,
                        default=cfg.model.atom_encoder.in_dim, help="atom_encoder_in_dim")
    parser.add_argument("--model__atom_encoder__linear", type=str2bool,
                        default=cfg.model.atom_encoder.linear, help="atom_encoder_linear")
    

    # model -> edge_encoder
    parser.add_argument("--model__edge_encoder__in_dim", type=int,
                        default=cfg.model.edge_encoder.in_dim, help="edge_encoder_in_dim")
    parser.add_argument("--model__edge_encoder__linear", type=str2bool,
                        default=cfg.model.edge_encoder.linear, help="edge_encoder_linear")
    parser.add_argument("--model__edge_encoder__use_edge_attr_vL", type=str2bool, default=cfg.model.edge_encoder.use_edge_attr_vL, help="edge_encoder_use_edge_attr")
    
    parser.add_argument("--model__edge_encoder__use_edge_attr_uL", type=str2bool,default=cfg.model.edge_encoder.use_edge_attr_uL, help="edge_encoder_use_edge_attr")

    # model -> layer_encoder
    parser.add_argument("--model__layer_encoder__linear", type=str2bool,
                        default=cfg.model.layer_encoder.linear, help="layer_encoder_linear")


    # training
    parser.add_argument("--training__lr", type=float,
                        default=cfg.training.lr, help="lr")
    parser.add_argument("--training__wd", type=float,
                        default=cfg.training.wd, help="wd")
    parser.add_argument("--training__epochs", type=int,
                        default=cfg.training.epochs, help="epochs")
    parser.add_argument("--training__patience", type=int,
                        default=cfg.training.patience, help="patience")
    parser.add_argument("--training__warmup", type=int,
                        default=cfg.training.warmup, help="warmup")

    # wandb
    parser.add_argument("--wandb__project_name", type=str,
                        default=cfg.wandb.project_name, help="project_name")
    # ================================================================================================ #
    # ================================================================================================ #
    # ================================================================================================ #
    args, _ = parser.parse_known_args()

    # general
    if args.general__seed != cfg.general.seed:
        cfg.general.seed = args.general__seed
    if args.general__device != cfg.general.device:
        cfg.general.device = args.general__device

    # data
    if args.data__name != cfg.data.name:
        cfg.data.name = args.data__name
    if args.data__bs != cfg.data.bs:
        cfg.data.bs = args.data__bs
    if args.data__num_workers != cfg.data.num_workers:
        cfg.data.num_workers = args.data__num_workers
    #       data -> preprocess
    if args.data__preprocess__max_dis != cfg.data.preprocess.max_dis:
        cfg.data.preprocess.max_dis = args.data__preprocess__max_dis
    if args.data__preprocess__inf_value != cfg.data.preprocess.inf_value:
        cfg.data.preprocess.inf_value = args.data__preprocess__inf_value
    if args.data__preprocess__max_spd_elements != cfg.data.preprocess.max_spd_elements:
        cfg.data.preprocess.max_spd_elements = args.data__preprocess__max_spd_elements
    if args.data__preprocess__pad_value != cfg.data.preprocess.pad_value:
        cfg.data.preprocess.pad_value = args.data__preprocess__pad_value
    if args.data__preprocess__global_attr_max_val != cfg.data.preprocess.global_attr_max_val:
        cfg.data.preprocess.global_attr_max_val = args.data__preprocess__global_attr_max_val
    if args.data__preprocess__n_cluster != cfg.data.preprocess.n_cluster:
        cfg.data.preprocess.n_cluster = args.data__preprocess__n_cluster
    if args.data__preprocess__dim_laplacian != cfg.data.preprocess.dim_laplacian:
        cfg.data.preprocess.dim_laplacian = args.data__preprocess__dim_laplacian
        

    # model
    if args.model__residual != cfg.model.residual:
        cfg.model.residual = args.model__residual
    if args.model__aggs != cfg.model.aggs:
        cfg.model.aggs = args.model__aggs
    if args.model__sum_pooling != cfg.model.sum_pooling:
        cfg.model.sum_pooling = args.model__sum_pooling
    if args.model__base_mpnn != cfg.model.base_mpnn:
        cfg.model.base_mpnn = args.model__base_mpnn
    if args.model__num_layer != cfg.model.num_layer:
        cfg.model.num_layer = args.model__num_layer
    if args.model__dim_embed != cfg.model.dim_embed:
        cfg.model.dim_embed = args.model__dim_embed
    if args.model__final_dim != cfg.model.final_dim:
        cfg.model.final_dim = args.model__final_dim
    if args.model__dropout != cfg.model.dropout:
        cfg.model.dropout = args.model__dropout
    if args.model__H != cfg.model.H:
        cfg.model.H = args.model__H
    if args.model__point_encoder != cfg.model.point_encoder:
        cfg.model.point_encoder = args.model__point_encoder

    #       model -> atom_encoder
    if args.model__atom_encoder__in_dim != cfg.model.atom_encoder.in_dim:
        cfg.model.atom_encoder.in_dim = args.model__atom_encoder__in_dim
    if args.model__atom_encoder__linear != cfg.model.atom_encoder.linear:
        cfg.model.atom_encoder.linear = args.model__atom_encoder__linear
    #       model -> edge_encoder
    if args.model__edge_encoder__in_dim != cfg.model.edge_encoder.in_dim:
        cfg.model.edge_encoder.in_dim = args.model__edge_encoder__in_dim
    if args.model__edge_encoder__linear != cfg.model.edge_encoder.linear:
        cfg.model.edge_encoder.linear = args.model__edge_encoder__linear
    if args.model__edge_encoder__use_edge_attr_uL != cfg.model.edge_encoder.use_edge_attr_uL:
        cfg.model.edge_encoder.use_edge_attr_uL = args.model__edge_encoder__use_edge_attr_uL
    if args.model__edge_encoder__use_edge_attr_vL != cfg.model.edge_encoder.use_edge_attr_vL:
        cfg.model.edge_encoder.use_edge_attr_vL = args.model__edge_encoder__use_edge_attr_vL
    #       model -> layer_encoder
    if args.model__layer_encoder__linear != cfg.model.layer_encoder.linear:
        cfg.model.layer_encoder.linear = args.model__layer_encoder__linear


    # training
    if args.training__lr != cfg.training.lr:
        cfg.training.lr = args.training__lr
    if args.training__wd != cfg.training.wd:
        cfg.training.wd = args.training__wd
    if args.training__epochs != cfg.training.epochs:
        cfg.training.epochs = args.training__epochs
    if args.training__patience != cfg.training.patience:
        cfg.training.patience = args.training__patience
    if args.training__warmup != cfg.training.warmup:
        cfg.training.warmup = args.training__warmup
        
    # wandb
    if args.wandb__project_name != cfg.wandb.project_name:
        cfg.wandb.project_name = args.wandb__project_name

    if return_nested:
        return cfg
    else:
        return vars(args)


# --------------------------------------------------------------------------------- #
#                           Helpers for Main functions                              #
# --------------------------------------------------------------------------------- #

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')