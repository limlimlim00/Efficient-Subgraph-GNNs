import wandb
import logging

def set_wandb(cfg):
    num_layer = cfg.model.num_layer
    dim_embed = cfg.model.dim_embed
    mpnn_type = cfg.model.base_mpnn
    aggs = cfg.model.aggs
    
    lr = cfg.training.lr
    wd = cfg.training.wd
    epochs = cfg.training.epochs
    bs = cfg.data.bs
    data_name = cfg.data.name
    project_name = cfg.wandb.project_name
    seed = cfg.general.seed
    
    model_dropout = cfg.model.dropout
    tag = f"Aggs_{aggs}||MPNN_type_{mpnn_type}||Num_layers_{num_layer}||Bs_{bs}||dim_embed_{dim_embed}||SEED_{seed}||LR_{lr}||WD_{wd}||Epochs_{epochs}||Model_Dropout_p_{model_dropout}||Dataset_{data_name}"

    logging.info(f"{tag}")

    wandb.init(settings=wandb.Settings(
        start_method='thread'), project=project_name, name=tag, config=cfg)