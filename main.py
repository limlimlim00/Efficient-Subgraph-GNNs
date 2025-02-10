import utils.arguments as u_args
import torch
import utils.my_logging as u_log
import utils.data as u_data
import utils.webiases as u_wandb
import utils.training as u_train
import logging
import wandb
import models.CsGNN as Sub
from tqdm import tqdm
import utils.training as u_train
import numpy as np


# ===================================================================================================================== #
# ========================================               logging               ======================================== #
# ===================================================================================================================== #
u_log.setup_logging()
# ===================================================================================================================== #
# ========================================               load args             ======================================== #
# ===================================================================================================================== #
logging.info(f"Loading arguments.")
cfg_file__path = './config/config.yaml'
cfg = u_args.load_config(path=cfg_file__path)
cfg = u_args.override_config_with_args(cfg=cfg)
cfg.data.dir = "./datasets/" + cfg.data.name

# ===================================================================================================================== #
# ===================================               set device and seed             =================================== #
# ===================================================================================================================== #
logging.info(f"Setting device and seed.")

device = torch.device(f"cuda:{cfg.general.device}")
torch.manual_seed(cfg.general.seed)

# ===================================================================================================================== #
# ========================================               dataloaders             ====================================== #
# ===================================================================================================================== #
logging.info(f"Loading the {cfg.data.name} dataset; using Pre-transform")

dataloader, num_elements_in_target = u_data.get_dataloader(cfg)


# ===================================================================================================================== #
# ========================================               model              =========================================== #
# ===================================================================================================================== #
logging.info(f"Loading the model.")

model = Sub.Coarsen_based_model(cfg)

total_params, unused_params, used_params = Sub.get_model_params(
    model=model, dim_embed=cfg.model.dim_embed)
cfg.model.params = {
    "total_params": total_params,
    "unused_params": unused_params,
    "used_params": used_params
}

logging.info(f"model size = {total_params}")
logging.info(f"unused parameters = {unused_params}")
logging.info(f"used parameters = {total_params - unused_params}")

model = model.to(device)

# ===================================================================================================================== #
# ========================================               load wandb             ======================================= #
# ===================================================================================================================== #
logging.info(f"Setting wandb.")

u_wandb.set_wandb(cfg=cfg)
wandb.watch(model)
# ===================================================================================================================== #
# ========================================              training            =========================================== #
# ===================================================================================================================== #
logging.info(f"Loading loss function.")

critn, goal, task = u_train.get_loss_func(cfg=cfg)
assert task in ['regression', 'classification'], \
    f"Invalid task type: {task}. Expected 'regression' or 'classification'."

logging.info(f"Loading optimizer.")
optim = u_train.get_optim_func(cfg=cfg, model=model)


logging.info(f"Loading schedular.")
sched = u_train.get_sched_func(cfg=cfg, optim=optim, warmup_epochs=cfg.training.warmup)

logging.info(f"Loading evaluator.")
eval = u_train.get_evaluator(cfg=cfg)


assert cfg.data.preprocess.max_spd_elements < 100, "max_spd_elements should be less than 100"

# ===================================================================================================================== #
# ========================================            starting training         ======================================= #
# ===================================================================================================================== #
logging.info(f"Starting Training.")

best_metrics = u_log.initialize_best_metrics(goal=goal)

pbar = tqdm(range(cfg.training.epochs))
for epoch in pbar:
    logging.info(f"Train loop.")
    # =========================== Training =========================== #
    if cfg.data.name == "ogbg-molhiv":
        loss_list, epoch_time_train = u_train.train_loop_ASAM(
            model=model, loader=dataloader["train"], critn=critn, optim=optim, epoch=epoch, device=device, task=task)
    else:
        loss_list, epoch_time_train = u_train.train_loop(
            model=model, loader=dataloader["train"], critn=critn, optim=optim, epoch=epoch, device=device, task=task)
    logging.info(f"Validation loop.")
    # =========================== Validation =========================== #
    if cfg.data.name == "Peptides-struc" or cfg.data.name == "Peptides-func":
        # Warning: checks val everu 10 epochs, or last 5 epochs
        validate_flag = (epoch % 10 == 0 or epoch > (cfg.training.epochs - 5))
        if validate_flag:
            val_metric, epoch_time_val = u_train.eval_loop_peptides(
                model=model, loaders=[dataloader], eval=eval, device=device, task=task, eval_type="val")
            test_metric, epoch_time_test = u_train.eval_loop_peptides(
                model=model, loaders=[dataloader], eval=eval, device=device, task=task, eval_type="test")
        else:
            val_metric = best_metrics["val_loss"]
            test_metric = best_metrics["test_loss"]
    else:
        val_metric, epoch_time_val = u_train.eval_loop(
            model=model, loader=dataloader["val"], eval=eval, device=device, task=task)
        test_metric, epoch_time_test = u_train.eval_loop(
            model=model, loader=dataloader["test"], eval=eval, device=device, task=task)

    best_metrics = u_log.update_best_metrics(
        best_metrics=best_metrics, val_metric=val_metric, test_metric=test_metric, epoch=epoch, goal=goal)

    u_log.log_wandb(epoch=epoch, optim=optim, loss_list=loss_list, val_metric=val_metric,
                    test_metric=test_metric, best_metrics=best_metrics, epoch_time_train=epoch_time_train, epoch_time_val=epoch_time_val, epoch_time_test=epoch_time_test)
    u_log.set_posfix(optim=optim, loss_list=loss_list, val_metric=val_metric,
                     test_metric=test_metric, pbar=pbar)
    u_train.sched_step(cfg=cfg, sched=sched, val_metric=val_metric)

wandb.finish()
