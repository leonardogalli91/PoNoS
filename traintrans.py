import os
import argparse
import torch 
import numpy as np
import time
import math
import pprint
import wandb
import exp_configs
from src import datasets, models, optimizers, metrics
from sls import utils as ut

from haven import haven_utils as hu
from haven import haven_chk as hc

from src.transformer_utils.transformer import evaluate, evaluate_transformer_xl


def transformer_encoder_loss(model, data, target, seq_len, device, backwards=False):
    src_mask = model.generate_square_subsequent_mask(seq_len).to(device)
    output = model(data, src_mask)
    output_flat = output.view(-1, model.ntoken)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(output_flat, target.view(-1))
    if backwards and loss.requires_grad:
        loss.backward()
    return loss

def transformer_encoder_loss_single(model, data, target, indexes, seq_len, device):
    src_mask = model.generate_square_subsequent_mask(seq_len).to(device)
    output = model(data, src_mask)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    output_t = torch.transpose(output, 0, 1)
    loss = 0
    losses = []
    for log, lab in zip(output_t, target.T):
        single_loss = criterion(log, lab)/data.shape[1]
        loss += single_loss
        losses.append(single_loss)
    return loss, losses, indexes

def transformer_xl_loss(model, data, target, backwards=False):
    mems = tuple()
    ret = model(data, target, *mems)
    loss, mems = ret[0], ret[1:]
    final_loss = loss.float().mean().type_as(loss)
    if backwards and final_loss.requires_grad:
        final_loss.backward()
    return final_loss

def transformer_xl_loss_single(model, data, target, indexes):
    mems = tuple()
    ret = model(data, target, *mems)
    loss, mems = ret[0], ret[1:]
    new_loss = loss.T
    losses = [trans_loss.sum()/(data.shape[1]*data.shape[0]) for trans_loss in new_loss]
    deleteme = 0
    for l in losses:
        deleteme += l.item()
#    print(loss.float().mean().item(), deleteme)
    return loss.float().mean().type_as(loss), losses, indexes

def trainval(exp_dict, savedir_base, datadir, reset=False, metrics_flag=True):
    # bookkeeping
    # ---------------
    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    # possibly set a logging on weights and bias (wandb)
    if ut.check_debug_mode_value(exp_dict["opt"]) and exp_dict["runs"] != 0:
        exp_dict["opt"]["debug_mode"] = 0
    if ut.check_debug_mode_value(exp_dict["opt"]):
        wandb.init(project="step_" + exp_dict["dataset"] + "_" + exp_dict["model"], config=exp_dict)
        wandb.run.name = exp_id
        wandb.run.save()
    extra_info = {}

    # delete and backup experiment
    if reset:
        hc.delete_experiment(savedir, backup_flag=True)
    
    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)

    # set seed
    seed = 42 + exp_dict['runs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Dataset
    # -----------
    # Load Train Dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if exp_dict.get("multiple_gpu"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    (train_set, val_set, *optional_transformer_len, ) = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                                                             train_flag=True,
                                                                             datadir=datadir,
                                                                             exp_dict=exp_dict,
                                                                             device=device)


    # Model
    # -----------
    if len(optional_transformer_len) > 0:
        model = models.get_model(exp_dict["model"], train_set=train_set, model_args=exp_dict.get("model_args"), features_dim=optional_transformer_len[0])
    else:
        if exp_dict["model"] in ["transformer_xl", "transformer_enc"]:
            features_dim = next(iter(train_set))[0].shape[1]
        else:
            features_dim = 0
        model = models.get_model(exp_dict["model"], train_set=train_set, model_args=exp_dict.get("model_args"), features_dim=features_dim)
    if exp_dict.get("multiple_gpu"):
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    if exp_dict.get("half"):
        model = model.half()
    elif exp_dict.get("double"):
        model = model.double()

    # Load Optimizer
    if exp_dict.get("shuffle"):
        n_batches_per_epoch = 60000
    else:
        n_batches_per_epoch = len(train_set)
    opt = optimizers.get_optimizer(opt=exp_dict["opt"],
                                   params=model.parameters(),
                                   n_batches_per_epoch =n_batches_per_epoch,
                                   train_set_len=int(n_batches_per_epoch*float(exp_dict["batch_size"])))

    # Checkpoint
    # -----------
    model_path = os.path.join(savedir, 'model.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')
    opt_path = os.path.join(savedir, 'opt_state_dict.pth')

    if os.path.exists(score_list_path):
        # resume experiment
        score_list = hu.load_pkl(score_list_path)
        model.load_state_dict(torch.load(model_path))
        opt.load_state_dict(torch.load(opt_path))
        s_epoch = score_list[-1]['epoch'] + 1
        if exp_dict["opt"]["name"] in exp_configs.ours_opt_list:
            opt.new_epoch()
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ------------
    print('Starting experiment at epoch %d/%d' % (s_epoch, exp_dict['max_epoch']))
    if exp_dict["opt"]["name"] in exp_configs.ours_opt_list:
        # avg_sharp {:11.3e}
        print("epoch  train_loss  train_metric  val_metric  grad_norm  forwards  backwards  backtracks   avg_step  spec_count  epoch_time")
    else:
        print("epoch  train_loss  train_metric  val_metric  epoch_time")

    full_time = 0
    for epoch in range(s_epoch, exp_dict['max_epoch']):
        # Set seed
        np.random.seed(exp_dict['runs']+epoch)
        torch.manual_seed(exp_dict['runs']+epoch)
        torch.cuda.manual_seed_all(exp_dict['runs']+epoch)


        score_dict = {"epoch": epoch}

        if metrics_flag:
            # 1. Compute train loss over train set

            model.eval()
            if exp_dict["acc_func"] == "ppl":
                try:
                    if exp_dict["model"] == "transformer_encoder":
                        train_ppl_loss, loss = evaluate(model, train_set, device, False, 1, no_loss=False)
                        ppl_loss, _ = evaluate(model, val_set, device, False, 1, no_loss=True)
                    elif exp_dict["model"] == "transformer_xl":
                        train_ppl_loss, loss = evaluate_transformer_xl(model, train_set, device, False, 1, no_loss=False)
                        ppl_loss, _ = evaluate_transformer_xl(model, val_set, device, False, 1, no_loss=True)
                    score_dict["train_loss"] = loss
                    score_dict["train_metric"] = math.exp(train_ppl_loss)
                    score_dict["val_metric"] = math.exp(ppl_loss)
                except OverflowError:
                    score_dict["val_metric"] = float("inf")
                    score_dict["train_metric"] = float("inf")
                    score_dict["val_metric"] = float("inf")
            else:
                # 2. Compute val acc over val set
                score_dict["val_metric"] = metrics.compute_metric_on_dataset(model, val_set,
                                                                             metric_name=exp_dict["acc_func"])

        # 3. Train over train loader
        model.train()
        s_time = time.time()
        for i, (images, labels, *seq_len) in enumerate(train_set):
            indexes = np.arange(i*exp_dict["batch_size"], (i+1)*exp_dict["batch_size"])
            images, labels = images.to(device), labels.to(device)
            if exp_dict.get("half"):
                images = images.half()
            elif exp_dict.get("double"):
                images = images.double()

            opt.zero_grad()

            if exp_dict["opt"]["name"] in exp_configs.ours_opt_list:
                if exp_dict["model"] == "transformer_encoder":
                    closure = lambda: transformer_encoder_loss(model, images, labels, seq_len[0], device)
                    closure_single = lambda: transformer_encoder_loss_single(model, images, labels, indexes, seq_len[0], device)
                elif exp_dict["model"] == "transformer_xl":
                    closure = lambda: transformer_xl_loss(model, images, labels)
                    closure_single = lambda: transformer_xl_loss_single(model, images, labels, indexes)
                else:
                    raise ValueError("model %s does not exist..." % exp_dict["model"])
                opt.step(closure, closure_single)
            else:
                if exp_dict["model"] == "transformer_encoder":
                    loss = transformer_encoder_loss(model, images, labels, seq_len[0], device)
                elif exp_dict["model"] == "transformer_xl":
                    loss = transformer_xl_loss(model, images, labels)
                loss.backward()
                score_dict["grad_norm"] = ut.compute_grad_norm(ut.get_grad_list(model.parameters())).item()
                score_dict["avg_step_size"] = exp_dict["opt"]["lr"]
                opt.step()

        e_time = time.time()

        # Record metrics
        to_be_recorded = ["step_size", "n_forwards", "n_backwards", "backtracks"]
        if exp_dict["opt"]["name"] in exp_configs.ours_opt_list:
            to_be_recorded = to_be_recorded + ["n_backtr", "special_count", "all_grad_norm", "all_lipschitz", "d_norm",
                                               "all_relative_dec", "all_orig_step", "all_sharp", "all_sharp", "all_step_size",
                                               "all_suff_dec", "all_dec"]
            if ut.check_debug_mode_value(exp_dict["opt"], 0.625):
                to_be_recorded = to_be_recorded + ["all_lip_smooth"]
            if ut.check_debug_mode_value(exp_dict["opt"], 1):
                to_be_recorded = to_be_recorded + ["all_losses", "all_orig_step", "zero_steps",
                                                   "numerical_error", "all_sharp", "all_grad_norm"]
        for metric in to_be_recorded:
            if opt.state.get(metric) or opt.state[metric] == 0:
                score_dict[metric] = opt.state[metric]
                if "all" in metric:
                    new_metric_name = metric.split("all_")[1]
                    if metric == "all_step_size":
                        new_metric_name = "avg_step_size"
                    score_dict[new_metric_name] = sum(opt.state[metric])/max(len(opt.state[metric]), 1)
        score_dict["batch_size"] = exp_dict["batch_size"]
        score_dict["train_epoch_time"] = e_time - s_time
        full_time += (e_time - s_time)
        score_dict["time"] = full_time
        if exp_dict["opt"]["name"] in exp_configs.ours_opt_list:
            opt.new_epoch()

        if ut.check_debug_mode_value(exp_dict["opt"]):
            minimal_dict = {"epoch": epoch, "train_loss": score_dict["train_loss"], 'train_metric': score_dict["train_metric"], 'val_metric': score_dict["val_metric"],
                            "train_epoch_time": score_dict["train_epoch_time"]}
            final_dict = {**minimal_dict, **extra_info}
            wandb.log(final_dict)

        score_list += [score_dict]

        if exp_dict["opt"]["name"] in exp_configs.ours_opt_list:
            print("{:5d}{:12.6f}{:14.4f}{:12.4f}{:11.6f}{:10d}{:11d}{:12d}{:11.4f}{:12.3f}{:12.4f}".format(score_dict["epoch"], score_dict["train_loss"], score_dict["train_metric"], score_dict["val_metric"], score_dict["grad_norm"], score_dict["n_forwards"], score_dict["n_backwards"], score_dict["backtracks"], score_dict["avg_step_size"], score_dict["special_count"], score_dict["train_epoch_time"]))
        else:
            print("{:5d}{:12.6f}{:14.4f}{:12.4f}{:12.4f}".format(score_dict["epoch"], score_dict["train_loss"], score_dict["train_metric"], score_dict["val_metric"], score_dict["train_epoch_time"]))
        hu.save_pkl(score_list_path, score_list)
        if not exp_dict.get("not_save_pth"):
            hu.torch_save(model_path, model.state_dict())
            hu.torch_save(opt_path, opt.state_dict())
        if score_dict["train_loss"] < 1e-06:
            print('Very Small Loss')
            break

    print("Saved in: %s" % savedir)
    print('Experiment completed')
    if ut.check_debug_mode_value(exp_dict["opt"]):
        wandb.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))
        exp_list = [exp_dict]
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]


    # Run experiments
    # ----------------------------
    for exp_dict in exp_list:
        # do trainval
        trainval(exp_dict=exp_dict,
                savedir_base=args.savedir_base,
                datadir=args.datadir,
                reset=args.reset)
