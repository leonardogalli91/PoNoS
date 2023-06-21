import os
import argparse
import torch 
import numpy as np
import time
import pprint
import exp_configs
import wandb
from src import datasets, models, optimizers, metrics
from sls import utils as ut

from haven import haven_utils as hu
from haven import haven_chk as hc


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
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     train_flag=True,
                                     datadir=datadir,
                                     exp_dict=exp_dict,
                                     device=device)
    # Load Val Dataset
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   train_flag=False,
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   device=device)


    # Model
    # -----------
    model = models.get_model(exp_dict["model"], train_set=train_set)
    if exp_dict.get("multiple_gpu"):
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    if exp_dict.get("half"):
        model = model.half()
    elif exp_dict.get("double"):
        model = model.double()
    # Choose loss and metric function
    loss_function = metrics.get_metric_function(exp_dict["loss_func"])
    if "trueNM" in exp_dict["opt"]["name"]:
        loss_function_single = metrics.get_metric_function(exp_dict["loss_func"]+"_single")

    # Load Optimizer
    n_batches_per_epoch = len(train_set)/float(exp_dict["batch_size"])
    opt = optimizers.get_optimizer(opt=exp_dict["opt"],
                                   params=model.parameters(),
                                   n_batches_per_epoch =n_batches_per_epoch,
                                   train_set_len=len(train_set))

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
        print("epoch  train_loss  val_acc  grad_norm  forwards  backwards  backtracks   avg_step  spec_count  epoch_time")
    else:
        print("epoch  train_loss  val_acc  epoch_time")

    full_time = 0
    for epoch in range(s_epoch, exp_dict['max_epoch']):
        # Set seed
        np.random.seed(exp_dict['runs']+epoch)
        torch.manual_seed(exp_dict['runs']+epoch)
        torch.cuda.manual_seed_all(exp_dict['runs']+epoch)

        #Data Loader
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   drop_last=False,
                                                   shuffle=True,
                                                   batch_size=exp_dict["batch_size"], num_workers=6)#, pin_memory=True)
        iterator = train_loader.__iter__()

        score_dict = {"epoch": epoch}

        # Evaluation of the largest eigenvalue (extremely costly, use it only on small networks)
        if ut.check_debug_mode_value(exp_dict["opt"], 0.75): 
            score_dict["eig1"] = ut.get_hessian_eigenvalues(model, ut.take_first(train_set, 5000), neigs=1).item()
            extra_info = {**extra_info, **{"eig1": score_dict.get("eig1")}}

        if metrics_flag:
            # 1. Compute train loss over train set
            score_dict["train_loss"] = metrics.compute_metric_on_dataset(model, train_set, metric_name=exp_dict["loss_func"], device=device)

            # 2. Compute val acc over val set
            score_dict["val_acc"] = metrics.compute_metric_on_dataset(model, val_set, metric_name=exp_dict["acc_func"], device=device)

        # 3. Train over train loader
        model.train()
        s_time = time.time()
        for i in range(int(np.ceil(n_batches_per_epoch))):
            images, labels, indexes = next(iterator)
            images, labels = images.to(device), labels.to(device)
            if exp_dict.get("half"):
                images = images.half()
            elif exp_dict.get("double"):
                images = images.double()

            opt.zero_grad()

            if exp_dict["opt"]["name"] in exp_configs.ours_opt_list:
                closure = lambda: loss_function(model, images, labels, backwards=False)
                if "trueNM" in exp_dict["opt"]["name"]:
                    closure_single = lambda: loss_function_single(model, images, labels, indexes, backwards=False)
                else:
                    closure_single = lambda: exit(1)
                opt.step(closure, closure_single)
            else:
                loss = loss_function(model, images, labels)
                loss.backward()
                opt.step()
                if score_dict.get("all_losses") is None:
                    score_dict["all_losses"] = []
                score_dict["all_losses"].append(loss.item())
                if opt.param_groups[0].get("lr"):
                    score_dict["avg_step_size"] = (opt.param_groups[0]["lr"]/int(np.ceil(n_batches_per_epoch))) + (score_dict.get("avg_step_size") or 0)
                else:
                    score_dict["avg_step_size"] = 1

        e_time = time.time()

        # Record metrics
        to_be_recorded = ["step_size", "n_forwards", "n_backwards", "backtracks"]
        if exp_dict["opt"]["name"] in exp_configs.ours_opt_list:
            to_be_recorded = to_be_recorded + ["n_backtr", "special_count", "all_losses", "all_grad_norm", "all_lipschitz",
                                               "all_relative_dec", "all_orig_step", "all_sharp", "all_sharp", "all_step_size",
                                               "all_suff_dec", "all_dec"]
            if ut.check_debug_mode_value(exp_dict["opt"], 0.625):
                to_be_recorded = to_be_recorded + ["all_lip_smooth"]
            if ut.check_debug_mode_value(exp_dict["opt"], 1):
                to_be_recorded = to_be_recorded + ["zero_steps", "numerical_error"]
        for metric in to_be_recorded:
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
            minimal_dict = {"epoch": epoch, "train_loss": score_dict["train_loss"], 'val_acc': score_dict["val_acc"],
                            "train_epoch_time": score_dict["train_epoch_time"]}
            final_dict = {**minimal_dict, **extra_info}
            wandb.log(final_dict)

        score_list += [score_dict]

        if exp_dict["opt"]["name"] in exp_configs.ours_opt_list:
            print("{:5d}{:12.6f}{:9.4f}{:11.6f}{:10d}{:11d}{:12d}{:11.4f}{:12.3f}{:12.4f}".format(score_dict["epoch"], score_dict["train_loss"], score_dict["val_acc"], score_dict["grad_norm"], score_dict["n_forwards"], score_dict["n_backwards"], score_dict["backtracks"], score_dict["avg_step_size"], score_dict["special_count"], score_dict["train_epoch_time"]))
        else:
            print("{:5d}{:12.6f}{:9.4f}{:12.4f}".format(score_dict["epoch"], score_dict["train_loss"],
                                                        score_dict["val_acc"], score_dict["train_epoch_time"]))
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
