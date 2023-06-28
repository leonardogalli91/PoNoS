from haven import haven_utils as hu

suffixes = ["", "_NM", "_trueNM", "_zhangNM", "_epochNM"]
armijo_list = ["sgd" + suff + "_armijo" for suff in suffixes]
sls_ada_list = ["sls_ada" + suff for suff in suffixes]
sls_polyak_list = ["sls" + suff + "_polyak" for suff in suffixes] + ["polyak"]

ours_opt_list = armijo_list + sls_polyak_list + sls_ada_list

exp1 = [{"beta_b": 0.9, "name": "sgd_armijo", "reset_option": 11}, 
        {"c_step": 0.2, "name": "polyak", "max_eta": 10, "averaging_mode": 2000},
        {"beta_b": 0.5, "c": 0.5, "c_step": 0.1, "max_eta": 10, "name": "sls_zhangNM_polyak", "averaging_mode": 13}] # PoNoS

reset =[{"beta_b": 0.5, "c": 0.5, "c_step": 0.1, "name": "sls_zhangNM_polyak", "averaging_mode": 13},
        {"beta_b": 0.5, "c": 0.5, "c_step": 0.1, "max_eta": 10, "name": "sls_zhangNM_polyak", "averaging_mode": None},
        {"beta_b": 0.5, "c": 0.5, "c_p": 0.1, "name": "sgd_zhangNM_armijo", "reset_option": 3},
        {"beta_b": 0.5, "c": 0.5, "c_p": 0.1, "name": "sgd_zhangNM_armijo", "reset_option": 4},
        {"beta_b": 0.5, "c": 0.5, "name": "sgd_zhangNM_armijo", "reset_option": 11},
        {"beta_b": 0.5, "c": 0.5, "c_step": 0.1, "name": "sls_zhangNM_polyak", "sls_every": 2}]

trans = [{"name": "polyak", "c_step": 0.2, "max_eta": 10, "averaging_mode": 2000},
         {"name": "sgd_armijo", "beta_b": 0.9, "reset_option": 11},
         {"name": "sls_zhangNM_polyak", "beta_b": 0.5, "c": 0.5, "c_step": 0.1, "max_eta": 10, "averaging_mode": 13},
         {"name": "sls_ada", 'reset_option': 2000, "c_step": 0.2, "eta_max": 10, "suff_decr": "grad_norm"},
         {"name": "sls_ada", 'reset_option': 11,  "beta_b": 0.9, "suff_decr": "pp_norm", "eta_max": 10},
         {"name": "sls_ada_zhangNM", "suff_decr": "pp_norm", "c_step": 0.1, "reset_option": 200, "eta_max": 10}]

convex = [{"beta_b": 0.9, "name": "sgd_armijo", "reset_option": 11, "eta_max": 1e10},
          {"c_step": 0.2, "name": "polyak", "max_eta": 1e10, "averaging_mode": 2000},
          {"beta_b": 0.5, "c": 0.5, "c_step": 0.1, "max_eta": 1e10, "name": "sls_zhangNM_polyak", "averaging_mode": 13},
          {"beta_b": 0.5, "c": 0.5, "c_step": 0.1, "name": "sls_zhangNM_polyak"},
          {"beta_b": 0.5, "c": 0.1, "c_step": 0.1, "name": "sls_zhangNM_polyak"},
          {"beta_b": 0.5, "c": 0.1, "c_step": 0.1, "name": "sls_polyak"},
          {"beta_b": 0.5, "c": 0.5, "c_step": 0.1, "name": "sls_polyak"}]

study_on_c = [{"beta_b": 0.5, "c": 0.5, "c_step": 0.1, "name": "sls_zhangNM_polyak"},
              {"beta_b": 0.5, "c": 0.1, "c_step": 0.1, "name": "sls_zhangNM_polyak"},
              {"beta_b": 0.5, "c": 0.1, "c_step": 0.1, "name": "sls_polyak"},
              {"beta_b": 0.5, "c": 0.5, "c_step": 0.1, "name": "sls_polyak"}]

line_search = [{"beta_b": 0.5, "c": 0.5, "c_step": 0.1, "name": "sls_zhangNM_polyak", "averaging_mode": 13},
               {"NM_window": None, "beta_b": 0.5, "c": 0.5, "c_step": 0.1, "name": "sls_trueNM_polyak"},
               {"beta_b": 0.5, "c": 0.5, "c_step": 0.1, "name": "sls_epochNM_polyak"},
               {"beta_b": 0.5, "c": 0.5, "c_step": 0.1, "name": "sls_polyak"}]

long_run = 200
short_run = 75
many_runs = [0,1,2,3,4]
# Experiments definition
EXP_GROUPS = {
        "mnist_mlp":{"dataset":["mnist"],
            "model":["mlp"],
            "not_save_pth": True,
            "loss_func": ["softmax_loss"],
            "opt": exp1,
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[long_run],
            "runs":[0]},

    "cifar10_resnet":{"dataset":["cifar10"],
            "model":["resnet34"],
            "not_save_pth": True,
            "loss_func": ["softmax_loss"],
            "opt": exp1,
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[long_run],
            "runs":[0]},

    "cifar10_densenet":{"dataset":["cifar10"],
            "model":["densenet121"],
            "not_save_pth": True,
            "loss_func": ["softmax_loss"],
            "opt": exp1,
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[long_run],
            "runs":[0]},

        "cifar100_res":{"dataset":["cifar100"],
            "model":["resnet34_100"],
            "not_save_pth": True,
            "loss_func": ["softmax_loss"],
            "opt": exp1,
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[long_run],
            "runs":[0]},

        "cifar100_dense":{"dataset":["cifar100"],
            "model":["densenet121_100"],
            "not_save_pth": True,
            "loss_func": ["softmax_loss"],
            "opt": exp1,
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[long_run],
            "runs":[0]},

    "fashion_effb1": {"dataset": ["fashion"],
                       "model": ["efficientnet-b1"],
                       "not_save_pth": True,
                       "loss_func": ["softmax_loss"],
                       "opt": exp1,
                       "acc_func": ["softmax_accuracy"],
                       "batch_size": [128],
            "max_epoch":[long_run],
                       "runs":[0]},

        "svhn_wrn":{"dataset":["svhn"],
            "model":["wrn_10"],
            "not_save_pth": True,
            "loss_func": ["softmax_loss"],
            "opt": exp1,
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[long_run],
            "runs":[0]},

    "mushrooms": {"dataset": ["mushrooms"],
                  "model": ["logistic"],
                  "loss_func": ['logistic_loss'],
                  "acc_func": ["logistic_accuracy"],
                  "opt": convex,
                  "batch_size": [100],
                  "max_epoch": [35],
                  "runs": [0]},

    "ijcnn": {"dataset": ["ijcnn"],
               "model": ["logistic"],
               "loss_func": ['logistic_loss'],
               "acc_func": ["logistic_accuracy"],
               "opt": convex,
               "batch_size": [100],
               "max_epoch": [35],
               "runs": [0]},

    "rcv1": {"dataset": ['rcv1'],
                  "model": ["logistic"],
                  "loss_func": ['logistic_loss'],
                  "acc_func": ["logistic_accuracy"],
                  "opt": convex,
                  "batch_size": [100],
                  "max_epoch": [35],
                  "runs": [0]},

    "w8a": {"dataset": ['w8a'],
                  "model": ["logistic"],
                  "loss_func": ['logistic_loss'],
                  "acc_func": ["logistic_accuracy"],
                  "opt": convex,
                  "batch_size": [100],
                  "max_epoch": [35],
                  "runs": [0]},

    "trans_enc": {"dataset": ["wikitext2"],
                  "model": ["transformer_encoder"],
                  "not_save_pth": True,
                  "model_args": {"tgt_len": 35},
                  "loss_func": ["softmax_loss"],
                  "opt": trans,
                  "acc_func": ["ppl"],
                  "batch_size": [64],
                  "max_epoch": [100],
                  "runs": [0]},

    "trans_xl": {"dataset": ["ptb"],
                 "model": ["transformer_xl"],
                 "not_save_pth": True,
                 "model_args": {
                       "n_layer": 6,
                       "d_model": 512,
                       "n_head": 8,
                       "d_head": 64,
                       "d_inner": 2048,
                       "dropout": 0.1,
                       "dropatt": 0.0,
                       "tgt_len": 128,
                       "mem_len": 128,
                 },
                 "loss_func": ["softmax_loss"],
                  "opt": trans,
                 "acc_func": ["ppl"],
                 "batch_size": [64],
                 "max_epoch": [100],
                 "runs": [0]},

    #=========================================


            }

EXP_GROUPS = {k:hu.cartesian_exp_group(v) for k,v in EXP_GROUPS.items()}
