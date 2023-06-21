import sls
import exp_configs
import torch



def get_optimizer(opt, params, n_batches_per_epoch=None, train_set_len=None):
    """
    opt: name or dict
    params: model parameters
    n_batches_per_epoch: b/n
    """
    if isinstance(opt, dict):
        opt_name = opt["name"]
        opt_dict = opt
    else:
        opt_name = opt
        opt_dict = {}

    # ===============================================
    # our optimizers and their parameters: not all are reported here,
    # so if more are required, you need to add them in the function call
    # Note that these arguments are the same as the one in the file exp_configs.py
    n_batches_per_epoch = opt_dict.get("n_batches_per_epoch") or n_batches_per_epoch
    NM_window = opt_dict.get("NM_window") or -1
    if "trueNM" in opt_name:
        line_search = "trueNM"
    elif "zhangNM" in opt_name:
        line_search = "zhangNM_armijo"
    elif "epochNM" in opt_name:
        line_search = "NM_armijo"
        NM_window = int(n_batches_per_epoch)
    elif "NM" in opt_name:
        line_search = "NM_armijo"
    else:
        if opt_dict.get("line_search_fn") == "no":
            line_search = "no"
        else:
            line_search = "armijo"


    if opt_name in exp_configs.armijo_list:

        opt = sls.Sls(params,
                      c=opt_dict.get("c") or 0.1,
                      n_batches_per_epoch=n_batches_per_epoch,
                      train_set_len= train_set_len,
                      init_step_size=opt_dict.get("init_step_size") or 1,
                      reset_option=opt_dict.get("reset_option") or 1,
                      beta_b=opt_dict.get("beta_b") or 0.9,
                      NM_window=NM_window,
                      line_search_fn=line_search,
                      zhang_eta=opt_dict.get("zhang_eta") or 1,
                      sls_every=opt_dict.get("sls_every") or 1,
                      eta_max=opt_dict.get("eta_max") or 10,
                      c_p=opt_dict.get("c_p") or 0.1,
                      debug_mode=opt_dict.get("debug_mode") or False)

    elif opt_name in exp_configs.sls_polyak_list:
        if opt_name == "polyak":
            do_line_search = False
        else:
            do_line_search = True
        opt = sls.SlsPolyak(params,
                            n_batches_per_epoch=n_batches_per_epoch,
                            train_set_len=train_set_len,
                            c=opt_dict.get("c") or 0.1,
                            c_step=opt_dict.get("c_step") or 0.2,
                            beta_b=opt_dict.get("beta_b") or 0.9,
                            NM_window=NM_window,
                            line_search_fn=line_search,
                            max_eta=opt_dict.get("max_eta") or 10,
                            averaging_mode=opt_dict.get("averaging_mode") or 0,
                            f_star=opt_dict.get("f_star") or 0,
                            do_line_search=do_line_search,
                            zhang_eta=opt_dict.get("zhang_eta") or 1,
                            sls_every=opt_dict.get("sls_every") or 1,
                            debug_mode=opt_dict.get("debug_mode") or False)

    elif opt_name in exp_configs.sls_ada_list:
        opt = sls.AdaSLS(params,
                         c=opt_dict.get("c") or 0.1,
                         n_batches_per_epoch=n_batches_per_epoch,
                         train_set_len=train_set_len,
                         gv_option=opt_dict.get('gv_option', 'per_param'),
                         base_opt=opt_dict.get('base_opt') or "adam",
                         clip_grad=opt_dict.get("clip_grad"),
                         pp_norm_method=opt_dict.get('pp_norm_method') or "pp_armijo",
                         suff_decr=opt_dict.get('suff_decr') or "pp_norm",
                         momentum=opt_dict.get('momentum', 0.9),
                         beta=opt_dict.get('beta', 0.999),
                         gamma=opt_dict.get('gamma', 2),
                         init_step_size=opt_dict.get("init_step_size") or 1,
                         reset_option=opt_dict.get("reset_option") or 1,
                         beta_b=opt_dict.get("beta_b") or 0.9,
                         beta_f=opt_dict.get('beta_f', 2.),
                         eta_max=opt_dict.get("eta_max") or 10,
                         c_step=opt_dict.get("c_step") or 0.2,
                         f_star=opt_dict.get("f_star") or 0,
                         NM_window=NM_window,
                         averaging_mode=opt_dict.get("avg_mod") or False,
                         line_search_fn=line_search,
                         mom_type=opt_dict.get('mom_type', "standard"),
                         zhang_eta=opt_dict.get("zhang_eta") or 1,
                         debug_mode=opt_dict.get("debug_mode") or False)


    # ===============================================
    # others
    elif opt_name == "adam":
        opt = torch.optim.Adam(params, lr=opt_dict.get("lr") or 1e-3)

    elif opt_name == 'sgd':
        opt = torch.optim.SGD(params, lr=opt_dict.get("lr") or 1e-3)

    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt
