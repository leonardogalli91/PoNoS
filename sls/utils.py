import torch

import numpy as np
import contextlib


# single batch Gripppo nonmonotone line search
def check_trueNM_conditions(step_size, loss, loss_history, indexes, suff_dec,
                      loss_next, c, beta_b, stats=None):
    max_loss = 0
    for i in range(len(loss_history[indexes[0].item()])):
        intermediate_loss = 0
        for j in indexes:
            max_i = min(len(loss_history[j.item()])-1, i)
            intermediate_loss += loss_history[j.item()][max_i]
        if intermediate_loss >= max_loss:
            max_loss = intermediate_loss
    max_loss = torch.tensor(data=max(max_loss, loss.item()), device=loss.device)
    return check_armijo_conditions(step_size, max_loss, suff_dec, loss_next, c, beta_b, stats)


# cross batch Gripppo nonmonotone line search
def check_NM_armijo_conditions(step_size, loss, loss_history, suff_dec,
                      loss_next, c, beta_b, stats=None):
    max_loss = max(loss_history, default=0)
    max_loss = torch.tensor(data=max(max_loss, loss.item()), device=loss.device)
    return check_armijo_conditions(step_size, max_loss, suff_dec, loss_next, c, beta_b, stats)


# Armijo line search
def check_armijo_conditions(step_size, loss, suff_dec,
                      loss_next, c, beta_b, stats=None):
    found = 0
    sufficient_decrease = (step_size) * c * suff_dec
    rhs = loss - sufficient_decrease
    break_condition = loss_next - rhs
    if stats is not None:
        stats.append({"ineq_value":maybe_torch(break_condition), "step": step_size, "new_loss": maybe_torch(loss_next), "ref_loss": maybe_torch(loss), "rhs": maybe_torch(rhs),
                      "diff": maybe_torch(loss_next-loss), "suff_dec": sufficient_decrease})
    if (break_condition <= 0):
        found = 1
    else:
        step_size = step_size * beta_b

    return found, step_size


# function setting the initial step size of the new line search
def reset_step(step_size, n_batches_per_epoch=None, gamma=None, reset_option=1,
               init_step_size=None, max_value = 10, grad_norm=None, grad_norm_old=None,
               suff_dec=None, c_p=0.5):

    if reset_option == 0:
        pass

    elif reset_option == 1:
        step_size = step_size * gamma**(1. / n_batches_per_epoch)

    elif reset_option == 11:
        step_size = min(step_size * gamma**(1. / n_batches_per_epoch), max_value)

    elif reset_option == 2:
        step_size = init_step_size

    elif reset_option == 3:
        if grad_norm_old and grad_norm:
            step_size = step_size * (grad_norm_old**2) / (grad_norm**2)
        else:
            step_size = init_step_size

    elif reset_option == 4:
        if suff_dec:
            step_size = suff_dec / (c_p*grad_norm_old**2)
            step_size = min(1.01*step_size, max_value)
        else:
            step_size = init_step_size

    else:
        raise ValueError("reset_option {} does not exist".format(reset_option))

    return max(step_size, 1e-08)


def try_sgd_update(params, step_size, params_current, grad_current):
    zipped = zip(params, params_current, grad_current)

    for p_next, p_current, g_current in zipped:
        p_next.data = p_current - step_size * g_current


def compute_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def compute_grad_norm_list(grad_list):
    norm_list = []
    for g in grad_list:
        if g is None:
            continue
        norm_list.append(torch.sum(torch.mul(g, g)).item())
    return norm_list


def compute_inf_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        inf_norm = torch.max(torch.abs(g))
        if inf_norm > grad_norm:
            grad_norm = inf_norm
    return grad_norm


def compute_l1_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.abs(g))
    return grad_norm


def compute_last_norm(grad_list):
    grad_norm = 0.
    last_layer = len(grad_list)
    for i, g in enumerate(grad_list):
        if g is None:
            continue
        if i == (last_layer-1):
            grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def compute_dot_product(vect1_list, vect2_list):
    dot_product = 0
    for v1, v2 in zip(vect1_list, vect2_list):
        dot_product += torch.sum(torch.mul(v1, v2))
    return dot_product


def get_grad_list(params):
    return [p.grad for p in params]


def check_debug_mode_value(opt_dict, value=0.25):
    if isinstance(opt_dict, dict):
        return opt_dict.get("debug_mode") and opt_dict.get("debug_mode") >= value
    else:
        return opt_dict >= value

@contextlib.contextmanager
def random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

@contextlib.contextmanager
def random_seed_torch(seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(gpu_rng_state, device)


def maybe_torch(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_params(params):
    return sum(p.numel() for p in params if p.requires_grad)
