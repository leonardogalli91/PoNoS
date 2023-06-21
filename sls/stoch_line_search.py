import torch
import copy
import collections
import wandb

from . import utils as ut

class StochLineSearch(torch.optim.Optimizer):
    def __init__(self, params,
                 n_batches_per_epoch=500,
                 train_set_len=60000,
                 init_step_size=1,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0,
                 reset_option=0,
                 line_search_fn="armijo",
                 NM_window=-1,
                 nm_type="min",
                 zhang_eta=1,
                 rho=0.05,
                 debug_mode=False):
        params = list(params)
        super().__init__(params, {})

        self.params = params
        self.c = c
        self.beta_b = beta_b
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.n_batches_per_epoch = n_batches_per_epoch
        self.line_search_fn = line_search_fn
        self.NM_window = NM_window
        self.lk = 0
        self.state['step'] = 0
        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.debug_mode = debug_mode
        self.nm_type = nm_type
        self.zhang_eta = zhang_eta
        self.rho = rho

        self.reset_option = reset_option

        if line_search_fn == "trueNM":
            self.state['loss_history'] = [None for i in range(train_set_len)]
            for i in range(train_set_len):
                self.state['loss_history'][i] = collections.deque(maxlen=NM_window)
        else:
            self.state["Q_k"] = 0
            self.state["C_k"] = 0
            if NM_window >= 0:
                self.state['loss_history'] = collections.deque(maxlen=NM_window)
            else:
                self.state['loss_history'] = []
        self.new_epoch()


    def step(self, closure, closure_single):
        # deterministic closure
        raise RuntimeError("This function should not be called")

    def line_search(self, step_size, params_current, grad_current, loss, closure_deterministic, grad_norm, indexes, non_parab_dec=None, precond=False):
        with torch.no_grad():
            stats = None
            if ut.check_debug_mode_value(self.debug_mode, 1):
                stats = []

            if self.line_search_fn == "zhangNM_armijo":
                if self.NM_window < 0:
                    q_kplus1 = self.zhang_eta*self.state["Q_k"] + 1
                    self.state["C_k"] = (self.zhang_eta*self.state["Q_k"]*self.state["C_k"] + loss.item())/q_kplus1
                    self.state["Q_k"] = q_kplus1
                elif self.NM_window > 0:
                    if self.state["step"] >= self.NM_window:
                        self.state["zhang_sigma"] = self.zhang_eta**self.NM_window
                    q_kplus1 = self.zhang_eta*self.state["Q_k"] + 1 - (self.state.get("zhang_sigma") or 0)
                    if self.state.get("zhang_sigma"):
                        self.state["C_k"] = (self.zhang_eta*self.state["Q_k"]*self.state["C_k"] + loss.item() - self.state.get("zhang_sigma")*self.state['loss_history'].popleft())/q_kplus1
                    else:
                        self.state["C_k"] = (self.zhang_eta*self.state["Q_k"]*self.state["C_k"] + loss.item())/q_kplus1
                    self.state["Q_k"] = q_kplus1

            grad_norm = ut.maybe_torch(grad_norm)
            if grad_norm >= 1e-8 and (loss.item() != 0 or self.line_search_fn != "armijo"):
                # check if condition is satisfied
                found = 0

                if non_parab_dec is not None:
                    suff_dec = non_parab_dec
                else:
                    suff_dec = grad_norm**2

                for e in range(100):
                    # try a prospective step
                    if precond:
                        self.try_sgd_precond_update(self.params, step_size, params_current, grad_current, momentum=self.momentum2)
                    else:
                        ut.try_sgd_update(self.params, step_size, params_current, grad_current)

                    # compute the loss at the next step; no need to compute gradients.
                    loss_next = closure_deterministic()
                    self.state['n_forwards'] += 1

                    if self.line_search_fn == "armijo" or (
                            self.line_search_fn == "trueNM" and self.state['step'] <= int(self.n_batches_per_epoch)):
                        found, step_size = ut.check_armijo_conditions(step_size=step_size,
                                                                      loss=loss,
                                                                      suff_dec=suff_dec,
                                                                      loss_next=loss_next,
                                                                      c=self.c,
                                                                      beta_b=self.beta_b,
                                                                      stats=stats)

                    elif self.line_search_fn == "NM_armijo":
                        found, step_size = ut.check_NM_armijo_conditions(step_size=step_size,
                                                                         loss=loss,
                                                                         loss_history=self.state['loss_history'],
                                                                         suff_dec=suff_dec,
                                                                         loss_next=loss_next,
                                                                         c=self.c,
                                                                         beta_b=self.beta_b,
                                                                         stats=stats)

                    elif self.line_search_fn == "zhangNM_armijo":
                        ref_value = max(self.state["C_k"], loss.item())
                        found, step_size = ut.check_armijo_conditions(step_size=step_size,
                                                                      loss=ref_value,
                                                                      suff_dec=suff_dec,
                                                                      loss_next=loss_next,
                                                                      c=self.c,
                                                                      beta_b=self.beta_b,
                                                                      stats=stats)

                    elif self.line_search_fn == "trueNM" and self.state['step'] > int(self.n_batches_per_epoch):
                        found, step_size = ut.check_trueNM_conditions(step_size=step_size,
                                                                      loss=loss,
                                                                      loss_history=self.state['loss_history'],
                                                                      indexes=indexes,
                                                                      suff_dec=suff_dec,
                                                                      loss_next=loss_next,
                                                                      c=self.c,
                                                                      beta_b=self.beta_b,
                                                                      stats=stats)
                    if ut.check_debug_mode_value(self.debug_mode, 1):
                        norm = ut.compute_grad_norm(self.params)
                        stats[-1]["w_norm"] = norm
                    if found == 1:
                        break
                   
                # if line search exceeds 100 internal iterations
                if found == 0:
                    step_size = torch.tensor(data=1e-6)
                    ut.try_sgd_update(self.params, 1e-6, params_current, grad_current)

                self.state['backtracks'] += e
                self.state['n_backtr'].append(e)
                self.lk = max(self.lk + e -1, 0)

                if ut.check_debug_mode_value(self.debug_mode, 1):
                    if stats[-1]["new_loss"] == stats[-1]["ref_loss"]:
                        self.state['numerical_error'] += 1
                    norm = ut.compute_grad_norm(self.params)
                    print(
                        "\tIter {:4d}  loss {:15.9e}  g_norm {:14.9e}  step {:7.5f}  backtracks {:2d}  L_approx {:14.9e}  w_norm {:14.8e}".format(
                            self.state['step'], loss.item(), grad_norm, stats[-1]["step"], e, self.state.get("L_approx") or 0., norm))
                    if ut.check_debug_mode_value(self.debug_mode, 2):
                        print("\t\t  step_size       new_loss            rhs       ref_loss           diff       suff_dec     ineq_value           w_norm")
                        for stat in stats:
                            print("\t\t{:11.4f}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:15.7e}{:17.12f}".format(stat["step"], stat["new_loss"], stat["rhs"], stat["ref_loss"], stat["diff"], stat["suff_dec"], stat["ineq_value"], stat["w_norm"]))

            else:
                print("Grad norm is {} and loss is {}".format(grad_norm, loss.item()))
                if loss.item() == 0:
                    self.state['numerical_error'] += 1
                if grad_norm == 0:
                    self.state["zero_steps"] += 1
                step_size = 0
                loss_next = closure_deterministic()

        return step_size, loss_next

    def step_with_no_line_search(self, step_size, params_current, grad_current, loss, precond=False):
        self.state['n_backtr'].append(0)
        if precond:
            self.try_sgd_precond_update(self.params, step_size, params_current, grad_current, momentum=self.momentum2)
        else:
            ut.try_sgd_update(self.params, step_size, params_current, grad_current)
        if self.line_search_fn == "zhangNM_armijo":
            q_kplus1 = self.zhang_eta*self.state["Q_k"] + 1
            self.state["C_k"] = (self.zhang_eta*self.state["Q_k"]*self.state["C_k"] + loss.item())/q_kplus1
            self.state["Q_k"] = q_kplus1
        return step_size, loss-1  # this should be loss_next, but in this case it is not used

    def save_state(self, step_size, loss, loss_next, indexes, losses, grad_norm, orig_step=1, extra_info={}):
        step_size = ut.maybe_torch(step_size)
        orig_step = ut.maybe_torch(orig_step)
        grad_norm = ut.maybe_torch(grad_norm)
        dec = max(loss.item()-loss_next.item(), 0)
        self.state['step'] += 1
        self.state['step_size'] = step_size
        self.state['grad_norm'] = grad_norm
        self.state['loss'] = loss.item()
        self.state['new_loss'] = loss_next.item()
        self.state['all_step_size'].append(step_size)
        self.state['all_losses'].append(loss.item())
        self.state['dec'] = dec
        self.state['all_dec'].append(dec)
        self.state["all_grad_norm"].append(grad_norm)
        self.state['all_orig_step'].append(orig_step)
        self.state["sufficient_dec"] = dec/max(grad_norm**2, 1e-08)
        self.state["all_suff_dec"].append(dec/max(grad_norm**2, 1e-08))
        self.state["all_relative_dec"].append(dec/max(grad_norm, 1e-08))
        self.state["all_lipschitz"].append(dec/max((step_size*grad_norm), 1e-08))

        if ut.check_debug_mode_value(self.debug_mode):
            if ut.check_debug_mode_value(self.debug_mode, 0.5):
                extra_info = {**extra_info, **{"sharpness": self.state["all_sharp"][-1]}}
            if ut.check_debug_mode_value(self.debug_mode, 0.625):
                extra_info = {**extra_info, **{'lip_smooth': self.state['all_lip_smooth'][-1]}}
            if ut.check_debug_mode_value(self.debug_mode, 0.7):
                extra_info = {**extra_info, **{'sgc': self.state['sgc'][-1]}}
            minimal_dict = {"step": self.state['step'], 'step_size': step_size, 'loss': loss.item(), 'loss_decr': loss.item() - loss_next.item(),
                            'grad_norm': grad_norm, 'orig_step': orig_step, 'backtracks': self.state['n_backtr'][-1], 'n_forward': self.state['n_forwards'],
                            "n_batches_per_epoch": self.n_batches_per_epoch, 'lipschitz': self.state["all_lipschitz"][-1], 'relative_dec': self.state["all_relative_dec"][-1],
                            "n_params": ut.count_params(self.params)}
            final_dict = {**minimal_dict, **extra_info}
            wandb.log(final_dict)
        if self.line_search_fn == "trueNM":
            for i, l in zip(indexes, losses):
                self.state['loss_history'][i.item()].append(l.item())
        elif self.NM_window > 0:
            self.state['loss_history'].append(loss.item())

    def new_epoch(self):
        self.state['all_step_size'] = []
        self.state['all_losses'] = []
        self.state["all_relative_dec"] = []
        self.state["all_lipschitz"] = []
        self.state["all_lip_smooth"] = []
        self.state['all_dec'] = []
        self.state["all_suff_dec"] = []
        self.state["sgc"] = []
        self.state["all_grad_norm"] = []
        self.state['all_orig_step'] = []
        self.state['all_sharp'] = []
        self.state['backtracks'] = 0
        self.state['n_backtr'] = []
        self.state['zero_steps'] = 0
        self.state['numerical_error'] = 0
        self.state['special_count'] = 0

    def gather_flat_grad(self, params):
        views = []
        for p in params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def compute_sharp(self, params, params_current, grad_current, grad_norm, loss, closure_deterministic):
        scale = self.rho / (grad_norm + 1e-12)
        zipped = zip(params, params_current, grad_current)

        for p_next, p_current, g_current in zipped:
            if g_current is None:
                continue
            e_w = g_current * p_next * scale
            p_next.data = p_current + e_w  # climb to the local maximum "w + e(w)"
        return closure_deterministic().item() - loss.item()

    def flatten_vect(self, vect):
        views = []
        for p in vect:
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)
