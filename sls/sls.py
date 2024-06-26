import copy
import time

from . import utils as ut
from .stoch_line_search import StochLineSearch


class Sls(StochLineSearch):
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        n_batches_per_epoch (int, recommended):: the number batches in an epoch
        init_step_size (float, optional): initial step size (default: 1)
        c (float, optional): armijo condition constant (default: 0.1)
        gamma (float, optional): factor used by Armijo for scaling the step-size at each line-search step (default: 2.0)
        reset_option (float, optional): sets the rest option strategy (default: 1)
        line_search_fn (float, optional): the condition used by the line-search to find the 
                    step-size (default: Armijo)
    """

    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 train_set_len=60000,
                 init_step_size=1,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0,
                 reset_option=1,
                 line_search_fn="armijo",
                 NM_window=10,
                 eta_max=10,
                 zhang_eta=1,
                 sls_every=1,
                 c_p=0.1,
                 debug_mode=False):
        params = list(params)
        super().__init__(params, n_batches_per_epoch=n_batches_per_epoch, train_set_len=train_set_len,
                         init_step_size=init_step_size, c=c, beta_b=beta_b, gamma=gamma, reset_option=reset_option,
                         line_search_fn=line_search_fn, NM_window=NM_window, zhang_eta=zhang_eta,
                         debug_mode=debug_mode)
        self.sls_every = sls_every
        self.eta_max = eta_max
        self.c_p = c_p


    def step(self, closure, closure_single):
        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()
        def closure_single_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure_single()

        # get loss and compute gradients
        if self.line_search_fn == "trueNM":
            loss, losses, indexes = closure_single_deterministic()
        else:
            loss = closure_deterministic()
            indexes = None
            losses = None
        loss.backward()

        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = ut.get_grad_list(self.params)
        grad_norm = ut.compute_grad_norm(grad_current)

        if ut.check_debug_mode_value(self.debug_mode, 0.5):
            self.state["all_sharp"].append(self.compute_sharp(self.params, params_current, grad_current, grad_norm, loss, closure_deterministic))

        if self.state["step"] % self.sls_every == 0:
            step_size = ut.reset_step(step_size=self.state.get('step_size') or self.init_step_size,
                                      n_batches_per_epoch=self.n_batches_per_epoch,
                                      gamma=self.gamma,
                                      reset_option=self.reset_option,
                                      init_step_size=self.init_step_size,
                                      max_value=self.eta_max, grad_norm=grad_norm, grad_norm_old=self.state.get("grad_norm"),
                                      suff_dec=self.state.get("sufficient_dec"), c_p=self.c_p)
            saved_step = ut.maybe_torch(step_size)
            step_size, loss_next = self.line_search(step_size, params_current, grad_current, loss, closure_deterministic, grad_norm, indexes)
        else:
            step_size, loss_next = self.step_with_no_line_search(self.state["step_size"], params_current, grad_current, loss)
            saved_step = ut.maybe_torch(step_size)

        self.save_state(step_size, loss, loss_next, indexes, losses, grad_norm, orig_step=saved_step)

        return loss
