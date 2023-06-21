import copy
import time
import numpy as np
from . import utils as ut
from .stoch_line_search import StochLineSearch


class SlsPolyak(StochLineSearch):
    """
    Arguments:

    """

    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 train_set_len=60000,
                 c=0.1,
                 c_step=0.2,
                 beta_b=0.9,
                 line_search_fn="armijo",
                 NM_window=10,
                 max_eta=10,
                 averaging_mode=0,
                 f_star=0,
                 do_line_search=True,
                 zhang_eta=1,
                 sls_every=1,
                 debug_mode=False):
        params = list(params)
        super().__init__(params, n_batches_per_epoch=n_batches_per_epoch, train_set_len=train_set_len, c=c,
                         beta_b=beta_b, line_search_fn=line_search_fn, NM_window=NM_window,
                         zhang_eta=zhang_eta, debug_mode=debug_mode)
        self.max_eta = max_eta
        self.c_step = c_step
        self.do_line_search = do_line_search
        self.averaging_mode = averaging_mode
        self.f_star = f_star
        self.sls_every = sls_every


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
            polyak_step_size = loss / (self.c_step * grad_norm**2 + 1e-8)
            if self.averaging_mode == 13:
                step_size = polyak_step_size * (self.beta_b**self.lk)
            elif self.averaging_mode == 2000:
                coeff = self.gamma ** (1. / self.n_batches_per_epoch)
                step_size = min(polyak_step_size.item(), coeff * (self.state.get('step_size') or self.init_step_size))
            else:
                step_size = polyak_step_size

            eta_min = 1e-06
            step_size = max(min(step_size, self.max_eta), eta_min)

            if step_size == self.max_eta:
                self.state["special_count"] += (1/np.ceil(self.n_batches_per_epoch))
            saved_step = ut.maybe_torch(step_size)
            if self.do_line_search:
                step_size, loss_next = self.line_search(step_size, params_current, grad_current, loss, closure_deterministic, grad_norm, indexes)
            else:
                step_size, loss_next = self.step_with_no_line_search(step_size, params_current, grad_current, loss)
        else:
            step_size, loss_next = self.step_with_no_line_search(self.state["step_size"], params_current, grad_current, loss)
            saved_step = ut.maybe_torch(step_size)

        self.save_state(step_size, loss, loss_next, indexes, losses, grad_norm, orig_step=saved_step)

        return loss
