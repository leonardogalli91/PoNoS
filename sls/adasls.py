import copy
import time
import torch
import numpy as np

from . import utils as ut
from .stoch_line_search import StochLineSearch


class AdaSLS(StochLineSearch):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 train_set_len=60000,
                 init_step_size=1,
                 c=0.1,
                 gamma=2.0,
                 beta=0.999,
                 momentum=0.9,
                 gv_option='per_param',
                 base_opt='adam',
                 pp_norm_method='pp_armijo',
                 clip_grad=False,
                 # Polyak stuff
                 eta_max=None,
                 c_step=0.2,
                 f_star=0,
                 mom_type='standard',
                 # sls stuff
                 beta_b=0.9,
                 beta_f=2.0,
                 reset_option=1,
                 line_search_fn="armijo",
                 suff_decr="pp_norm",
                 NM_window=10,
                 averaging_mode=False,
		         # Zhang stuff
                 zhang_eta=1,
                 debug_mode=False):
        params = list(params)
        super().__init__(params, n_batches_per_epoch=n_batches_per_epoch, train_set_len=train_set_len,
                         init_step_size=init_step_size, c=c, beta_b=beta_b, gamma=gamma, reset_option=reset_option,
                         line_search_fn=line_search_fn, NM_window=NM_window, zhang_eta=zhang_eta,
                         debug_mode=debug_mode)
        self.mom_type = mom_type
        self.pp_norm_method = pp_norm_method
        self.beta_f = beta_f
        self.beta_b = beta_b
        self.reset_option = reset_option
        self.averaging_mode = averaging_mode
        self.line_search_fn = line_search_fn
        self.suff_decr = suff_decr
        self.params = params
        if self.mom_type == 'heavy_ball':
            self.params_prev = copy.deepcopy(params)
        self.eta_max = eta_max
        self.c_step = c_step
        self.f_star = f_star
        self.momentum = momentum
        self.beta = beta
        self.clip_grad = clip_grad
        self.gv_option = gv_option
        self.base_opt = base_opt
        if self.gv_option in ['scalar']:
            self.state['gv'] = 0.

        elif self.gv_option == 'per_param':
            self.state['gv'] = [torch.zeros(p.shape).to(p.device) for p in params]

            if self.base_opt in ['amsgrad', 'adam']:
                self.state['mv'] = [torch.zeros(p.shape).to(p.device) for p in params]
            
            if self.base_opt == 'amsgrad':
                self.state['gv_max'] = [torch.zeros(p.shape).to(p.device) for p in params]

    def step(self, closure, closure_single):
        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()
        def closure_single_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure_single()

        if self.line_search_fn == "trueNM":
            loss, losses, indexes = closure_single_deterministic()
        else:
            loss = closure_deterministic()
            indexes = None
            losses = None
        loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.params, 0.25)
        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1        
        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = ut.get_grad_list(self.params)
        grad_norm = ut.compute_grad_norm(grad_current)

        if ut.check_debug_mode_value(self.debug_mode, 0.5):
            self.state["all_sharp"].append(self.compute_sharp(self.params, params_current, grad_current, grad_norm, loss, closure_deterministic))
        #  Gv options
        # =============
        if self.gv_option in ['scalar']:
            # update gv
            self.state['gv'] += (grad_norm.item())**2

        elif self.gv_option == 'per_param':
            # update gv
            for i, g in enumerate(grad_current):
                if self.base_opt == 'adagrad':
                    self.state['gv'][i] += g**2 

                elif self.base_opt == 'rmsprop':
                    self.state['gv'][i] = (1-self.beta)*(g**2) + (self.beta) * self.state['gv'][i]

                elif self.base_opt in ['amsgrad', 'adam']:
                    self.state['gv'][i] = (1-self.beta)*(g**2) + (self.beta) * self.state['gv'][i]
                    self.state['mv'][i] = (1-self.momentum)*g + (self.momentum) * self.state['mv'][i]

                else:
                    raise ValueError('%s does not exist' % self.base_opt)

        pp_norm = self.get_pp_norm(grad_current=grad_current, pp_norm_method=self.pp_norm_method)
        if self.suff_decr != "grad_norm":
            suff_dec = self.get_pp_norm(grad_current=grad_current, pp_norm_method=self.suff_decr)

        numerator = loss - self.f_star
        if numerator < 0:
            numerator = 1e-06
        if self.reset_option == 200:
            polyak_step_size = numerator / (self.c_step * pp_norm + 1e-8)
            step_size = min(polyak_step_size, self.eta_max)
        elif self.reset_option == 213:
            polyak_step_size = (numerator / (self.c_step * pp_norm + 1e-8)) * (self.beta_b**self.lk)
            step_size = min(polyak_step_size, self.eta_max)
        elif self.reset_option == 2000:
            polyak_step_size = numerator / (self.c_step * pp_norm + 1e-8)
            coeff = self.gamma ** (1. / self.n_batches_per_epoch)
            step_size = min(polyak_step_size.item(), coeff * (self.state.get('step_size') or self.init_step_size))
        elif self.reset_option == 20:
            step_size = self.init_step_size
        else:
            step_size = ut.reset_step(step_size=self.state.get('step_size') or self.init_step_size,
                                      n_batches_per_epoch=self.n_batches_per_epoch,
                                      gamma=self.gamma,
                                      reset_option=self.reset_option,
                                      init_step_size=self.init_step_size)
        # compute step size
        # =================
        saved_step = ut.maybe_torch(step_size)
        if self.reset_option not in [2000, 20]:
            if self.suff_decr == "grad_norm":
                step_size, loss_next = self.line_search(step_size, params_current, grad_current, loss, closure_deterministic, grad_norm, indexes, precond=True)
            else:
                step_size, loss_next = self.line_search(step_size, params_current, grad_current, loss, closure_deterministic, grad_norm, indexes, non_parab_dec=suff_dec, precond=True)
        else:
            step_size, loss_next = self.step_with_no_line_search(step_size, params_current, grad_current, loss, precond=True)
        # save the new step-size
        wandb_also = {"pp_norm": pp_norm.item()}
        self.state["d_norm"] = pp_norm.item()
        self.save_state(step_size, loss, loss_next, indexes, losses, grad_norm, orig_step=saved_step, extra_info=wandb_also)

        # compute gv stats
        gv_max = 0.    
        gv_min = np.inf 
        gv_sum =  0
        gv_count = 0   

        for i, gv in enumerate(self.state['gv']):
            gv_max = max(gv_max, gv.max().item())    
            gv_min = min(gv_min, gv.min().item())    
            gv_sum += gv.sum().item()
            gv_count += len(gv.view(-1))   
    
        self.state['gv_stats'] = {'gv_max':gv_max, 'gv_min':gv_min, 'gv_mean': gv_sum/gv_count}  

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('nans detected')

        return loss

    def get_pp_norm(self, grad_current, pp_norm_method):
        if pp_norm_method in ['pp_armijo', "adam", "dir_der"]:
            pp_norm = 0
            for i, (g_i, gv_i, mv_i) in enumerate(zip(grad_current, self.state['gv'], self.state['mv'])):
                if self.base_opt in ['diag_hessian', 'diag_ggn_ex', 'diag_ggn_mc']:
                    pv_i = 1. / (gv_i+ 1e-8) # computing 1 / diagonal for using in the preconditioner

                elif self.base_opt == 'adam':
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
                    pv_i = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)
                    if self.momentum == 0. or self.mom_type == 'heavy_ball':
                        mv_i_scaled = g_i
                    else:
                        mv_i_scaled = scale_vector(mv_i, self.momentum, self.state['step'] + 1)

                elif self.base_opt == 'amsgrad':
                    self.state['gv_max'][i] = torch.max(gv_i, self.state['gv_max'][i])
                    gv_i_scaled = scale_vector(self.state['gv_max'][i], self.beta, self.state['step']+1)

                    pv_i = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                elif self.base_opt in ['adagrad', 'rmsprop']:
                    pv_i = 1./(torch.sqrt(gv_i) + 1e-8)
                else:
                    raise ValueError('%s not found' % self.base_opt)

                if pp_norm_method == 'pp_armijo':
                    layer_norm = ((g_i**2) * pv_i).sum()
                elif pp_norm_method in "adam":
                    layer_norm = ((pv_i * mv_i_scaled)**2).sum()
                elif pp_norm_method == "dir_der":
                    if self.base_opt == 'adam':
                        layer_norm = (g_i * (pv_i * mv_i_scaled)).sum()
                    elif self.base_opt == 'amsgrad':
                        layer_norm = (g_i * (pv_i * g_i)).sum()

                pp_norm += layer_norm

        elif pp_norm_method in ['pp_lipschitz']:
            pp_norm = 0

            for g_i in grad_current:
                if isinstance(g_i, float) and g_i == 0:
                    continue
                pp_norm += (g_i * (g_i + 1e-8)).sum()

        else:
            raise ValueError('%s does not exist' % pp_norm_method)

        return pp_norm

    @torch.no_grad()
    def try_sgd_precond_update(self, params, step_size, params_current, grad_current, momentum):
        if self.gv_option in ['scalar']:
            zipped = zip(params, params_current, grad_current, self.state['gv'])
        
            for p_next, p_current, g_current, gv_i in zipped:
                p_next.data = p_current - (step_size / torch.sqrt(gv_i)) * g_current
        
        elif self.gv_option == 'per_param':
            if self.base_opt == 'adam':
                zipped = zip(params, params_current, grad_current, self.state['gv'], self.state['mv'])
                for p_next, p_current, g_current, gv_i, mv_i in zipped:
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
                    pv_list = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                    if momentum == 0. or self.mom_type == 'heavy_ball':
                        mv_i_scaled = g_current
                    elif self.mom_type == 'standard':
                        mv_i_scaled = scale_vector(mv_i, momentum, self.state['step']+1)

                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list *  mv_i_scaled), alpha=- step_size)
            
            elif self.base_opt == 'amsgrad':
                zipped = zip(params, params_current, grad_current, self.state['gv'], self.state['mv'])
                
                for i, (p_next, p_current, g_current, gv_i, mv_i) in enumerate(zipped):
                    self.state['gv_max'][i] = torch.max(gv_i, self.state['gv_max'][i])
                    gv_i_scaled = scale_vector(self.state['gv_max'][i], self.beta, self.state['step']+1)
                    pv_list = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)
                    
                    if momentum == 0. or  self.mom_type == 'heavy_ball':
                        mv_i_scaled = g_current
                    elif self.mom_type == 'standard':
                        mv_i_scaled = scale_vector(mv_i, momentum, self.state['step']+1)
                    else:
                        raise ValueError('does not exist')

                    # p_next.data = p_current - step_size * (pv_list *  mv_i_scaled)
                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list *  mv_i_scaled), alpha=- step_size)

            elif (self.base_opt in ['rmsprop', 'adagrad']):
                zipped = zip(params, params_current, grad_current, self.state['gv'])
                for p_next, p_current, g_current, gv_i in zipped:
                    pv_list = 1./ (torch.sqrt(gv_i) + 1e-8)
                    # p_next.data = p_current - step_size * (pv_list *  g_current)
    
                    p_next.data[:] = p_current.data
                    p_next.data.add_( (pv_list *  g_current), alpha=- step_size)

            elif (self.base_opt in ['diag_hessian', 'diag_ggn_ex', 'diag_ggn_mc']):
                zipped = zip(params, params_current, grad_current, self.state['gv'])
                for p_next, p_current, g_current, gv_i in zipped:
                    pv_list = 1./ (gv_i+ 1e-8)  # adding 1e-8 to avoid overflow.
                    # p_next.data = p_current - step_size * (pv_list *  g_current)

                    # need to do this variant of the update for LSTM memory problems.
                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list *  g_current), alpha=- step_size)
            

            else:
                raise ValueError('%s does not exist' % self.base_opt)

        else:
            raise ValueError('%s does not exist' % self.gv_option)

def scale_vector(vector, alpha, step, eps=1e-8):
    scale = (1-alpha**(max(1, step)))
    return vector / scale

