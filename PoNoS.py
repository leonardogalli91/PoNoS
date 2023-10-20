import time
import copy
import torch
import numpy as np
import contextlib


class PoNoS(torch.optim.Optimizer):
    """
    PoNoS Arguments:
         c=0.5, # line search sufficient decrease scaling constant
         c_p=0.1, # Polyak step size scaling constant
         delta=0.5, # cutting step
         zhang_xi=1, # Zhang xi, controlling the nonmonotonicity
         max_eta=10, # maximum step size
         min_eta=1e-06, #minimum step size
         f_star=0, # estimate of the min value of f
         save_backtracks=True # activate the memory-based resetting technique

         Note that PoNoS is like LBFGS from the LBFGS optimizer from pytorch,
         the step needs to be called like in the following:
         closure = lambda: loss_function(model, images, labels, backwards=False)
         opt.step(closure)
    """

    def __init__(self,
                 params,
                 c=0.5,
                 c_p=0.1,
                 delta=0.5,
                 zhang_xi=1,
                 max_eta=10,
                 min_eta=1e-06,
                 f_star=0,
                 save_backtracks=True):

        params = list(params)
        super().__init__(params, {})

        self.params = params
        self.c = c
        self.delta = delta
        self.lk = 0
        self.zhang_xi = zhang_xi

        self.state["Q_k"] = 0
        self.state["C_k"] = 0
        self.max_eta = max_eta
        self.min_eta = min_eta
        self.c_p = c_p
        self.save_backtracks = save_backtracks
        self.f_star = f_star

    def step(self, closure):
        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with self.random_seed_torch(int(seed)):
                return closure()

        # get loss and compute gradients
        loss = closure_deterministic()
        loss.backward()

        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = self.get_grad_list(self.params)
        grad_norm = self.compute_grad_norm(grad_current)

        # setting the Polyak initial step size
        polyak_step_size = loss / (self.c_p * grad_norm ** 2 + 1e-8)
        if self.save_backtracks:
            polyak_step_size = polyak_step_size * (self.delta ** self.lk)
        step_size = max(min(polyak_step_size, self.max_eta), self.min_eta)

        self.line_search(step_size, params_current, grad_current, loss, closure_deterministic, grad_norm)
        return loss

    def line_search(self, step_size, params_current, grad_current, loss, closure_deterministic, grad_norm):
        with torch.no_grad():

            # compute nonmonotone terms for the Zhang&Hager line search
            q_kplus1 = self.zhang_xi * self.state["Q_k"] + 1
            self.state["C_k"] = (self.zhang_xi * self.state["Q_k"] * self.state["C_k"] + loss.item()) / q_kplus1
            self.state["Q_k"] = q_kplus1

            grad_norm = self.maybe_torch(grad_norm)
            if grad_norm >= 1e-8 and loss.item() != 0:
                # check if condition is satisfied
                found = 0

                suff_dec = grad_norm ** 2

                for e in range(100):
                    # try a prospective step
                    self.try_sgd_update(self.params, step_size, params_current, grad_current)

                    # compute the loss at the next step; no need to compute gradients.
                    loss_next = closure_deterministic()
                    ref_value = max(self.state["C_k"], loss.item())
                    found, step_size = self.check_armijo_conditions(step_size=step_size,
                                                                    loss=ref_value,
                                                                    suff_dec=suff_dec,
                                                                    loss_next=loss_next,
                                                                    c=self.c,
                                                                    beta_b=self.delta)

                    if found == 1:
                        break

                # if line search exceeds 100 internal iterations
                if found == 0:
                    step_size = torch.tensor(data=1e-6)
                    self.try_sgd_update(self.params, 1e-6, params_current, grad_current)

                self.lk = max(self.lk + e - 1, 0)

            else:
                print("Grad norm is {} and loss is {}".format(grad_norm, loss.item()))
                step_size = 0
                loss_next = closure_deterministic()

        return step_size, loss_next

    def maybe_torch(self,value):
        if isinstance(value, torch.Tensor):
            return value.item()
        else:
            return value

    # Armijo line search
    def check_armijo_conditions(self, step_size, loss, suff_dec,
                                loss_next, c, beta_b):
        found = 0
        sufficient_decrease = (step_size) * c * suff_dec
        rhs = loss - sufficient_decrease
        break_condition = loss_next - rhs
        if (break_condition <= 0):
            found = 1
        else:
            step_size = step_size * beta_b

        return found, step_size

    def try_sgd_update(self, params, step_size, params_current, grad_current):
        zipped = zip(params, params_current, grad_current)

        for p_next, p_current, g_current in zipped:
            p_next.data = p_current - step_size * g_current

    def compute_grad_norm(self, grad_list):
        grad_norm = 0.
        for g in grad_list:
            if g is None:
                continue
            grad_norm += torch.sum(torch.mul(g, g))
        grad_norm = torch.sqrt(grad_norm)
        return grad_norm

    @contextlib.contextmanager
    def random_seed_torch(self, seed, device=0):
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

    def get_grad_list(self, params):
        return [p.grad for p in params]


#################### example of use of PoNoS ######################


if __name__ == '__main__':

    import torchvision
    from torch import nn
    from torch.nn import functional as F

    class Mlp(nn.Module):
        def __init__(self, input_size=784,
                     hidden_sizes=[512, 256],
                     n_classes=10,
                     bias=True, dropout=False):
            super().__init__()
            self.input_size = input_size
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for
                                                in_size, out_size in
                                                zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
            self.output_layer = nn.Linear(hidden_sizes[-1], n_classes, bias=bias)

        def forward(self, x):
            x = x.view(-1, self.input_size)
            out = x
            for layer in self.hidden_layers:
                Z = layer(out)
                out = F.relu(Z)

            logits = self.output_layer(out)

            return logits

        def n_params(self):
            return sum(p.numel() for p in self.parameters())

    def softmax_loss(model, images, labels, backwards=False):
        logits = model(images)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(logits, labels.view(-1))

        if backwards and loss.requires_grad:
            loss.backward()

        return loss

    @torch.no_grad()
    def compute_metric_on_dataset(model, dataset, device):
        model.eval()
        loader =  torch.utils.data.DataLoader(dataset, drop_last=False, batch_size=1024)
        score_sum = 0.
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            score_sum += softmax_loss(model, images, labels).item() * images.shape[0]
        score = float(score_sum / len(loader.dataset))
        return score

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = Mlp(n_classes=10)
    model.to(device)
    opt = PoNoS(params=model.parameters())
    train_set = torchvision.datasets.MNIST("../data", train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.5,), (0.5,))
                                          ]))

    for epoch in range(5):
        train_loader = torch.utils.data.DataLoader(train_set, drop_last=False, shuffle=True, batch_size=128)
        print(epoch, compute_metric_on_dataset(model, train_set, device))

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            closure = lambda: softmax_loss(model, images, labels, backwards=False)
            opt.step(closure)
