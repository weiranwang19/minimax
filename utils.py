import torch

def clone_vals(vals, requires_grad=False):
    return [p.clone().detach().requires_grad_(requires_grad) for p in vals]


def zero_vals(vals, requires_grad=False):
    return [torch.zeros_like(p).detach().requires_grad_(requires_grad) for p in vals]


def scale_vals(vals, scale):
    return [scale * v for v in vals]


def add_vals(vals1, vals2):
    return [v1 + v2 for v1, v2 in zip(vals1, vals2)]


def blend_vals(vals1, vals2, coeff1, coeff2):
    return [coeff1 * v1 + coeff2 * v2 for v1, v2 in zip(vals1, vals2)]


@torch.no_grad()
def assign_vals(params, values):
    for p, v in zip(params, values):
        p.data.copy_(v)


def compute_norm(vals):
    s = 0
    for v in vals:
        s += torch.sum(torch.square(v))
    return torch.sqrt(s)


def compute_prox(vals, prox_funcs, prox_coeff):
    prox_vals = []
    for p, prox in zip(vals, prox_funcs):
        prox_vals.append(prox(p, prox_coeff))
    return prox_vals


def compute_value_and_grad(values, func):
    working = clone_vals(values, requires_grad=True)
    loss = func(working)
    grads = torch.autograd.grad(loss, working, allow_unused=True)
    detached_grads = []
    for v, g in zip(working, grads):
        detached_grads.append(torch.zeros_like(v) if g is None else g.detach())
    return loss.detach(), detached_grads
