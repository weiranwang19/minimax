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


def project_onto_simplex(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Projects a tensor onto the probability simplex along a specified dimension.
    
    Args:
        v (torch.Tensor): The input tensor to project.
        dim (int): The dimension along which to project. Default is -1.
        
    Returns:
        torch.Tensor: The projected tensor, where values along `dim` sum to 1 and are >= 0.
    """
    # 1. Sort the vector in descending order
    u, _ = torch.sort(v, descending=True, dim=dim)

    # 2. Compute the cumulative sum of the sorted tensor
    cssv = torch.cumsum(u, dim=dim)

    # 3. Create a sequence of indices (1, 2, ..., N)
    # Reshape it to broadcast correctly against the target dimension
    N = v.shape[dim]
    seq = torch.arange(1, N + 1, dtype=v.dtype, device=v.device)
    shape = [1] * v.dim()
    shape[dim] = -1
    seq = seq.view(*shape)

    # 4. Find the threshold index (rho)
    # The condition is: u_j - (sum_{i=1}^j u_i - 1) / j > 0
    cond = u - (cssv - 1.0) / seq > 0
    
    # Since the sequence is monotonically decreasing, the number of True 
    # values exactly identifies our target index rho
    rho = cond.sum(dim=dim, keepdim=True)

    # 5. Compute the Lagrange multiplier threshold (tau)
    # We need the cumulative sum at index rho - 1
    rho_idx = (rho - 1).clamp(min=0) 
    cssv_rho = torch.gather(cssv, dim=dim, index=rho_idx)
    tau = (cssv_rho - 1.0) / rho.type(v.dtype)

    # 6. Apply the soft-thresholding to project the original vector
    w = torch.clamp(v - tau, min=0.0)
    
    return w
