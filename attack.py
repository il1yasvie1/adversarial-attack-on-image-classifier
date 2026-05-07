import torch
import torch.nn as nn
import numpy as np


"""
Linf: pgd, fw, fgsm, ifgsm, mifgsm
L2: lbfgs, pgd, fw, fgsm, ifgsm
L1: fw, fgsm, ifgsm
"""


def lbfgs(model, x0, y, e, p, N):
    d = torch.zeros_like(x0, requires_grad=True)
    c = 1.0
    optimizer = torch.optim.LBFGS([d], lr=e, max_iter=N)
    def closure():
        optimizer.zero_grad()
        J_per_sample = nn.CrossEntropyLoss(reduction='none')(model(x0 + d), y)
        norm_per_sample = d.flatten(1).norm(p=p, dim=1)
        total_loss = (norm_per_sample - c * J_per_sample).sum()
        total_loss.backward()
        return total_loss
    optimizer.step(closure)
    return d.detach()


def clip_l1(d, e):
    v = d.view(d.shape[0], -1)
    v_abs = v.abs()
    if (v_abs.sum(1) <= e).all(): return d
    u, _ = torch.sort(v_abs, descending=True, dim=1)
    sv = torch.cumsum(u, dim=1)
    rho = torch.count_nonzero(u * torch.arange(1, v.shape[1]+1, device=d.device) > (sv - e), dim=1)
    theta = (sv[torch.arange(d.shape[0]), rho - 1] - e) / rho
    return (torch.sign(v) * torch.max(v_abs - theta.view(-1, 1), torch.zeros_like(v))).view_as(d)


def projected_gradient_descent(model, x0, y, e, p, N):
    assert p in [1,2,np.inf]
    d = torch.zeros_like(x0, requires_grad=True)
    a = e/N
    for i in range(N):
        x = x0 + d
        J = nn.CrossEntropyLoss(reduction='sum')(model(x), y)
        J.backward()
        with torch.no_grad():
            if p == np.inf:
                d.data = (d + a * d.grad).clamp(-e, e)
            elif p == 2:
                g = d.grad
                g_norm = g.flatten(1).norm(2, 1).view(-1,1,1,1)
                d_ = d + a * g / g_norm
                d_norm = d_.flatten(1).norm(2, 1).view(-1,1,1,1)
                d.data = d_ * torch.clamp(e / d_norm, max=1)
            elif p == 1:
                d_ = d + a * d.grad 
                d.data = clip_l1(d_, e)
            else:
                raise NotImplementedError
            d.grad.zero_()
    return d.detach()


def frank_wolfe(model, x0, y, e, p, N):
    assert p in [1,2,np.inf]
    d = torch.zeros_like(x0, requires_grad=True)
    for i in range(N):
        nn.CrossEntropyLoss(reduction='sum')(model(x0 + d), y).backward()
        with torch.no_grad():
            if p == 1:
                g = d.grad.flatten(1)
                idx = g.abs().argmax(1, keepdim=True)
                s = torch.zeros_like(g).scatter_(1, idx, e * g.gather(1, idx).sign()).view_as(d)
            
            elif p == 2:
                g = d.grad
                g_norm = g.flatten(1).norm(2, 1).view(-1,1,1,1)
                s = e * g / g_norm
            
            elif p == np.inf:
                s = e * d.grad.sign()
            
            else:
                raise NotImplementedError

            gamma = 2 / (i + 2)
            d.data = (1 - gamma) * d + gamma * s

        d.grad.zero_()
    return d.detach()


def fgsm(model, x0, y, e, p, N=1):
	assert p in [1,2,np.inf]
	assert N == 1
	d = torch.zeros_like(x0)
	x = x0.clone()
	x.requires_grad = True
	J = nn.CrossEntropyLoss(reduction='sum')(model(x), y)
	J.backward()
	with torch.no_grad():
		if p == np.inf:
			d = e * x.grad.sign()

		elif p == 2:
			g_norm = x.grad.flatten(1).norm(2, 1).view(-1,1,1,1)
			d = e * x.grad / g_norm

		elif p == 1:
			g = x.grad.flatten(1)
			K = 10
			_, indices = g.abs().topk(K, dim=1)
			s = torch.zeros_like(g)
			val = (e / K) * g.gather(1, indices).sign()
			d = s.scatter_(1, indices, val).view_as(x0)

		else:
			raise NotImplementedError
	return d


def ifgsm(model, x0, y, e, p, N):
    assert p in [1,2,np.inf]
    d = torch.zeros_like(x0, requires_grad=True)
    a = e / N
    
    for i in range(N):
        J = nn.CrossEntropyLoss(reduction='sum')(model(x0 + d), y)
        J.backward()
        with torch.no_grad():
            grad = d.grad
            if p == np.inf:
                d.data = (d + a * grad.sign()).clamp(-e, e)
                
            elif p == 2:
                g_norm = grad.flatten(1).norm(2, 1).view(-1,1,1,1)
                d.data = d + a * grad / g_norm
                d_norm = d.flatten(1).norm(2, 1).view(-1,1,1,1)
                d.data = d * torch.clamp(e / d_norm, max=1.0)
                
            elif p == 1:
                g = grad.flatten(1)
                K = int(10/N)
                _, indices = g.abs().topk(K, dim=1)
                delta = torch.zeros_like(g)
                val = (a / K) * g.gather(1, indices).sign()
                delta = delta.scatter_(1, indices, val).view_as(x0)
                d.data = d + delta
                d_norm = d.flatten(1).norm(1, 1).view(-1,1,1,1)
                d.data = d * torch.clamp(e / d_norm, max=1.0)

            else:
                raise NotImplementedError
        d.grad.zero_()
    return d.detach()


def mifgsm(model, x0, y, e, p, N):
    assert p in [1,2,np.inf]
    d = torch.zeros_like(x0, requires_grad=True)
    a = e / N
    mu = 1.0
    g_acc = torch.zeros_like(x0)
    
    for i in range(N):
        nn.CrossEntropyLoss(reduction='sum')(model(x0 + d), y).backward()
        
        with torch.no_grad():
            grad = d.grad
            grad_l1 = grad.flatten(1).norm(p=1, dim=1).view(-1,1,1,1)
            g_acc = mu * g_acc + grad / grad_l1
            
            if p == np.inf:
                d.data = (d + a * g_acc.sign()).clamp(-e, e)
                
            elif p == 2:
                g_norm = g_acc.flatten(1).norm(p=2, dim=1).view(-1,1,1,1)
                d_ = d + a * g_acc / g_norm
                d_norm = d_.flatten(1).norm(p=2, dim=1).view(-1,1,1,1)
                d.data = d_ * torch.clamp(e / d_norm, max=1.0)
                
            elif p == 1:
                g_flat = g_acc.flatten(1)
                K = int(10/N)
                _, indices = g_flat.abs().topk(K, dim=1)
                delta = torch.zeros_like(g_flat)
                val = (a / K) * g_flat.gather(1, indices).sign()
                delta = delta.scatter_(1, indices, val).view_as(x0)
                d_new = d + delta
                d_norm = d_new.flatten(1).norm(1, 1).view(-1,1,1,1)
                d.data = d_new * torch.clamp(e / d_norm, max=1.0)
            else:
                raise NotImplementedError
        d.grad.zero_()
    return d.detach()
