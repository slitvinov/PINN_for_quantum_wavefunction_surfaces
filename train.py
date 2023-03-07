import torch
import itertools
import math


def d2(f, x):
	df = torch.autograd.grad([f], [x],
	                         grad_outputs=torch.ones(x.shape, dtype=dtype),
	                         create_graph=True)[0]
	return torch.autograd.grad(df, [x],
	                           grad_outputs=torch.ones(x.shape, dtype=dtype),
	                           create_graph=True)[0]


def train():
	optimizer = torch.optim.Adam(params, lr=lr)
	for tt in range(epochs):
		optimizer.zero_grad()
		if tt % sc_sampling == 0 and tt < 0.9 * epochs:
			x = 2 * L * torch.rand(n_train, 1) + L
			y = 2 * L * torch.rand(n_train, 1) + L
			z = 2 * L * torch.rand(n_train, 1) + L
			R = (RxL - RxR) * torch.rand(n_train, 1) + RxR
			r1sq = (x - R)**2 + y**2 + z**2
			r2sq = (x + R)**2 + y**2 + z**2
			x[r1sq < cutOff**2] = cutOff
			x[r2sq < cutOff**2] = cutOff
			r1sq = (x - R)**2 + y**2 + z**2
			r2sq = (x + R)**2 + y**2 + z**2
			i1 = torch.where(r1sq >= BCcutoff**2)
			i2 = torch.where(r2sq >= BCcutoff**2)
			x.requires_grad = True
			y.requires_grad = True
			z.requires_grad = True
			R.requires_grad = True
		r1 = torch.sqrt((x - R)**2 + y**2 + z**2)
		r2 = torch.sqrt((x + R)**2 + y**2 + z**2)
		f1 = torch.exp(-r1)
		f2 = torch.exp(-r2)
		psi = H3(
		    2 * torch.sigmoid(H2(torch.sigmoid(H1(torch.hstack(
		        (f1, f2))))))) * L2(torch.sigmoid(L1(R))) + f1 + f2
		res = d2(psi, x) + d2(psi, y) + d2(
		    psi, z) + (E3(torch.sigmoid(E2(torch.sigmoid(E1(R))))) + E3b +
		               1 / r1 + 1 / r2) * psi
		Ltot = (res**2).mean() + (psi[i1]**2).mean() + (psi[i2]**2).mean()
		Ltot.backward(retain_graph=False)
		optimizer.step()
		if tt % 100 == 0:
			print("%8d: %.8e" % (tt, Ltot.detach().numpy()))


torch.manual_seed(123456)
dtype = torch.double
torch.set_default_tensor_type('torch.DoubleTensor')

BCcutoff = 17.5
cutOff = 0.005
L = 18
n_train = 100000
RxL = 0.2
RxR = 3
sc_sampling = 1
h = 16
e = 32
l = 10
H1 = torch.nn.Linear(2, h)
H2 = torch.nn.Linear(h, h)
H3 = torch.nn.Linear(h, 1)
E1 = torch.nn.Linear(1, e)
E2 = torch.nn.Linear(e, e)
E3 = torch.nn.Linear(e, 1, bias=False)
E3b = -1
L1 = torch.nn.Linear(1, l)
L2 = torch.nn.Linear(l, 1)

epochs = 5000
lr = 8e-3
params = itertools.chain(*(params.parameters()
                           for params in [H1, H2, H3, E1, E2, E3, L1, L2]))
train()

epochs = 5000
lr = 1e-4
params = itertools.chain(*(params.parameters() for params in [E1, E2, E3]))
train()
