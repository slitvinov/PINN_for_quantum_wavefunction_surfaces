import torch
import itertools


def d2(f, x):
	df = torch.autograd.grad([f], [x],
	                         grad_outputs=torch.ones(x.shape, dtype=dtype),
	                         create_graph=True)[0]
	return torch.autograd.grad(df, [x],
	                           grad_outputs=torch.ones(x.shape, dtype=dtype),
	                           create_graph=True)[0]


def train():
	params = itertools.chain(
	    *(params.parameters()
	      for params in [H1, H2, out, E1, E2, Eout, netDecayL, netDecay]))
	optimizer = torch.optim.Adam(params, lr=lr)
	for tt in range(epochs):
		optimizer.zero_grad()
		if tt % sc_sampling == 0 and tt < 0.9 * epochs:
			x = 2 * L * torch.rand(n_train, 1) + L
			y = 2 * L * torch.rand(n_train, 1) + L
			z = 2 * L * torch.rand(n_train, 1) + L
			R = (RxL - RxR) * torch.rand(n_train, 1) + RxR
			r1 = (x - R)**2 + y**2 + z**2
			r2 = (x + R)**2 + y**2 + z**2
			x[r1 < cutOff**2] = cutOff
			x[r2 < cutOff**2] = cutOff
			x.requires_grad = True
			y.requires_grad = True
			z.requires_grad = True
			R.requires_grad = True
			r1sq = (x - R)**2 + y**2 + z**2
			r2sq = (x + R)**2 + y**2 + z**2
			i1 = torch.where(r1sq >= BCcutoff**2)
			i2 = torch.where(r2sq >= BCcutoff**2)
		e = torch.sigmoid(E2(torch.sigmoid(E1(R))))
		r1 = torch.sqrt((x - R)**2 + y**2 + z**2)
		r2 = torch.sqrt((x + R)**2 + y**2 + z**2)
		f1 = torch.exp(-r1)
		f2 = torch.exp(-r2)
		ff = torch.cat((f1, f2), 1)
		B = 2 * torch.sigmoid(H2(torch.sigmoid(H1(ff))))
		f = torch.sigmoid(netDecayL(R))
		psi = out(B) * netDecay(f) + f1 + f2
		res = d2(psi, x) + d2(psi, y) + d2(
		    psi, z) + (Eout(e) + 1 / r1 + 1 / r2) * psi
		Ltot = (res**2).mean() + (psi[i1]**2).mean() + (psi[i2]**2).mean()
		Ltot.backward(retain_graph=False)
		print("%.16e" % Ltot.detach().numpy())
		optimizer.step()


torch.manual_seed(123456)
dtype = torch.double
torch.set_default_tensor_type('torch.DoubleTensor')

BCcutoff = 17.5
cutOff = 0.005
L = 18
lr = 8e-3
n_test = 80
n_train = 100000
RxL = 0.2
RxR = 4
sc_sampling = 1

dense_neurons = 16
dense_neurons_E = 32
netDecay_neurons = 10
H1 = torch.nn.Linear(2, dense_neurons)
H2 = torch.nn.Linear(dense_neurons, dense_neurons)
out = torch.nn.Linear(dense_neurons, 1)
E1 = torch.nn.Linear(1, dense_neurons_E)
E2 = torch.nn.Linear(dense_neurons_E, dense_neurons_E)
Eout = torch.nn.Linear(dense_neurons_E, 1)
torch.nn.init.constant_(Eout.bias[0], -1)
netDecayL = torch.nn.Linear(1, netDecay_neurons)
netDecay = torch.nn.Linear(netDecay_neurons, 1)

epochs = 10
lr = 8e-3
train()

epochs = 10
lr = 5e-4
H1.weight.requires_grad = False
H1.bias.requires_grad = False
H2.weight.requires_grad = False
H2.bias.requires_grad = False
out.weight.requires_grad = False
out.bias.requires_grad = False
netDecayL.weight.requires_grad = False
netDecayL.bias.requires_grad = False
netDecay.weight.requires_grad = False
netDecay.bias.requires_grad = False
train()
