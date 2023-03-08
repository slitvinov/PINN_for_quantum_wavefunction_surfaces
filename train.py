import torch

def ini(*shape):
	x = torch.empty(shape, requires_grad=True)
	L = 1 / shape[0]**0.5
	with torch.no_grad():
		x.uniform_(-L, L)
	return x


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
			with torch.no_grad():
				x.uniform_(-L, L)
				y.uniform_(-L, L)
				z.uniform_(-L, L)
				R.uniform_(Rlo, Rhi)
				r1sq = (x - R)**2 + y**2 + z**2
				r2sq = (x + R)**2 + y**2 + z**2
				x[r1sq < cutOff**2] = cutOff
				x[r2sq < cutOff**2] = cutOff
				r1sq = (x - R)**2 + y**2 + z**2
				r2sq = (x + R)**2 + y**2 + z**2
				i1 = torch.where(r1sq >= BCcutoff**2)
				i2 = torch.where(r2sq >= BCcutoff**2)
		r1 = torch.sqrt((x - R)**2 + y**2 + z**2)
		r2 = torch.sqrt((x + R)**2 + y**2 + z**2)
		f1 = torch.exp(-r1)
		f2 = torch.exp(-r2)
		ff = torch.hstack((f1, f2))
		h1 = torch.sigmoid(ff @ H1a + H1b)
		h2 = torch.sigmoid(h1 @ H2a + H2b)
		h3 = 2 * h2 @ H3a + H3b
		l1 = torch.sigmoid(R @ L1a + L1b)
		l2 = l1 @ L2a + L2b
		e1 = torch.sigmoid(R @ E1a + E1b)
		e2 = torch.sigmoid(e1 @ E2a + E2b)
		e3 = e2 @ E3a + E3b
		psi = h3 * l2 + f1 + f2
		res = d2(psi, x) + d2(psi, y) + d2(psi,
		                                   z) + (e3 + 1 / r1 + 1 / r2) * psi
		Ltot = (res**2).mean() + (psi[i1]**2).mean() + (psi[i2]**2).mean()
		Ltot.backward(retain_graph=False)
		optimizer.step()
		if tt % 10 == 0:
			print("%8d: %.8e" % (tt, Ltot.detach().numpy()))


torch.manual_seed(123456)
dtype = torch.double
torch.set_default_tensor_type('torch.DoubleTensor')

BCcutoff = 17.5
cutOff = 0.005
L = 18
n_train = 1000
Rlo = 0.2
Rhi = 3
sc_sampling = 1
h = 16
e = 32
l = 10
H1a = ini(2, h)
H1b = ini(h)
H2a = ini(h, h)
H2b = ini(h)
H3a = ini(h, 1)
H3b = ini(1)
E1a = ini(1, e)
E1b = ini(e)
E2a = ini(e, e)
E2b = ini(e)
E3a = ini(e, 1)
E3b = ini(1)
L1a = ini(1, l)
L1b = ini(l)
L2a = ini(l, 1)
L2b = ini(1)
x = torch.empty(n_train, 1, requires_grad=True)
y = torch.empty(n_train, 1, requires_grad=True)
z = torch.empty(n_train, 1, requires_grad=True)
R = torch.empty(n_train, 1, requires_grad=True)

epochs = 5000
lr = 8e-3
params = (H1a, H1b, H2a, H2b, H3a, H3b, E1a, E1b, E2a, E2b, E3a, L1a, L1b, L2a,
          L2b)
train()

epochs = 5000
lr = 1e-4
params = (E1a, E1b, E2a, E2b, E3a)
train()
