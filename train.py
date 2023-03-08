import torch


def d(f, x):
	df, = torch.autograd.grad(torch.sum(f), x, create_graph=True)
	return torch.autograd.grad(torch.sum(df), x, create_graph=True)[0]


def train(params, lr, epochs):
	tt = 0
	optimizer = torch.optim.Adam(params, lr=lr)
	while True:
		optimizer.zero_grad()
		if tt % sc_sampling == 0:
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

		def linear(x, A, b):
			return torch.einsum('ij,j...->i...', x, A) + b

		r1 = torch.sqrt((x - R)**2 + y**2 + z**2)
		r2 = torch.sqrt((x + R)**2 + y**2 + z**2)
		f1 = torch.exp(-r1)
		f2 = torch.exp(-r2)
		h = torch.sigmoid(linear(torch.hstack((f1, f2)), H1a, H1b))
		h = torch.sigmoid(linear(h, H2a, H2b))
		h = linear(2 * h, H3a, H3b)
		l = torch.sigmoid(linear(R, L1a, L1b))
		l = linear(l, L2a, L2b)
		e = torch.sigmoid(linear(R, E1a, E1b))
		e = torch.sigmoid(linear(e, E2a, E2b))
		e = linear(e, E3a, E3b)
		psi = h * l + f1 + f2
		res = d(psi, x) + d(psi, y) + d(psi, z) + (e + 1 / r1 + 1 / r2) * psi
		Ltot = (res**2).mean() + (psi[i1]**2).mean() + (psi[i2]**2).mean()
		if tt == 0 or Ltot.detach().numpy() < Lbest:
			Lbest = Ltot.detach().numpy()
			best = [params.clone().detach() for params in params]
		if tt % 10 == 0:
			print("%8d: %.5e [%.5e]" % (tt, Ltot.detach().numpy(), Lbest))
		if tt == epochs:
			with torch.no_grad():
				for a, b in zip(params, best):
					a.copy_(b)
			break
		tt += 1
		Ltot.backward(retain_graph=False)
		optimizer.step()


torch.manual_seed(123456)
dtype = torch.double
torch.set_default_tensor_type('torch.DoubleTensor')
BCcutoff = 17.5
cutOff = 0.005
L = 18
n = 500000
Rlo = 0.2
Rhi = 3
sc_sampling = 1
nh = 16
ne = 32
nl = 10
H1a = torch.zeros(2, nh, requires_grad=True)
H1b = torch.zeros(nh, requires_grad=True)
H2a = torch.zeros(nh, nh, requires_grad=True)
H2b = torch.zeros(nh, requires_grad=True)
H3a = torch.zeros(nh, 1, requires_grad=True)
H3b = torch.zeros(1, requires_grad=True)
L1a = torch.zeros(1, nl, requires_grad=True)
L1b = torch.zeros(nl, requires_grad=True)
L2a = torch.zeros(nl, 1, requires_grad=True)
L2b = torch.zeros(1, requires_grad=True)
E1a = torch.zeros(1, ne, requires_grad=True)
E1b = torch.zeros(ne, requires_grad=True)
E2a = torch.zeros(ne, ne, requires_grad=True)
E2b = torch.zeros(ne, requires_grad=True)
E3a = torch.zeros(ne, 1, requires_grad=True)
E3b = torch.zeros(1, requires_grad=True)
x = torch.empty(n, 1, dtype=dtype, requires_grad=True)
y = torch.empty(n, 1, dtype=dtype, requires_grad=True)
z = torch.empty(n, 1, dtype=dtype, requires_grad=True)
R = torch.empty(n, 1, dtype=dtype, requires_grad=True)
params = (H1a, H1b, H2a, H2b, H3a, H3b, L1a, L1b, L2a, L2b, E1a, E1b, E2a, E2b,
          E3a, E3b)
train(params, lr=8e-3, epochs=501)
train((E1a, E1b, E2a, E2b, E3a, E3b), lr=1e-4, epochs=501)
with torch.no_grad():
	with open("model.bin", "wb") as file:
		for x in params:
			x = x.numpy()
			file.write(x.ndim.to_bytes(4, "little"))
			for d in x.shape:
				file.write(d.to_bytes(4, "little"))
			file.write(x.tobytes())
