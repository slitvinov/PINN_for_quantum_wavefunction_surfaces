import torch


def d2fx(x, f):
	df = torch.autograd.grad([f], [x],
	                         grad_outputs=torch.ones(x.shape, dtype=dtype),
	                         create_graph=True)[0]
	return torch.autograd.grad(df, [x],
	                           grad_outputs=torch.ones(x.shape, dtype=dtype),
	                           create_graph=True)[0]


def hamiltonian(x, y, z, R, psi):
	r1 = torch.sqrt((x - R)**2 + y**2 + z**2)
	r2 = torch.sqrt((x + R)**2 + y**2 + z**2)
	return -0.5 * (d2fx(x, psi) + d2fx(y, psi) + d2fx(z, psi)) - (1 / r1 +
	                                                              1 / r2) * psi


def sampling(n_points):
	x = 2 * L * torch.rand(n_points, 1) + L
	y = 2 * L * torch.rand(n_points, 1) + L
	z = 2 * L * torch.rand(n_points, 1) + L
	R = (RxL - RxR) * torch.rand(n_points, 1) + RxR
	r1 = (x - R)**2 + y**2 + z**2
	r2 = (x + R)**2 + y**2 + z**2
	x[r1 < cutOff**2] = cutOff
	x[r2 < cutOff**2] = cutOff
	x.requires_grad = True
	y.requires_grad = True
	z.requires_grad = True
	R.requires_grad = True
	return x, y, z, R


class NN_ion(torch.nn.Module):

	def __init__(self,
	             dense_neurons=16,
	             dense_neurons_E=32,
	             netDecay_neurons=10):
		super(NN_ion, self).__init__()
		self.sig = torch.nn.Sigmoid()
		self.Lin_H1 = torch.nn.Linear(2, dense_neurons)
		self.Lin_H2 = torch.nn.Linear(dense_neurons, dense_neurons, bias=True)
		self.Lin_out = torch.nn.Linear(dense_neurons, 1)
		self.Lin_E1 = torch.nn.Linear(1, dense_neurons_E)
		self.Lin_E2 = torch.nn.Linear(dense_neurons_E, dense_neurons_E)
		self.Lin_Eout = torch.nn.Linear(dense_neurons_E, 1)
		torch.nn.init.constant_(self.Lin_Eout.bias[0], -1)
		self.netDecayL = torch.nn.Linear(1, netDecay_neurons, bias=True)
		self.netDecay = torch.nn.Linear(netDecay_neurons, 1, bias=True)

	def atomicUnit(self, x, y, z, R):
		r1 = torch.sqrt((x - R)**2 + y**2 + z**2)
		r2 = torch.sqrt((x + R)**2 + y**2 + z**2)
		return torch.exp(-r1), torch.exp(-r2)

	def base(self, fi_r1, fi_r2):
		fi_r = torch.cat((fi_r1, fi_r2), 1)
		fi_r = self.Lin_H1(fi_r)
		fi_r = self.sig(fi_r)
		fi_r = self.Lin_H2(fi_r)
		return self.sig(fi_r)

	def LossFunctions(self, x, y, z, R, bIndex1, bIndex2):
		e = self.Lin_E1(R)
		e = self.sig(e)
		e = self.Lin_E2(e)
		e = self.sig(e)
		fi_r1, fi_r2 = self.atomicUnit(x, y, z, R)
		fi_r1m, fi_r2m = self.atomicUnit(-x, y, z, R)
		B = self.base(fi_r1, fi_r2) + self.base(fi_r1m, fi_r2m)
		f = self.netDecayL(R)
		f = self.sig(f)
		psi = self.Lin_out(B) * self.netDecay(f) + fi_r1 + fi_r2
		E = self.Lin_Eout(e)
		res = hamiltonian(x, y, z, R, psi) - E * psi
		LossPDE = (res.pow(2)).mean()
		Lbc = (psi[bIndex1].pow(2)).mean() + (psi[bIndex2].pow(2)).mean()
		return LossPDE + Lbc


def train():
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
	                                            step_size=sc_step,
	                                            gamma=sc_decay)
	for tt in range(epochs):
		optimizer.zero_grad()
		if tt % sc_sampling == 0 and tt < 0.9 * epochs:
			x, y, z, R = sampling(n_train)
			r1 = (x - R)**2 + y**2 + z**2
			r2 = (x + R)**2 + y**2 + z**2
			bIndex1 = torch.where(r1 >= BCcutoff**2)
			bIndex2 = torch.where(r2 >= BCcutoff**2)
		Ltot = model.LossFunctions(x, y, z, R, bIndex1, bIndex2)
		Ltot.backward(retain_graph=False)
		print("%.16e" % Ltot.detach().numpy())
		optimizer.step()
	print('last learning rate: ', scheduler.get_last_lr())


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
sc_decay = .7
sc_sampling = 1
sc_step = 3000
model = NN_ion()

epochs = 10
lr = 8e-3
train()

epochs = 10
lr = 5e-4
model.Lin_H1.weight.requires_grad = False
model.Lin_H1.bias.requires_grad = False
model.Lin_H2.weight.requires_grad = False
model.Lin_H2.bias.requires_grad = False
model.Lin_out.weight.requires_grad = False
model.Lin_out.bias.requires_grad = False
model.netDecayL.weight.requires_grad = False
model.netDecayL.bias.requires_grad = False
model.netDecay.weight.requires_grad = False
model.netDecay.bias.requires_grad = False
train()
