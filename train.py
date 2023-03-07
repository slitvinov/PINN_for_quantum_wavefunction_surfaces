import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import grad
from os import path
import pickle
def set_params():
	params = dict()
	boundaries = 18
	params['xL'] = -boundaries
	params['xR'] = boundaries
	params['yL'] = -boundaries
	params['yR'] = boundaries
	params['zL'] = -boundaries
	params['zR'] = boundaries
	params['BCcutoff'] = 17.5
	params['RxL'] = 0.2
	params['RxR'] = 4
	params['Ry'] = 0
	params['Rz'] = 0
	params['cutOff'] = 0.005
	params['lossPath'] = "loss_ionH.pkl"
	params['saveModelPath'] = "models/ionHsym.pt"
	params['loadModelPath'] = "models/ionHsym.pt"
	params['EnrR_path'] = "data/energy_R_ion.pkl"
	params['sc_step'] = 3000
	params['sc_decay'] = .7  ## WAS 3000
	params['sc_sampling'] = 1
	params['n_train'] = 100000
	params['n_test'] = 80
	params['epochs'] = int(5e3)
	#2e3
	params['lr'] = 8e-3
	params['inversion_symmetry'] = 1
	return params


class toR(torch.nn.Module):

	@staticmethod
	def forward(input):
		r2 = input[:, 0].pow(2) + input[:, 1].pow(2) + input[:, 2].pow(2)
		r = torch.sqrt(r2)
		r = r.reshape(-1, 1)
		return r


class atomicAct_s(torch.nn.Module):

	@staticmethod
	def forward(input):
		return torch.exp(-input)


	## Differential Operators using autograd:
def dfx(x, f):
	return grad([f], [x],
	            grad_outputs=torch.ones(x.shape, dtype=dtype),
	            create_graph=True)[0]


def d2fx(x, f):
	return grad(dfx(x, f), [x],
	            grad_outputs=torch.ones(x.shape, dtype=dtype),
	            create_graph=True)[0]


def lapl(x, y, z, f):
	# Laplacian operator
	f_xx, f_yy, f_zz = d2fx(x, f), d2fx(y, f), d2fx(z, f)
	return f_xx + f_yy + f_zz


def radial(x, y, z, R, params):
	Rx = R
	Ry = params['Ry']
	Rz = params['Rz']
	r1 = torch.sqrt((x - Rx).pow(2) + (y - Ry).pow(2) + (z - Rz).pow(2))
	r2 = torch.sqrt((x + Rx).pow(2) + (y + Ry).pow(2) + (z + Rz).pow(2))
	return r1, r2


def V(x, y, z, R, params):
	## Potential energy function
	r1, r2 = radial(x, y, z, R, params)
	potential = -1 / r1 - 1 / r2
	return potential


def hamiltonian(x, y, z, R, psi, params):
	laplacian = lapl(x, y, z, psi)
	return -0.5 * laplacian + V(x, y, z, R, params) * psi


def sampling(params, n_points):
	xR = params['xR']
	xL = params['xL']
	yR = params['yR']
	yL = params['yL']
	zR = params['zR']
	zL = params['zL']
	cutOff = params['cutOff']
	x = (xL - xR) * torch.rand(n_points, 1) + xR
	y = (yL - yR) * torch.rand(n_points, 1) + yR
	z = (zL - zR) * torch.rand(n_points, 1) + zR
	R = (params['RxL'] - params['RxR']) * torch.rand(n_points,
		                                         1) + params['RxR']
	r1, r2 = radial(x, y, z, R, params)
	x[r1 < cutOff] = cutOff
	x[r2 < cutOff] = cutOff
	x, y, z = x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)
	R = R.reshape(-1, 1)
	x.requires_grad = True
	y.requires_grad = True
	z.requires_grad = True
	R.requires_grad = True
	return x, y, z, R


def saveLoss(params, lossDictionary):
	with open(params['lossPath'], 'wb') as f:
		pickle.dump(lossDictionary, f)


class NN_ion(nn.Module):
	def __init__(self,
	             params,
	             dense_neurons=16,
	             dense_neurons_E=32,
	             netDecay_neurons=10):
		super(NN_ion, self).__init__()
		self.sig = nn.Sigmoid()
		self.toR = toR()
		self.actAO_s = atomicAct_s()
		self.Lin_H1 = torch.nn.Linear(2, dense_neurons)
		self.Lin_H2 = torch.nn.Linear(dense_neurons, dense_neurons, bias=True)
		self.Lin_out = torch.nn.Linear(dense_neurons, 1)
		self.Lin_E1 = torch.nn.Linear(1, dense_neurons_E)
		self.Lin_E2 = torch.nn.Linear(dense_neurons_E, dense_neurons_E)
		self.Lin_Eout = torch.nn.Linear(dense_neurons_E, 1)
		nn.init.constant_(self.Lin_Eout.bias[0], -1)
		self.Ry = params['Ry']
		self.Rz = params['Rz']
		self.P = params['inversion_symmetry']
		self.netDecayL = torch.nn.Linear(1, netDecay_neurons, bias=True)
		self.netDecay = torch.nn.Linear(netDecay_neurons, 1, bias=True)

	def forward(self, x, y, z, R):
		e = self.Lin_E1(R)
		e = self.sig(e)
		e = self.Lin_E2(e)
		e = self.sig(e)
		E = self.Lin_Eout(e)
		## ATOMIC Layer: Radial part and physics-based activation
		fi_r1, fi_r2 = self.atomicUnit(x, y, z, R)
		fi_r1m, fi_r2m = self.atomicUnit(-x, y, z, R)
		## LCAO SOLUTION
		N_LCAO = self.lcao_solution(fi_r1, fi_r2)
		## NONLINEAR HIDDEN LAYERS
		B = self.base(fi_r1, fi_r2) + self.P * self.base(fi_r1m, fi_r2m)
		NN = self.Lin_out(B)
		f = self.netDecayL(R)
		f = self.sig(f)
		f = self.netDecay(f)
		NN = NN * f
		Nout = NN + N_LCAO
		return Nout, E

	def atomicUnit(self, x, y, z, R):
		x1 = x - R
		y1 = y - self.Ry
		z1 = z - self.Rz  # Cartesian Translation & Scaling:
		rVec1 = torch.cat((x1, y1, z1), 1)
		r1 = self.toR(rVec1)
		fi_r1 = self.actAO_s(r1)
		# s- ATOMIC ORBITAL ACTIVATION
		# --
		x2 = x + R
		y2 = y + self.Ry
		z2 = z + self.Rz
		rVec2 = torch.cat((x2, y2, z2), 1)
		r2 = self.toR(rVec2)
		fi_r2 = self.actAO_s(r2)
		return fi_r1, fi_r2

	def lcao_solution(
	    self,
	    fi_r1,
	    fi_r2,
	):
		## LCAO solution: Linear combination
		N_LCAO = (fi_r1 + self.P * fi_r2)
		return N_LCAO

	def base(self, fi_r1, fi_r2):
		## NONLINEAR HIDDEN LAYERS; Black box
		fi_r = torch.cat((fi_r1, fi_r2), 1)
		fi_r = self.Lin_H1(fi_r)
		fi_r = self.sig(fi_r)
		fi_r = self.Lin_H2(fi_r)
		fi_r = self.sig(fi_r)
		# fi_r = self.Lin_H3(fi_r);         fi_r = self.sig(fi_r)
		return fi_r

	def freezeBase(self):
		#         for p in self.parameters():
		#             p.requires_grad=False
		self.Lin_H1.weight.requires_grad = False
		self.Lin_H1.bias.requires_grad = False
		self.Lin_H2.weight.requires_grad = False
		self.Lin_H2.bias.requires_grad = False
		self.Lin_out.weight.requires_grad = False
		self.Lin_out.bias.requires_grad = False

	def freezeDecayUnit(self):
		self.netDecayL.weight.requires_grad = False
		self.netDecayL.bias.requires_grad = False
		self.netDecay.weight.requires_grad = False
		self.netDecay.bias.requires_grad = False

	def parametricPsi(self, x, y, z, R):
		N, E = self.forward(x, y, z, R)
		return N, E

	def loadModel(self, params):
		checkpoint = torch.load(params['loadModelPath'], map_location=torch.device('cpu'))
		self.load_state_dict(checkpoint['model_state_dict'])
		self.eval()

	def saveModel(self, params, optimizer):
		torch.save(
		    {
		        # 'epoch': epoch,
		        'model_state_dict': self.state_dict(),
		        'optimizer_state_dict': optimizer.state_dict(),
		        # 'loss': loss
		    },
		    params['saveModelPath'])

	def LossFunctions(self, x, y, z, R, params, bIndex1, bIndex2):
		lam_bc, lam_pde = 1, 1  #lam_tr = 1e-9
		psi, E = self.parametricPsi(x, y, z, R)
		#--# PDE
		res = hamiltonian(x, y, z, R, psi, params) - E * psi
		LossPDE = (res.pow(2)).mean() * lam_pde
		Ltot = LossPDE
		#--# BC
		Lbc = lam_bc * ((psi[bIndex1].pow(2)).mean() +
		                (psi[bIndex2].pow(2)).mean())
		Ltot = LossPDE + Lbc
		#
		#--# Trivial
		# Ltriv = 1/(psi.pow(2)).mean()* lam_tr ;    Ltot = Ltot + Ltriv
		return Ltot, LossPDE, Lbc, E


# ## Training: Helper Function
def train(params, loadWeights=False, freezeUnits=False):
	lr = params['lr']
	model = NN_ion(params)
	# modelBest=copy.deepcopy(model)
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
	print('train with Adam')
	# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
	# print('train with SGD')
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
	                                            step_size=params['sc_step'],
	                                            gamma=params['sc_decay'])
	Llim = 10
	optEpoch = 0
	epochs = params['epochs']  # For Adam
	total_epochs = epochs
	# total_epochs = params['epochs_LB'] + epochs
	Ltot_h = np.zeros([total_epochs, 1])
	Lpde_h = np.zeros([total_epochs, 1])
	Lbc_h = np.zeros([total_epochs, 1])
	E_h = np.zeros([total_epochs, 1])
	# Ltr_h = np.zeros([total_epochs,1]); Linv_h= np.zeros([total_epochs,1])
	## LOADING pre-trained model if PATH file exists and loadWeights=True
	if path.exists(params['loadModelPath']) and loadWeights == True:
		print('loading model')
		model.loadModel(params)
	if freezeUnits == True:
		print('Freezeing Basis unit and Gate')
		model.freezeDecayUnit()
		model.freezeBase()
	n_points = params['n_train']  # the training batch size
	x, y, z, R = sampling(params, n_points)
	r1, r2 = radial(x, y, z, R, params)
	bIndex1 = torch.where(r1 >= params['BCcutoff'])
	bIndex2 = torch.where(r2 >= params['BCcutoff'])
	for tt in range(epochs):
		optimizer.zero_grad()
		if tt % params['sc_sampling'] == 0 and tt < 0.9 * epochs:
			x, y, z, R = sampling(params, n_points)
			r1, r2 = radial(x, y, z, R, params)
			bIndex1 = torch.where(r1 >= params['BCcutoff'])
			bIndex2 = torch.where(r2 >= params['BCcutoff'])
		Ltot, LossPDE, Lbc, E = model.LossFunctions(x, y, z, R, params,
		                                            bIndex1, bIndex2)
		Ltot.backward(retain_graph=False)
		optimizer.step()
		# if  tt < 2001:
		#     scheduler.step()
		# keep history
		Ltot_h[tt] = Ltot.cpu().data.numpy()
		Lpde_h[tt] = LossPDE.cpu().data.numpy()
		Lbc_h[tt] = Lbc.cpu().data.numpy()
		E_h[tt] = E[-1].cpu().data.numpy()
		# Ltr_h[tt]  = Ltriv.data.numpy();
		#    Keep the best model (lowest loss). Checking after 50% of the total epochs
		if tt > 0.5 * epochs and Ltot < Llim:
			Llim = Ltot
			model.saveModel(params, optimizer)
			optEpoch = tt
	print('Optimal epoch: ', optEpoch)
	lossDictionary = {
	    'Ltot': Ltot_h,
	    'Lbc': Lbc_h,
	    'Lpde': Lpde_h,
	    'Energy': E_h
	}
	saveLoss(params, lossDictionary)
	print('last learning rate: ', scheduler.get_last_lr())

dtype = torch.double
torch.set_default_tensor_type('torch.DoubleTensor')
params = set_params()
params['epochs'] = 1
nEpoch1 = params['epochs']
params['n_train'] = 100000
params['lr'] = 8e-3
train(params, loadWeights=False, freezeUnits=False);
optEpoch = 3598
params['lossPath'] = "loss_ionH_fineTune.pkl"
params['sc_sampling'] = 1
params['epochs'] = 1
nEpoch2 = params['epochs']
params['n_train'] = 100000
params['lr'] = 5e-4
train(params, loadWeights=True, freezeUnits=True);
