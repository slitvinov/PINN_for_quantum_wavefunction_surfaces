import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import grad
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from os import path
import pickle
from scipy.integrate import simps
import warnings
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
	params['lossPath'] = "data/loss_ionH.pkl"
	params['EnergyPath'] = "data/energy_ionH.pkl"
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


	## Misc physical functions
def radial(x, y, z, R, params):
	# Returns the radial part from cartesian coordinates
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


	## Misc helper functions
def sampling(params, n_points, linearSampling=False):
	# Sampling from a 4d space: 3d variable (x,y,z) and 1d parameter (R) space
	xR = params['xR']
	xL = params['xL']
	yR = params['yR']
	yL = params['yL']
	zR = params['zR']
	zL = params['zL']
	cutOff = params['cutOff']
	if linearSampling == True:
		x = torch.linspace(xL, xR, n_points, requires_grad=False)
		y = torch.linspace(yL, yR, n_points, requires_grad=False)
		z = torch.linspace(zL, zR, n_points, requires_grad=False)
		R = torch.linspace(params['RxL'],
		                   params['RxR'],
		                   n_points,
		                   requires_grad=False)
	else:
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


def returnGate():
	modelTest = NN_ion(params)
	modelTest.loadModel(params)
	R = torch.linspace(params['RxL'],
	                   params['RxR'],
	                   params['n_train'],
	                   requires_grad=False)
	R = R.reshape(-1, 1)
	R.requires_grad = True
	f = modelTest.netDecayL(R)
	f = modelTest.sig(f)
	f = modelTest.netDecay(f)
	return R.cpu().detach().numpy(), f.cpu().detach().numpy()


def plotLoss(params, saveFig=True):
	with open(params['lossPath'], 'rb') as f:
		loaded_dict = pickle.load(f)
	plt.figure(figsize=[19, 8])
	plt.subplot(1, 2, 1)
	plt.plot(loaded_dict['Ltot'], label='total', linewidth=lineW * 2)
	plt.plot(loaded_dict['Lpde'], label='pde', linewidth=lineW)
	plt.plot(loaded_dict['Lbc'], label='bc', linewidth=lineW)
	plt.ylabel('Loss')
	plt.xlabel('epochs')
	plt.axvline(params['epochs'],
	            c='r',
	            linestyle='--',
	            linewidth=lineW * 1.5,
	            alpha=0.7)
	plt.yscale('log')
	plt.legend()
	plt.subplot(1, 2, 2)
	plt.plot(loaded_dict['Energy'], '-k', linewidth=lineW)
	plt.ylabel('Energy')
	plt.xlabel('epochs')
	plt.axvline(params['epochs'],
	            c='r',
	            linestyle='--',
	            linewidth=lineW * 1.5,
	            alpha=0.7)
	plt.tight_layout()
	if saveFig == True:
		plt.savefig('figures/loss_figure.jpg', format='jpg')


# ## Network Architecture
##----------------------- Network Class ---------------------
# Neural Network Architecture
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
		## ENERGY PARAMETER
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
	TeP0 = time.time()  # for counting the training time
	n_points = params['n_train']  # the training batch size
	x, y, z, R = sampling(params, n_points, linearSampling=False)
	r1, r2 = radial(x, y, z, R, params)
	bIndex1 = torch.where(r1 >= params['BCcutoff'])
	bIndex2 = torch.where(r2 >= params['BCcutoff'])
	for tt in range(epochs):
		optimizer.zero_grad()
		if tt % params['sc_sampling'] == 0 and tt < 0.9 * epochs:
			x, y, z, R = sampling(params, n_points, linearSampling=False)
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
	TePf = time.time()
	runTime = TePf - TeP0
	lossDictionary = {
	    'Ltot': Ltot_h,
	    'Lbc': Lbc_h,
	    'Lpde': Lpde_h,
	    'Energy': E_h
	}
	saveLoss(params, lossDictionary)
	print('Runtime (min): ', runTime / 60)
	print('last learning rate: ', scheduler.get_last_lr())
	# return E,R


def evaluateMultipleModels(params, Rx_list, plots=False, saveEnergies=False):
	# params['xL']= -5; params['xR']=  5;
	params['yL'] = -0
	params['yR'] = 0
	params['zL'] = -0
	params['zR'] = 0
	x, y, z = sampling(params, params['n_test'], linearSampling=True)
	e_exact = np.zeros([len(Rx_list), 1])
	## R ->    [0.4 : 0.1 : 4.0 ]
	e_exact = [
	    -1.8, -1.67, -1.55, -1.45, -1.36, -1.28, -1.22, -1.156, -1.10, -1.06,
	    -1.01, -0.98, -0.94, -0.91, -0.88, -0.86, -0.84, -0.81, -0.80
	]
	e_all = np.zeros([len(Rx_list), 1])
	Etot = np.zeros([len(Rx_list), 1])
	Etot_exact = np.zeros([len(Rx_list), 1])
	E_int = np.zeros([len(Rx_list), 1])
	E_int_L = np.zeros([len(Rx_list), 1])
	j = 0
	for r in Rx_list:
		params['Rx'] = r
		params['lossPath'] = "data/loss_ionH" + '_R=' + str(round(2 * r,
		                                                          1)) + '.pkl'
		params['saveModelPath'] = "models/ionH" + '_R=' + str(round(2 * r,
		                                                            1)) + ".pt"
		params['loadModelPath'] = "models/ionH" + '_R=' + str(round(2 * r,
		                                                            1)) + ".pt"
		modelTest = NN_ion(params)
		modelTest.loadModel(params)
		psi, E = modelTest.parametricPsi(x, y, z)
		e_all[j] = E[-1].detach().numpy()
		# print('Interatomic distance: ', 2*params['Rx'])
		Etot[j] = E[-1].detach().numpy() + 1 / (2 * r)
		Etot_exact[j] = e_exact[j] + 1 / (2 * r)
		if plots == True:
			plot_psi(params, plotSurf=False)
			plt.title('r = ' + str(r))
			plotLoss(params, saveFig=False)
			plt.title('r = ' + str(r))
		E_int[j], Eparameter, E_int_L[j] = energy_from_psi(params,
		                                                   printE=False,
		                                                   calcLCAO=False)
		E_int[j] = E_int[j] + 1 / (2 * r)
		E_int_L[j] = E_int_L[j] + 1 / (2 * r)
		j += 1
	if saveEnergies == True:
		EnergyDictionary = {
		    'Etot': Etot,
		    'e_all': e_all,
		    'Etot_exact': Etot_exact,
		    'E_int': E_int,
		    'E_int_L': E_int_L
		}
		with open(params['EnergyPath'], 'wb') as f:
			pickle.dump(EnergyDictionary, f)


	# return Etot, e_all, Etot_exact, E_int, E_int_L
def plot_EforR(params, Rx_list, plotIntegral=False):
	with open(params['EnergyPath'], 'rb') as f:
		enr_dic = pickle.load(f)
	Etot, Etot_exact = enr_dic['Etot'], enr_dic['Etot_exact']
	E_int, E_int_L = enr_dic['E_int'], enr_dic['E_int_L']
	# e_all =enr_dic['e_all'],
	e_lcao, Etot_lcao, R_lcao = LCAO_dispersion()
	plt.figure(figsize=[12, 8])
	plt.plot(Rx_list, Etot, 'o-b', label='Network')
	plt.plot(Rx_list, Etot_exact, '^-k', label='Exact')
	plt.plot(R_lcao, Etot_lcao, '*-m', label='LCAO', alpha=0.7)
	if plotIntegral == True:
		plt.plot(Rx_list, E_int, 'x-g', label='network_integral')
		# plt.plot(Rx_list ,E_int_L,'*-r',label='LCAO integral', alpha=0.7)
	plt.legend()
	# plt.xlim([0.9*min(Rx_list),1.1*max(Rx_list)])
	plt.xlim([0, 2])
	plt.ylim([-1, 1])
	plt.xlabel('Half interatomic distance')
	plt.ylabel('Energy')
	plt.tight_layout()
	# plt.ylim([-2.5,-0.5])
	plt.savefig('figures/dispersion.jpg', format='jpg')

warnings.filterwarnings('ignore')
dtype = torch.double
torch.set_default_tensor_type('torch.DoubleTensor')
lineW = 3
lineBoxW = 2
params = set_params()
params['epochs'] = 1
nEpoch1 = params['epochs']
params['n_train'] = 100000
params['lr'] = 8e-3
#### ----- Training: Single model ---=---------
train(params, loadWeights=False);
optEpoch = 3598
plotLoss(params, saveFig=False)
# ### GATE: Network-importance function
Rg, gate = returnGate()
plt.plot(Rg, gate, linewidth=lineW)
# ### Fine tuning
# params=set_params()
params['loadModelPath'] = "models/ionHsym.pt"
params['lossPath'] = "data/loss_ionH_fineTune.pkl"
params['EnergyPath'] = "data/energy_ionH_fineTune.pkl"
params['saveModelPath'] = "models/ionHsym_fineTune.pt"
# params['sc_step'] = 10000; params['sc_decay']=.7
params['sc_sampling'] = 1
params['epochs'] = 1
nEpoch2 = params['epochs']
params['n_train'] = 100000
params['lr'] = 5e-4
train(params, loadWeights=True, freezeUnits=True);
