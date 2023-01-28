"""
	Class definition of AJoint-CRep, the algorithm to perform inference in networks with anomaly-reciprocity.
	The latent variables are related to community memberships, anomaly, and reciprocity value.
"""
#Updat the community parameters/eta/pi/mu/ likelihood for reciprocated case. 
# For the moment self.constrained AND use_approximation should be False, assortative. 

from __future__ import print_function
import time
import sys
import sktensor as skt
import numpy as np
import pandas as pd
from termcolor import colored 
import numpy.random as rn
import scipy.special
from scipy.stats import poisson

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  
from scipy.optimize import brentq, root, fixed_point

import time_glob as gl


EPS = 1e-12

class AnomalyDetection:
	def __init__(self, N=100, L=1, K=3, undirected=False, initialization=0, ag=1.1,bg=0.05,  rseed=1, inf=1e14, err_max=1e-8, err=0.1,
				 N_real=1, tolerance=0.001, decision=10, max_iter=700, out_inference=True,eta0=2000,use_approximation= False,
				 out_folder='../data/output/5-fold_cv/', end_file='.dat', assortative=True, pibr0 = 0.1, mupr0= 0.002,
				 in_parameters = '../data/input/synthetic/edge_anomalies/theta_500_3_20.0_0.3_0.1_False_0',
				 fix_eta=False,fix_communities=False,fix_w=False,fix_pibr=False, fix_mupr=False,plot_loglik=True,
				 constrained=False, verbose=False, flag_anomaly = False):
		self.N = N  # number of nodes
		self.L = L  # number of layers
		self.K = K  # number of communities
		self.undirected = undirected  # flag to call the undirected network
		self.rseed = rseed  # random seed for the initialization
		self.inf = inf  # initial value of the pseudo log-likelihood
		self.err_max = err_max  # minimum value for the parameters
		self.err = err  # noise for the initialization
		self.N_real = N_real  # number of iterations with different random initialization
		self.tolerance = tolerance  # tolerance parameter for convergence
		self.decision = decision  # convergence parameter
		self.max_iter = max_iter  # maximum number of EM steps before aborting
		self.out_inference = out_inference  # flag for storing the inferred parameters
		self.out_folder = out_folder  # path for storing the output
		self.end_file = end_file  # output file suffix
		self.assortative = assortative  # if True, the network is assortative
		self.fix_eta = fix_eta  # if True, keep the eta parameter fixed
		self.fix_pibr = fix_pibr  # if True, the pibr parameter is fixed
		self.fix_mupr = fix_mupr  # if True, the mupr parameter is fixed
		self.fix_communities = fix_communities
		self.use_approximation = use_approximation  # if True, use the approximated version of the updates
		self.fix_w = fix_w
		self.constrained = constrained  # if True, use the configuration with constraints on the updates
		self.verbose = verbose  # flag to print details
		self.ag = ag # shape of gamma prior
		self.bg = bg # rate of gamma prior
		self.pibr0 = pibr0  # initial value for the mu in
		self.mupr0 = mupr0  # initial value for the pi in Bernolie dist
		self.plot_loglik = plot_loglik
		self.flag_anomaly = flag_anomaly 
		self.in_parameters = in_parameters 

		if initialization not in {0, 1, 2}:  # indicator for choosing how to initialize u, v and w
			raise ValueError('The initialization parameter can be either 0, 1, or 2. It is used as an indicator to '
							 'initialize the membership matrices u and v and the affinity matrix w. If it is 0, they '
							 'will be generated randomly, if it is 1, u and v will be uploaded from file, and if 2,'
							 'then u,v, and w will be uploaded from file.')
		self.initialization = initialization  

		if (eta0 is not None) and (eta0 <= 0.):  # initial value for the pair interaction coefficient
			raise ValueError('If not None, the eta0 parameter has to be greater than 0.!')
		self.eta0 = eta0
		if self.fix_eta:
			if self.eta0 is None:
				raise ValueError('If fix_eta=True, provide a value for eta0.')

		if self.pibr0 is not None:
			if (self.pibr0 < 0) or (self.pibr0 > 1):
				raise ValueError('The value of pi  has to be in [0, 1]!') 


		if self.mupr0 is not None:
			if (self.mupr0 < 0) or (self.mupr0 > 1): 
				raise ValueError('The value of mu  has to be in [0, 1]!')

		if self.initialization > 0:
			theta = np.load(self.in_parameters + '.npz',allow_pickle=True) 
			self.N, self.K =  theta['u'].shape
			self.eta0 = theta['eta'] 
			# self.pibr0 = theta['pi']
			# self.mupr0 = theta['mu']
			print(self.N,self.K)

		# values of the parameters used during the update
		self.u = np.zeros((self.N, self.K), dtype=float)  # out-going membership
		self.v = np.zeros((self.N, self.K), dtype=float)  # in-going membership 
		self.eta = self.eta0  # pair interaction term

		# values of the parameters in the previous iteration
		self.u_old = np.zeros((self.N, self.K), dtype=float)  # out-going membership
		self.v_old = np.zeros((self.N, self.K), dtype=float)  # in-going membership
		self.eta_old = self.eta0  # pair interaction coefficient 
		self.pibr_old = self.pibr0  #
		self.mupr_old = self.mupr0 #

		# final values after convergence --> the ones that maximize the pseudo log-likelihood
		self.u_f = np.zeros((self.N, self.K), dtype=float)  # Out-going membership
		self.v_f = np.zeros((self.N, self.K), dtype=float)  # In-going membership 
		self.eta_f = self.eta0  # pair interaction coefficient
		self.pibr_f = self.pibr0  #
		self.mupr_f = self.mupr0 #

		# values of the affinity tensor: in this case w is always ASSORTATIVE 
		if self.assortative:  # purely diagonal matrix
			self.w = np.zeros((self.L, self.K), dtype=float)
			self.w_old = np.zeros((self.L, self.K), dtype=float)
			self.w_f = np.zeros((self.L, self.K), dtype=float)
		else:
			self.w = np.zeros((self.L, self.K, self.K), dtype=float)
			self.w_old = np.zeros((self.L, self.K, self.K), dtype=float)
			self.w_f = np.zeros((self.L, self.K, self.K), dtype=float)

		if self.undirected:
			if not (self.fix_eta and self.eta0 == 1):
				raise ValueError('If undirected=True, the parameter eta has to be fixed equal to 1 (s.t. log(eta)=0).')

		if self.fix_pibr == True:
			self.pibr = self.pibr_old = self.pibr_f = self.pibr0
		if self.fix_mupr == True:
			self.mupr = self.mupr_old = self.mupr_f = self.mupr0 

		if self.flag_anomaly == False:
			self.pibr = self.pibr_old = self.pibr_f = 0.
			self.mupr = self.mupr_old = self.mupr_f = 0.
			self.fix_pibr = self.fix_mupr = True
		
		if self.fix_eta:
			self.eta = self.eta_old = self.eta_f = self.eta0
		

	# @gl.timeit('fit')
	def fit(self, data, nodes, mask=None):
		"""
			Model directed networks by using a probabilistic generative model that assume community parameters and
			reciprocity coefficient. The inference is performed via EM algorithm.

			Parameters
			----------
			data : ndarray/sptensor
				   Graph adjacency tensor.
			data_T: None/sptensor
					Graph adjacency tensor (transpose).
			data_T_vals : None/ndarray
						  Array with values of entries A[j, i] given non-zero entry (i, j).
			nodes : list
					List of nodes IDs.
			mask : ndarray
				   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

			Returns
			-------
			u_f : ndarray
				  Out-going membership matrix.
			v_f : ndarray
				  In-coming membership matrix.
			w_f : ndarray
				  Affinity tensor.
			eta_f : float
					Pair interaction coefficient.
			pibr_f : float
					Bernolie parameter.
			mupr_f : float
					prior .
			maxL : float
				   Maximum pseudo log-likelihood.
			final_it : int
					   Total number of iterations.
		"""

		maxL = -self.inf  # initialization of the maximum pseudo log-likelihood 


		# if data_T is None:
		E = np.sum(data)  # weighted sum of edges (needed in the denominator of eta)
		data_T = np.einsum('aij->aji', data)
		data_T_vals = get_item_array_from_subs(data_T, data.nonzero())  
		# pre-processing of the data to handle the sparsity
		data = preprocess(data)
		data_T = preprocess(data_T) 

		# save the indexes of the nonzero entries
		if isinstance(data, skt.dtensor): 
			subs_nz = data.nonzero()
		elif isinstance(data, skt.sptensor): 
			subs_nz = data.subs  

		if mask is not None: 
			subs_nz_mask = mask.nonzero() 
			# mask = preprocess(mask)
			# if isinstance(mask, skt.dtensor):
			#     subs_nz_mask = mask.nonzero()
			# elif isinstance(mask, skt.sptensor):
			#     subs_nz_mask = mask.subs 
		else:
			subs_nz_mask = None
		
		self.AAtSum = (data.vals * data_T_vals).sum()


		rng = np.random.RandomState(self.rseed)

		for r in range(self.N_real):

			self._initialize(rng=rng, nodes=nodes)

			self._update_old_variables() 
			self._update_cache(data, data_T_vals, subs_nz)

			# convergence local variables
			coincide, it = 0, 0
			convergence = False
			loglik = self.inf 
			loglik_values = [loglik]

			if self.verbose  == 2:
				print(f'Updating realization {r} ...', end=' ')
			time_start = time.time()
			# --- single step iteration update ---
			while not convergence and it < self.max_iter:
				# main EM update: updates memberships and calculates max difference new vs old
				delta_u, delta_v, delta_w, delta_eta, delta_pibr, delta_mupr = self._update_em(data, data_T_vals, subs_nz, denominator=E,mask=mask,subs_nz_mask=subs_nz_mask)

				it, loglik, coincide, convergence, success = self._check_for_convergence(data,data_T, it, loglik, coincide, convergence,
															mask=mask,subs_nz_mask=subs_nz_mask,mod_it=1)
				loglik_values.append(loglik) 

			if maxL < loglik:
				self._update_optimal_parameters()
				maxL = loglik
				self.final_it = it
				conv = convergence
				best_loglik_values = list(loglik_values)
			self.rseed += rng.randint(100000000)
			

			if self.verbose > 0:
				print(f'Nreal = {r} - ELBO = {loglik} - ELBOmax = {maxL} - it = {it}  '

					  f'time = {np.round(time.time() - time_start, 2)} seconds')
		# end cycle over realizations

		self.maxL = maxL
		if self.final_it == self.max_iter and not conv:
			# convergence not reaches
			print(colored('Solution failed to converge in {0} EM steps!'.format(self.max_iter), 'blue'))

		if self.plot_loglik:
			plot_L(best_loglik_values, int_ticks=True)

		if self.out_inference:
			self.output_results(nodes) 
		

		return self.u_f, self.v_f, self.w_f, self.eta_f, self.pibr_f, self.mupr_f, maxL

	def _initialize(self, rng, nodes):
		"""
			Random initialization of the parameters u, v, w, beta.

			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		if rng is None:
			rng = np.random.RandomState(self.rseed)
		

		if self.eta0 is not None:
			self.eta = self.eta0
		else:
			if self.verbose:
				print('eta is initialized randomly.')
			self._randomize_eta(rng=rng)

		if (self.pibr0 is not None) and (not self.fix_pibr):
			self.pibr = self.pibr0
		else: 
			if self.fix_pibr == False: 
				if self.verbose:
					print('pi is initialized randomly.')
				self._randomize_pibr(rng) 
		
		if (self.mupr0 is not None) and (not self.fix_mupr):
			self.mupr = self.mupr0
		else: 
			if self.fix_mupr == False:   
				if self.verbose:
					print('mu is initialized randomly.')
				self._randomize_mupr(rng)

		if self.initialization == 0:
			if self.verbose == 2:
				print('u, v and w are initialized randomly.')
			self._randomize_w(rng=rng)
			self._randomize_u_v(rng=rng)

		

		elif self.initialization == 1:
			if self.verbose:
				print('u, v  are initialized using the input files; w is initialized randomly:')
				print(self.in_parameters + '.npz')
			theta = np.load(self.in_parameters + '.npz',allow_pickle=True)
			self._initialize_u(theta['u'])
			self._initialize_v(theta['v'])
			self.N = self.u.shape[0]

			self._randomize_w(rng=rng)
		

		elif self.initialization == 2:
			if self.verbose:
				print('u, v and w are initialized using the input files:')
				print(self.in_parameters + '.npz')
			theta = np.load(self.in_parameters + '.npz',allow_pickle=True)
			self._initialize_u(theta['u'])
			self._initialize_v(theta['v']) 
			self._initialize_w(theta['w'])
			self.N = self.u.shape[0]

	def _randomize_eta(self, rng):
		"""
			Generate a random number in (1., 50.).

			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		self.eta = rng.uniform(1.01, 49.99) 

	def _initialize_u(self, u0):
		if u0.shape[0] != self.N:
			raise ValueError('u.shape is different that the initialized one.',self.N,u0.shape[0])
		self.u = u0.copy()
		max_entry = np.max(u0)
		self.u += max_entry #* self.err * np.random.random_sample(self.u.shape)

	def _initialize_v(self, v0):
		if v0.shape[0] != self.N:
			raise ValueError('v.shape is different that the initialized one.',self.N,v0.shape[0])
		self.v = v0.copy()
		max_entry = np.max(v0)
		self.v += max_entry #* self.err * np.random.random_sample(self.v.shape)

	
	def _randomize_pibr(self, rng=None):
		"""
			Generate a random number in (0, 1.).

			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		if rng is None:
			rng = np.random.RandomState(self.rseed)
		self.pibr = rng.random_sample(1)[0]
	
	def _randomize_mupr(self, rng=None):
		"""
			Generate a random number in (0, 1.).

			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		if rng is None:
			rng = np.random.RandomState(self.rseed)
		self.mupr = rng.random_sample(1)[0]

	def _randomize_w(self, rng):
		"""
			Assign a random number in (0, 1.) to each entry of the affinity tensor w.

			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		if rng is None:
			rng = np.random.RandomState(self.rseed)
		for i in range(self.L):
			for k in range(self.K):
				if self.assortative:
					self.w[i, k] = 0.01 * rng.random_sample(1)
				else:
					for q in range(k, self.K):
						if q == k:
							self.w[i, k, q] = 0.01 * rng.random_sample(1)
						else:
							self.w[i, k, q] = self.w[i, q, k] = self.err * 0.01 * rng.random_sample(1)
		

	def _randomize_u_v(self, rng=None):
		"""
			Assign a random number in (0, 1.) to each entry of the membership matrices u and v, and normalize each row.

			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		if rng is None:
			rng = np.random.RandomState(self.rseed)
		self.u = rng.random_sample(self.u.shape)
		row_sums = self.u.sum(axis=1)
		self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

		# self.v = self.u.copy()
		# self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
		if not self.undirected:
			self.v = rng.random_sample(self.v.shape)
			row_sums = self.v.sum(axis=1)
			self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
		else:
			self.v = self.u

	def _initialize_w(self, w0):
		"""
			Initialize affinity tensor w from file.

			Parameters
			----------
			infile_name : str
						  Path of the input file.
		"""  

		# if self.assortative:
		# 	self.w = w0.copy()
		# else:
		# 	self.w = (w0.values[np.newaxis, :, :]).copy()
		
		# if self.fix_w == False:
		# 	max_entry = np.max(self.w)
		# 	self.w += max_entry * self.err * np.random.random_sample(self.w.shape)

		ww0 = w0.copy() 
		for i in range(self.L):
			for k in range(self.K):
				if self.assortative:
					self.w[i, k] = ww0[i, k] 

	def _update_old_variables(self):
		"""
			Update values of the parameters in the previous iteration.
		"""

		self.u_old[self.u > 0] = np.copy(self.u[self.u > 0])
		self.v_old[self.v > 0] = np.copy(self.v[self.v > 0])
		self.w_old[self.w > 0] = np.copy(self.w[self.w > 0])
		self.eta_old = np.copy(self.eta)
		self.pibr_old = np.copy(self.pibr)
		self.mupr_old = np.copy(self.mupr)

	# @gl.timeit('cache')
	def _update_cache(self, data, data_T_vals, subs_nz,update_Q=True):
		"""
			Update the cache used in the em_update.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T_vals : ndarray
						  Array with values of entries A[j, i] given non-zero entry (i, j).
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
		"""
		self.lambda_aij = self._lambda0_full()  # full matrix lambda
		if not self.use_approximation:
			self.lambdalambdaT = np.einsum('aij,aji->aij', self.lambda_aij, self.lambda_aij)  # to use in Z and eta
			self.Z = self._calculate_Z()   

		self.lambda0_nz = self._lambda0_nz(subs_nz, self.u, self.v, self.w) # matrix lambda for non-zero entries
		if self.assortative == False:
			self.lambda0_nzT = self._lambda0_nz(subs_nz, self.v, self.u, np.einsum('akq->aqk',self.w))
		else:
			self.lambda0_nzT = self._lambda0_nz(subs_nz, self.v, self.u,self.w)
		if self.flag_anomaly == True:
			if update_Q == True:
				self.Qij_dense,self.Qij_nz = self._QIJ(data, data_T_vals, subs_nz)  
			assert np.allclose(self.Qij_dense[0], self.Qij_dense[0].T, rtol=1e-05, atol=1e-08)
		self.M_nz = self.lambda0_nz 
		self.M_nz[self.M_nz == 0] = 1 

		if isinstance(data, skt.dtensor):
			# To Do: complete this part
			self.data_M_nz = data[subs_nz] / self.M_nz 
		elif isinstance(data, skt.sptensor):
			self.data_M_nz   = data.vals / self.M_nz
			if self.flag_anomaly == True:
				self.data_M_nz_Q = data.vals * (1-self.Qij_nz) / self.M_nz 
				self.eta_num_Q = (data.vals * (1-self.Qij_nz) * data_T_vals).sum()  
			else:
				self.data_M_nz_Q = data.vals / self.M_nz 
		

		self.den_updates = 1 + self.eta * self.lambda_aij  # to use in the updates


	def _calculate_Z(self):
		"""
			Compute the normalization constant of the Bivariate Bernoulli distribution.

			Returns
			-------
			Z : ndarray
				Normalization constant Z of the Bivariate Bernoulli distribution.
		""" 
 
		Z = self.lambda_aij + transpose_tensor(self.lambda_aij) + self.eta * self.lambdalambdaT + 1.
		for l in range(len(Z)):
			assert check_symmetric(Z[l])

		return Z
	
	def _calculate_Za(self):
		"""
			Compute the normalization constant of the Bivariate Bernoulli distribution.

			Returns
			-------
			Z : ndarray
				Normalization constant Z of the Bivariate Bernoulli distribution.
		""" 
		Za = (self.pibr + 1) * (self.pibr + 1)

		return Za

	# @gl.timeit('QIJ')
	def _QIJ(self, data, data_T_vals, subs_nz):
		"""
			Compute the mean lambda0_ij for only non-zero entries.

			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
			u : ndarray
				Out-going membership matrix.
			v : ndarray
				In-coming membership matrix.
			w : ndarray
				Affinity tensor.

			Returns
			-------
			nz_recon_I : ndarray
						 Mean lambda0_ij for only non-zero entries. 
		"""  

		if isinstance(data, skt.dtensor):
			# To Do: complete this part
			nz_recon_I =  np.power(self.pibr,data[subs_nz])
		elif isinstance(data, skt.sptensor):
			A_tot = data.vals + data_T_vals  
			if self.pibr > 0: 
				nz_recon_I =  np.sqrt(self.mupr) * np.exp( (data.vals+data_T_vals) * np.log(self.pibr)) / (self.pibr+1)/ (self.pibr+1)
				Z_vals = get_item_array_from_subs(self._calculate_Z(), subs_nz) 
				nz_recon_Id = nz_recon_I + np.sqrt((1-self.mupr)) * np.exp(data.vals * self.lambda0_nz 
				+ data_T_vals * self.lambda0_nzT + 0.5 * np.log(self.eta) * data.vals * data_T_vals+EPS) / (Z_vals) 
				non_zeros = nz_recon_Id > 0 
				nz_recon_I[non_zeros] /=  nz_recon_Id[non_zeros]   
			else:  
				print(self.pibr)
				raise ValueError('The value of pi  has to be in [0, 1]!')
		
		Q_ij_dense = np.ones(self.lambda_aij.shape)  
		Q_ij_dense *=  np.sqrt(self.mupr) * np.sqrt(self._calculate_Z())   
		Q_ij_dense_d = Q_ij_dense + np.sqrt((1-self.mupr)) * np.sqrt(self._calculate_Za())
		non_zeros = Q_ij_dense_d > 0
		Q_ij_dense[non_zeros] /= Q_ij_dense_d[non_zeros] 
		 
		Q_ij_dense[subs_nz] = nz_recon_I  

		Q_ij_dense = np.maximum( Q_ij_dense, transpose_tensor(Q_ij_dense)) # make it symmetriv
		np.fill_diagonal(Q_ij_dense[0], 0.) 

		assert (Q_ij_dense > 1).sum() == 0
		return Q_ij_dense, Q_ij_dense[subs_nz]

	def _lambda0_nz(self, subs_nz, u, v, w):
		"""
			Compute the mean lambda0_ij for only non-zero entries.

			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
			u : ndarray
				Out-going membership matrix.
			v : ndarray
				In-coming membership matrix.
			w : ndarray
				Affinity tensor.

			Returns
			-------
			nz_recon_I : ndarray
						 Mean lambda0_ij for only non-zero entries.
		"""

		if not self.assortative:
			nz_recon_IQ = np.einsum('Ik,Ikq->Iq', u[subs_nz[1], :], w[subs_nz[0], :, :])
		else:
			nz_recon_IQ = np.einsum('Ik,Ik->Ik', u[subs_nz[1], :], w[subs_nz[0], :])
		nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ, v[subs_nz[2], :])

		return nz_recon_I
	
	# @gl.timeit('em')
	def _update_em(self, data, data_T_vals, subs_nz, denominator=None,mask=None,subs_nz_mask=None):
		"""
			Update parameters via EM procedure.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T_vals : ndarray
						  Array with values of entries A[j, i] given non-zero entry (i, j).
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
			denominator : float
						  Denominator used in the update of the eta parameter.

			Returns
			-------
			d_u : float
				  Maximum distance between the old and the new membership matrix u.
			d_v : float
				  Maximum distance between the old and the new membership matrix v.
			d_w : float
				  Maximum distance between the old and the new affinity tensor w.
			d_eta : float
					Maximum distance between the old and the new reciprocity coefficient eta.
		"""

		if not self.fix_communities:
			if self.use_approximation:
				d_u = self._update_U_approx(subs_nz,mask=mask,subs_nz_mask=subs_nz_mask)
			else:
				d_u = self._update_U(subs_nz,mask=mask,subs_nz_mask=subs_nz_mask)
			self._update_cache(data, data_T_vals, subs_nz)
		else:
			d_u = 0.
		

		if self.undirected:
			self.v = self.u
			self.v_old = self.v
			d_v = d_u
			self._update_cache(data, data_T_vals, subs_nz)
		else: 
			if not self.fix_communities:
				if self.use_approximation:
					d_v = self._update_V_approx(subs_nz,mask=mask,subs_nz_mask=subs_nz_mask)
				else:
					d_v = self._update_V(subs_nz,mask=mask,subs_nz_mask=subs_nz_mask)
				self._update_cache(data, data_T_vals, subs_nz)
			else:
				d_v = 0.

		if not self.fix_w:
			if not self.assortative:
				if self.use_approximation:
					d_w = self._update_W_approx(subs_nz,mask=mask,subs_nz_mask=subs_nz_mask)
				else:
					d_w = self._update_W(subs_nz,mask=mask,subs_nz_mask=subs_nz_mask)
			else:
				if self.use_approximation:
					d_w = self._update_W_assortative_approx(subs_nz,mask=mask,subs_nz_mask=subs_nz_mask)
				else: 
					d_w = self._update_W_assortative(subs_nz,mask=mask,subs_nz_mask=subs_nz_mask)
			self._update_cache(data, data_T_vals, subs_nz)
		else:
			d_w = 0.

		if not self.fix_eta: 
			if self.use_approximation:
				d_eta = self._update_eta_approx()
			else:
				d_eta = self._update_eta(mask=mask)
			self._update_cache(data, data_T_vals, subs_nz) 
		else:
			d_eta = 0.
		 


		if not self.fix_pibr: 
			# s = time.time()
			d_pibr = self._update_pibr(data, data_T_vals, subs_nz, denominator=denominator,mask=mask,subs_nz_mask=subs_nz_mask)
			# e = time.time()
			# print('pi',e-s)
			self._update_cache(data, data_T_vals, subs_nz)

		else:
			d_pibr = 0.
		
		if not self.fix_mupr:
			# s = time.time()
			d_mupr = self._update_mupr(data, data_T_vals, subs_nz, denominator=denominator,mask=mask,subs_nz_mask=subs_nz_mask)
			# e = time.time()
			# print('mu',e-s)
			self._update_cache(data, data_T_vals, subs_nz)
		else:
			d_mupr = 0.

		return d_u, d_v, d_w, d_eta, d_pibr, d_mupr


	def enforce_constraintU(self,num,den, mult = 1):

		lambda_test = root(func_lagrange_multiplier, 1.0 ,args=(num,den,mult))  
		lambda_i  = lambda_test.x
		# if lambda_test.success == False:
		# 	print(num,den,lambda_i)

		return lambda_i,lambda_test.success

	# @gl.timeit('U')
	def _update_U(self, subs_nz,mask=None,subs_nz_mask=None):
		"""
			Update out-going membership matrix.

			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.

			Returns
			-------
			dist_u : float
					 Maximum distance between the old and the new membership matrix u.
		"""     

		if not self.assortative:
			VW = np.einsum('jq,akq->ajk', self.v, self.w)
		else:
			VW = np.einsum('jk,ak->ajk', self.v, self.w) 
		
		if self.flag_anomaly == True:  
			if mask is None:
				Z_uk = np.einsum('aij,ajk->aijk', (1-self.Qij_dense),VW)
			else:
				Z_uk = np.einsum('aij,ajk->aijk', mask * (1-self.Qij_dense),VW)
			VWL = np.einsum('aji,aijk->aijk', self.den_updates, Z_uk)
		
		else: # flag_anomaly == False 
			VWL = np.einsum('aji,ajk->aijk', self.den_updates, VW)
		
		den = np.einsum('aijk,aij->ik', VWL, 1. / self.Z)

	
		if not self.constrained: 
			self.u = self.ag - 1 + self.u_old * self._update_membership_Q(subs_nz, self.u, self.v, self.w, 1)   
			den += self.bg 

			non_zeros = den > 0.  

			self.u[den == 0] = 0.
			self.u[non_zeros] /=  den[non_zeros] 
		
		else: # self.constrained     TO-DO: update this part
			u_tmp = self.u_old * self._update_membership_Q(subs_nz, self.u, self.v, self.w, 1) 
			u_tmp[Z_uk == 0] = 0.
			Z_uk[Z_uk == 0] = 1.
			low_values_indices = (u_tmp / Z_uk) < self.err_max
			u_tmp[low_values_indices] = 0

			lambda_vec = np.zeros(self.u.shape[0])
			success = np.ones(self.u.shape[0]).astype('bool')
			for i in range(self.u.shape[0]):
				if np.sum(u_tmp[i]) > self.err_max:
					if self.flag_anomaly == False:
						lambda_vec[i],success[i] = self.enforce_constraintU(u_tmp[i] , Z_uk)
					else:
						lambda_vec[i],success[i] = self.enforce_constraintU(u_tmp[i] , Z_uk[i],(self.N-self.Qij_dense[0][i].sum()) )

			
			self.u = abs(u_tmp / (lambda_vec[:,np.newaxis] + Z_uk))
			self.u[np.logical_not(success)] = np.copy(self.u_old[np.logical_not(success)])

			# self.u = self.u_old * self._update_membership_Q(subs_nz, self.u, self.v, self.w, 1) 
			# row_sums = self.u.sum(axis=1)
			# self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

		low_values_indices = self.u < self.err_max  # values are too low
		self.u[low_values_indices] = 0.  # and set to 0.

		dist_u = np.amax(abs(self.u - self.u_old))
		self.u_old = np.copy(self.u)

		return dist_u
	

	def _update_V(self, subs_nz,mask=None,subs_nz_mask=None):
		"""
			Update out-going membership matrix.

			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.

			Returns
			-------
			dist_v : float
					 Maximum distance between the old and the new membership matrix u.
		"""     

		if not self.assortative: 
			UW = np.einsum('jq,aqk->ajk', self.u, self.w)
		else: 
			UW = np.einsum('jk,ak->ajk', self.u, self.w)
 

		if self.flag_anomaly == True:  
			if mask is None:
				Z_vk = np.einsum('aji,ajk->aijk', (1-self.Qij_dense),UW)
			else:
				Z_vk = np.einsum('aji,ajk->aijk', mask * (1-self.Qij_dense),UW)
			VWL = np.einsum('aij,aijk->aijk', self.den_updates, Z_vk) 
		
		else: # flag_anomaly == False 
			VWL = np.einsum('aij,ajk->aijk', self.den_updates, UW)
		
		den = np.einsum('aijk,aij->ik', VWL, 1. / self.Z)

	
		if not self.constrained: 
			self.v = self.ag - 1 + self.v_old * self._update_membership_Q(subs_nz, self.u, self.v, self.w, 2)   
			den += self.bg 

			non_zeros = den > 0.  

			self.v[den == 0] = 0.
			self.v[non_zeros] /=  den[non_zeros]  
		else: # self.constrained   TO-DO: update this part
			v_tmp = self.v_old * self._update_membership_Q(subs_nz, self.u, self.v, self.w, 2) 
			v_tmp[Z_vk == 0] = 0.
			Z_vk[Z_vk == 0] = 1.
			low_values_indices = (v_tmp / Z_vk) < self.err_max
			v_tmp[low_values_indices] = 0

			lambda_vec = np.zeros(self.v.shape[0])
			success = np.ones(self.v.shape[0]).astype('bool')
			for i in range(self.uvshape[0]):
				if np.sum(v_tmp[i]) > self.err_max:
					if self.flag_anomaly == False:
						lambda_vec[i],success[i] = self.enforce_constraintU(v_tmp[i] , Z_vk)
					else:
						lambda_vec[i],success[i] = self.enforce_constraintU(v_tmp[i] , Z_vk[i],(self.N-self.Qij_dense[0][i].sum()) )

			
			self.v = abs(v_tmp / (lambda_vec[:,np.newaxis] + Z_vk))
			self.v[np.logical_not(success)] = np.copy(self.v_old[np.logical_not(success)])

			# self.u = self.u_old * self._update_membership_Q(subs_nz, self.u, self.v, self.w, 1) 
			# row_sums = self.u.sum(axis=1)
			# self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

		low_values_indices = self.v < self.err_max  # values are too low
		self.v[low_values_indices] = 0.  # and set to 0.

		dist_v = np.amax(abs(self.v - self.v_old))
		self.v_old = np.copy(self.v)

		return dist_v


	# @gl.timeit('W_assortative')
	def _update_W_assortative(self, subs_nz,mask=None,subs_nz_mask=None):
		"""
			Update affinity tensor (assuming assortativity).

			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.

			Returns
			-------
			dist_w : float
					 Maximum distance between the old and the new affinity tensor w.
		"""
		# Let's make some changes!

		uttkrp_DKQ = np.zeros_like(self.w)

		UV = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
		uttkrp_I = self.data_M_nz_Q[:, np.newaxis] * UV
		for k in range(self.K):
			uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L)

		self.w = self.w * uttkrp_DKQ
		
		if self.flag_anomaly == True:
			if mask is None:
				UQk = np.einsum('aij,ik->aijk', (1-self.Qij_dense), self.u)
				# Zk = np.einsum('aijk,jk->aijk', UQk, self.v)
				Zk = np.einsum('jk, aijk->aijk', self.v, UQk)
			else:
				Zk = np.einsum('aij,ijk->aijk',mask * (1-self.Qij_dense),np.einsum('ik,jk->ijk',self.u,self.v))
			VWL = np.einsum('aji,aijk->aijk', self.den_updates, Zk)
		else: # flag_anomaly == False
			UL = np.einsum('ik,aji->aijk', self.u, self.den_updates)
			VWL = np.einsum('jk,aijk->aijk', self.v, UL)
		den = np.einsum('aijk,aij->ak', VWL, 1. / self.Z)

			# Zk = np.einsum('ik,jk->ijk', self.u, self.v)
			# Zk = Zk[np.newaxis,:]
			# VWL = np.einsum('aij,ijk->aijk', self.den_updates, Zk)
		# den = np.einsum('aijk,aij->ak', VWL, 1. / self.Z)
		# den += self.bg

		non_zeros = den > 0
		self.w[den == 0] = 0
		self.w[non_zeros] /= den[non_zeros]

		low_values_indices = self.w < self.err_max  # values are too low
		self.w[low_values_indices] = 0.  # and set to 0.

		dist_w = np.amax(abs(self.w - self.w_old))
		self.w_old = np.copy(self.w)  
		  

		return dist_w 


	def eta_fix_point(self,mask=None):  
		if self.flag_anomaly == True:
			if mask is None:  
				lambdalambdaT_Q = np.einsum('aij,aij->aij', (1-self.Qij_dense),self.lambdalambdaT) 
			else:
				lambdalambdaT_Q = np.einsum('aij,aij->aij', mask * (1-self.Qij_dense),self.lambdalambdaT)
			st = (lambdalambdaT_Q / self.Z).sum()  
			# print(self.lambdalambdaT.sum(), self.u.sum(), self.w.sum())  
			
			if st > 0:
				return self.eta_num_Q / st
			else: 
				raise ValueError('eta fix point has zero denominator!')
		else:  
			st = (self.lambdalambdaT / self.Z).sum()  
			if st > 0: 
				return self.AAtSum / st
			else: 
				raise ValueError('eta fix point has zero denominator!') 
	

	def eta_root(self,mask=None): 
		x0 = self.eta_fix_point(mask=mask) 
		if self.flag_anomaly == True:
			if mask is None:  
				lambdalambdaT_Q = np.einsum('aij,aij->aij', (1-self.Qij_dense),self.lambdalambdaT) 
			else:
				lambdalambdaT_Q = np.einsum('aij,aij->aij', mask * (1-self.Qij_dense),self.lambdalambdaT)
			# eta_new = brentq(func_eta_den, -10, 1000, args = (self.eta_num_Q, lambdalambdaT_Q,self.lambdalambdaT, self.lambda_aij, transpose_tensor(self.lambda_aij)))
			eta_new = root(func_eta_den, 200, args=(self.eta_num_Q, lambdalambdaT_Q,self.lambdalambdaT, self.lambda_aij, transpose_tensor(self.lambda_aij)))  
			# eta_new = fixed_point(func_eta_den, self.eta,args = (self.eta_num_Q, lambdalambdaT_Q,self.lambdalambdaT, self.lambda_aij, transpose_tensor(self.lambda_aij)) )
			# print(eta_new.x, x0) 
			return eta_new.x
		else:
			eta_new = root(func_eta_den, 1000 ,args=(self.AAtSum, self.lambdalambdaT,self.lambdalambdaT, self.lambda_aij, transpose_tensor(self.lambda_aij)))  
			# print(eta_new.x, x0)
			return eta_new.x


	def _update_eta(self,mask=None):
		"""
			Update pair interaction coefficient eta.

			Returns
			-------
			dist_eta : float
					   Maximum distance between the old and the new pair interaction coefficient eta.
		"""

		self.eta = self.eta_fix_point(mask=mask)  
		# self.eta = self.eta_root(mask=mask) 
		# print('eta:', self.eta, self.eta_root(mask=mask) )

		if self.eta < self.err_max:  # value is too low
			self.eta = 0.  # and set to 0.

		dist_eta = abs(self.eta - self.eta_old)
		self.eta_old = np.copy(self.eta)

		return dist_eta

	# @gl.timeit('pibr')
	def _update_pibr(self, data, data_T_vals, subs_nz, denominator=None,mask=None,subs_nz_mask=None):
		"""
			Update reciprocity coefficient eta.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T_vals : ndarray
						  Array with values of entries A[j, i] given non-zero entry (i, j).
			denominator : float
						  Denominator used in the update of the eta parameter.

			Returns
			-------
			dist_eta : float
					   Maximum distance between the old and the new reciprocity coefficient eta.
		"""   
		if isinstance(data, skt.dtensor):
			Adata = (data[subs_nz] * self.Qij_nz).sum()
		elif isinstance(data, skt.sptensor):
			Adata   = (data.vals * self.Qij_nz).sum()

		if mask is None:    
			self.pibr = Adata / (self.Qij_dense.sum() - Adata)
		else:
			self.pibr = Adata / (self.Qij_dense[subs_nz_mask].sum()-Adata)
 
		dist_pibr = abs(self.pibr - self.pibr_old) 
		self.pibr_old = np.copy(self.pibr) 

		return dist_pibr

	
	def _update_pibr_2step(self, data, data_T_vals, subs_nz, denominator=None,mask=None,subs_nz_mask=None):
		"""
			Update reciprocity coefficient eta.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T_vals : ndarray
						  Array with values of entries A[j, i] given non-zero entry (i, j).
			denominator : float
						  Denominator used in the update of the eta parameter.

			Returns
			-------
			dist_eta : float
					   Maximum distance between the old and the new reciprocity coefficient eta.
		"""   
		if isinstance(data, skt.dtensor):
			Adata = (data[subs_nz] * self.Qij_nz).sum()
		elif isinstance(data, skt.sptensor):
			Adata   = (data.vals * self.Qij_nz).sum()

		if mask is None:    
			self.pibr = Adata / (self.Qij_dense.sum() - Adata)
		else:
			self.pibr = Adata / (self.Qij_dense[subs_nz_mask].sum()-Adata)
 
		dist_pibr =  self.pibr - self.pibr_old  
		self.pibr_old = np.copy(self.pibr) 

		return dist_pibr


	# @gl.timeit('mupr')
	def _update_mupr(self, data, data_T_vals, subs_nz, denominator=None,mask=None,subs_nz_mask=None):
		"""
			Update reciprocity coefficient eta.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T_vals : ndarray
						  Array with values of entries A[j, i] given non-zero entry (i, j).
			denominator : float
						  Denominator used in the update of the eta parameter.

			Returns
			-------
			dist_eta : float
					   Maximum distance between the old and the new reciprocity coefficient eta.
		"""
		if mask is None:
			self.mupr = self.Qij_dense.sum() / ( self.N * (self.N-1) )
		else:
			self.mupr = self.Qij_dense[subs_nz_mask].sum() / ( self.N * (self.N-1) ) 
		
		if self.mupr < 0: 
			raise ValueError('The value of mu  has to be in [0, 1]!')

		dist_mupr = abs(self.mupr - self.mupr_old)
		self.mupr_old = np.copy(self.mupr) 

		return dist_mupr 

	
	def _update_mupr_2step(self, data, data_T_vals, subs_nz, denominator=None,mask=None,subs_nz_mask=None):
		"""
			Update reciprocity coefficient eta.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T_vals : ndarray
						  Array with values of entries A[j, i] given non-zero entry (i, j).
			denominator : float
						  Denominator used in the update of the eta parameter.

			Returns
			-------
			dist_eta : float
					   Maximum distance between the old and the new reciprocity coefficient eta.
		"""
		if mask is None:
			self.mupr = self.Qij_dense.sum() / ( self.N * (self.N-1) )
		else:
			self.mupr = self.Qij_dense[subs_nz_mask].sum() / ( self.N * (self.N-1) ) 
		

		dist_mupr = self.mupr - self.mupr_old
		self.mupr_old = np.copy(self.mupr) 

		return dist_mupr 
	
	

	def _update_membership_Q(self, subs_nz, u, v, w, m):
		"""
			Return the Khatri-Rao product (sparse version) used in the update of the membership matrices.

			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
			u : ndarray
				Out-going membership matrix.
			v : ndarray
				In-coming membership matrix.
			w : ndarray
				Affinity tensor.
			m : int
				Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
				works with the matrix u; if 2 it works with v.

			Returns
			-------
			uttkrp_DK : ndarray
						Matrix which is the result of the matrix product of the unfolding of the tensor and the
						Khatri-Rao product of the membership matrix.
		"""

		if not self.assortative:
			uttkrp_DK = sp_uttkrp(self.data_M_nz_Q, subs_nz, m, u, v, w)
		else:
			uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz_Q, subs_nz, m, u, v, w)

		return uttkrp_DK

	# @gl.timeit('convergence')
	def _check_for_convergence(self, data,data_T, it, loglik, coincide, convergence, mask=None,subs_nz_mask=None,mod_it = 10):
		"""
			Check for convergence by using the pseudo log-likelihood values.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			it : int
				 Number of iteration.
			loglik : float
					 Pseudo log-likelihood value.
			coincide : int
					   Number of time the update of the pseudo log-likelihood respects the tolerance.
			convergence : bool
						  Flag for convergence.
			data_T : sptensor/dtensor
					 Graph adjacency tensor (transpose).
			mask : ndarray
				   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

			Returns
			-------
			it : int
				 Number of iteration.
			loglik : float
					 Log-likelihood value.
			coincide : int
					   Number of time the update of the pseudo log-likelihood respects the tolerance.
			convergence : bool
						  Flag for convergence.
		"""
		success = True
		

		if it % mod_it == 0:
			old_L = loglik
			loglik = self._ELBO(data, data_T=data_T, mask=mask,subs_nz_mask=subs_nz_mask)

			if abs(loglik - old_L) < self.tolerance:
				coincide += 1
			else:
				coincide = 0
			if loglik - old_L < 0:
				success = False
			
		if coincide > self.decision:
			convergence = True
		it += 1

		return it, loglik, coincide, convergence, success
	

	def _check_for_convergencepi(self, data,data_T, it, loglik, coincide, convergence, mask=None,subs_nz_mask=None,mod_it = 10):
		"""
			Check for convergence by using the pseudo log-likelihood values.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			it : int
				 Number of iteration.
			loglik : float
					 Pseudo log-likelihood value.
			coincide : int
					   Number of time the update of the pseudo log-likelihood respects the tolerance.
			convergence : bool
						  Flag for convergence.
			data_T : sptensor/dtensor
					 Graph adjacency tensor (transpose).
			mask : ndarray
				   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

			Returns
			-------
			it : int
				 Number of iteration.
			loglik : float
					 Log-likelihood value.
			coincide : int
					   Number of time the update of the pseudo log-likelihood respects the tolerance.
			convergence : bool
						  Flag for convergence.
		"""
		success = True 

		if it % mod_it == 0:
			old_L = loglik
			loglik = self._ELBO(data, data_T=data_T, mask=mask,subs_nz_mask=subs_nz_mask)

			if abs(loglik - old_L) < self.tolerance * 5:
				coincide += 1
			else:
				coincide = 0
			if loglik - old_L < 0:
				success = False
			
		if coincide > self.decision:
			convergence = True
		it += 1

		return it, loglik, coincide, convergence, success


	def _Likelihood(self, data):
		"""
			Compute the log-likelihood of the data.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.

			Returns
			-------
			l : float
				Log-likelihood value.
		"""

		# self.lambdalambdaT = np.einsum('aij,aji->aij', self.lambda_aij, self.lambda_aij)  # to use in Z and eta 
		# self.Z = self._calculate_Z()

		ft = (data.vals * np.log(self.lambda0_nz)).sum()
		# if self.eta > 0:
		# 	st = 0.5 * np.log(self.eta) * self.AAtSum
		# else:
		# 	st = 0
		st = 0.5 * np.log(self.eta) * self.AAtSum

		tt = 0.5 * np.log(self.Z).sum()

		l = ft + st - tt
		

		if np.isnan(l):
			raise ValueError('log-likelihood is NaN!')
		else:
			return l
			
	# @gl.timeit('ELBO')
	def _ELBO(self, data, data_T, mask=None,subs_nz_mask=None):
		"""
			Compute the  ELBO of the data.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T : sptensor/dtensor
					 Graph adjacency tensor (transpose).
			mask : ndarray
				   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

			Returns
			-------
			l : float
				Pseudo ELBO value.
		""" 
		# self.lambda0_ija = self._lambda0_full() 
		

		if mask is not None: 
			subs_nz_mask = mask>0 
			Adense = data.toarray()
			ATdense = data_T.toarray()

		if self.flag_anomaly == False:
			if mask is  None:
				l = self._Likelihood(data)
			else:
				l = (data.vals * np.log(self.lambda_aij[data.subs]+EPS)).sum()  
				# lambdalambdaT_mask = np.einsum('aij,aji->aij', self.lambda_aij[subs_nz_mask], self.lambda_aij[subs_nz_mask])
				lambdalambdaT_mask = np.einsum('aij,aji->aij', self.lambda_aij, self.lambda_aij)
				l -= 0.5 * (np.log(self.lambda_aij[subs_nz_mask] + transpose_tensor(self.lambda_aij)[subs_nz_mask] + self.eta * lambdalambdaT_mask[subs_nz_mask] + 1 + EPS)).sum()
				if self.eta > 0 :
					if isinstance(data, skt.dtensor):
						l += (data.vals * data_T[subs_nz_mask] * np.log(self.eta + EPS)).sum()
					elif isinstance(data, skt.sptensor):
						l += (Adense[subs_nz_mask] * ATdense[subs_nz_mask] * np.log(self.eta + EPS)).sum()
			
		else:
			l = 0.
			if mask is None:
				l += ((1-self.Qij_dense) * self._Likelihood(data)).sum() # (1-Q) * (f*A + 0.5*J*A*AT - 0.5*log Z)  
			else:# (1-Q) * (f*A + 0.5*J*A*AT - 0.5*log Z) 
				non_zeros0 = np.logical_and( mask > 0,Adense > 0 ) 
				l += ( ((1-self.Qij_dense)[non_zeros0]) * Adense[non_zeros0] * np.log(self.lambda_aij[non_zeros0]+EPS)).sum() # (1-Q) * (f*A ) 
				if self.eta > 0:
					non_zeros1 = np.logical_and( non_zeros0 > 0,ATdense > 0 )
					# TO Do: Check this 
					# l += 0.5 * ((1-self.Qij_dense)[non_zeros1] * data.vals * data_T.vals).sum()  * np.log(self.eta+EPS) # (1-Q) * (0.5*J*A*AT)  
					l += 0.5 * ((1-self.Qij_dense)[non_zeros1] * Adense[non_zeros1] * ATdense[non_zeros1]).sum()  * np.log(self.eta+EPS) # (1-Q) * (0.5*J*A*AT)
				# To Do: Check this
				# lambdalambdaT_mask = np.einsum('aij,aji->aij', self.lambda_aij[subs_nz_mask], self.lambda_aij[subs_nz_mask])
				lambdalambdaT_mask = np.einsum('aij,aji->aij', self.lambda_aij, self.lambda_aij)
				# To Do this  transpose_tensor(self.lambda_aij)[subs_nz_mask] 
				l -= 0.5 * ((1-self.Qij_dense)[subs_nz_mask] * np.log(self.lambda_aij[subs_nz_mask] + transpose_tensor(self.lambda_aij)[subs_nz_mask] + self.eta * lambdalambdaT_mask[subs_nz_mask] + 1 + EPS)).sum() # (1-Q) * (- 0.5*log Z) 


			if mask is None: # Q * (f_a*A - 0.5*log Z_a)  
				if self.pibr > 0:
					l += np.log(self.pibr) * (self.Qij_dense[data.subs] * data.vals).sum()  # Q * (f_a * A) 
				l -= 0.5 * np.log(self._calculate_Za()) * (self.Qij_dense).sum() # Q * (- 0.5 * log Z_a)
			else:
				if self.pibr > 0: 
					non_zeros2 = np.logical_and(mask > 0,Adense > 0)
					l += (self.Qij_dense[non_zeros2] * Adense[non_zeros2]).sum() * np.log(self.pibr)  # Q * (f_a * A) 
				l -= 0.5 * self.Qij_dense[subs_nz_mask].sum() * self._calculate_Za()  # Q * (- 0.5 * log Z_a)
			
			if mask is None:
				non_zeros = self.Qij_dense > 0
				non_zeros3 = (1-self.Qij_dense) > 0
			else:
				non_zeros = np.logical_and( mask > 0,self.Qij_dense > 0 )
				non_zeros3 = np.logical_and( mask > 0, (1-self.Qij_dense ) > 0 )

			l -= 0.5 * (self.Qij_dense[non_zeros] * np.log(self.Qij_dense[non_zeros]+EPS)).sum() # - sum [Q * Log Q]
			l -= 0.5 * ((1-self.Qij_dense)[non_zeros3] * np.log((1-self.Qij_dense)[non_zeros3]+EPS)).sum() # - sum [(1-Q) * Log (1-Q)]

			if mask is None: 
				if (1 - self.mupr) > 0 :
					l += 0.5 * np.log(1-self.mupr+EPS) * (1-self.Qij_dense).sum() # sum [(1-Q) * Log (1-mu)]
				if self.mupr > 0 :
					l += 0.5 * np.log(self.mupr+EPS) * (self.Qij_dense).sum() #  sum [Q * Log mu]
			else:  
				if (1 - self.mupr) > 0 :
					l += 0.5 * np.log(1-self.mupr+EPS) * ((1-self.Qij_dense)[subs_nz_mask]).sum() #  sum [(1-Q) * Log (1-mu)]
				if self.mupr > 0 :
					l += 0.5 * np.log(self.mupr+EPS) * (self.Qij_dense[subs_nz_mask]).sum() #   sum [Q * Log mu]

		if self.ag > 1.:
			l += (self.ag -1) * np.log(self.u+EPS).sum()
			l += (self.ag -1) * np.log(self.v+EPS).sum()
		if self.bg >  0. :
			l -= self.bg * self.u.sum()
			l -= self.bg * self.v.sum() 

		if np.isnan(l):
			print("ELBO is NaN!!!!")
			sys.exit(1)
		else:
			return l


	def _lambda0_full(self):
		"""
			Compute the mean lambda0 for all entries.

			Parameters
			----------
			u : ndarray
				Out-going membership matrix.
			v : ndarray
				In-coming membership matrix.
			w : ndarray
				Affinity tensor.

			Returns
			-------
			M : ndarray
				Mean lambda0 for all entries.
		"""

		if self.w.ndim == 2:
			M = np.einsum('ik,jk->ijk', self.u, self.v)
			M = np.einsum('ijk,ak->aij', M, self.w)
		else:
			M = np.einsum('ik,jq->ijkq', self.u, self.v)
			M = np.einsum('ijkq,akq->aij', M, self.w)
		return M

	def _update_optimal_parameters(self):
		"""
			Update values of the parameters after convergence.
		"""

		self.u_f = np.copy(self.u)
		self.v_f = np.copy(self.v)
		self.w_f = np.copy(self.w)
		self.eta_f = np.copy(self.eta)
		self.pibr_f = np.copy(self.pibr)
		self.mupr_f = np.copy(self.mupr)
		if self.flag_anomaly == True:
			self.Q_ij_dense_f = np.copy(self.Qij_dense)
		else:
			self.Q_ij_dense_f = np.zeros((1,self.N,self.N))

	def output_results(self, nodes):
		"""
			Output results.

			Parameters
			----------
			nodes : list
					List of nodes IDs.
		"""

		outfile = self.out_folder + 'theta_inf_' + str(self.flag_anomaly) +'_'+ self.end_file
		np.savez_compressed(outfile + '.npz', u=self.u_f, v=self.v_f, w=self.w_f, eta=self.eta_f, pibr=self.pibr_f, mupr=self.mupr_f, max_it=self.final_it,
				Q = self.Q_ij_dense_f, maxL=self.maxL, nodes=nodes)
		print(f'\nInferred parameters saved in: {outfile + ".npz"}')
		print('To load: theta=np.load(filename), then e.g. theta["u"]')


def sp_uttkrp(vals, subs, m, u, v, w):
	"""
		Compute the Khatri-Rao product (sparse version).

		Parameters
		----------
		vals : ndarray
			   Values of the non-zero entries.
		subs : tuple
			   Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
			   equal to the dimension of tensor.
		m : int
			Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
			works with the matrix u; if 2 it works with v.
		u : ndarray
			Out-going membership matrix.
		v : ndarray
			In-coming membership matrix.
		w : ndarray
			Affinity tensor.

		Returns
		-------
		out : ndarray
			  Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
			  of the membership matrix.
	"""

	if m == 1:
		D, K = u.shape
		out = np.zeros_like(u)
	elif m == 2:
		D, K = v.shape
		out = np.zeros_like(v)

	for k in range(K):
		tmp = vals.copy()
		if m == 1:  # we are updating u
			tmp *= (w[subs[0], k, :].astype(tmp.dtype) * v[subs[2], :].astype(tmp.dtype)).sum(axis=1)
		elif m == 2:  # we are updating v
			tmp *= (w[subs[0], :, k].astype(tmp.dtype) * u[subs[1], :].astype(tmp.dtype)).sum(axis=1)
		out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

	return out


def func_lagrange_multiplier(lambda_i, num,den,mult=1):
	f = num / ( lambda_i * mult + den )   
	if (f < 0).sum() > 0:
		return 1
	else:
		return np.sum(f) - 1

def func_eta_den(lambda_i, num,den,lb, a,b):
	f = lambda_i -  num / ((den/(a+b+lambda_i*lb+1)).sum())   
	return f


def sp_uttkrp_assortative(vals, subs, m, u, v, w):
	"""
		Compute the Khatri-Rao product (sparse version) with the assumption of assortativity.

		Parameters
		----------
		vals : ndarray
			   Values of the non-zero entries.
		subs : tuple
			   Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
			   equal to the dimension of tensor.
		m : int
			Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
			works with the matrix u; if 2 it works with v.
		u : ndarray
			Out-going membership matrix.
		v : ndarray
			In-coming membership matrix.
		w : ndarray
			Affinity tensor.

		Returns
		-------
		out : ndarray
			  Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
			  of the membership matrix.
	"""

	if m == 1:
		D, K = u.shape
		out = np.zeros_like(u) 
	elif m == 2:
		D, K = v.shape
		out = np.zeros_like(v)

	for k in range(K):
		tmp = vals.copy()
		if m == 1:  # we are updating u
			tmp *= w[subs[0], k].astype(tmp.dtype) * v[subs[2], k].astype(tmp.dtype) 
		elif m == 2:  # we are updating v
			tmp *= w[subs[0], k].astype(tmp.dtype) * u[subs[1], k].astype(tmp.dtype)
		out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)
		

	return out


def get_item_array_from_subs(A, ref_subs):
	"""
		Get values of ref_subs entries of a dense tensor.
		Output is a 1-d array with dimension = number of non zero entries.
	"""

	return np.array([A[a, i, j] for a, i, j in zip(*ref_subs)])


def preprocess(X):
	"""
		Pre-process input data tensor.
		If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.

		Parameters
		----------
		X : ndarray
			Input data (tensor).

		Returns
		-------
		X : sptensor/dtensor
			Pre-processed data. If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.
	"""

	if not X.dtype == np.dtype(int).type:
		X = X.astype(int)
	if isinstance(X, np.ndarray) and is_sparse(X):
		X = sptensor_from_dense_array(X)
	else:
		X = skt.dtensor(X)

	return X


def is_sparse(X):
	"""
		Check whether the input tensor is sparse.
		It implements a heuristic definition of sparsity. A tensor is considered sparse if:
		given
		M = number of modes
		S = number of entries
		I = number of non-zero entries
		then
		N > M(I + 1)

		Parameters
		----------
		X : ndarray
			Input data.

		Returns
		-------
		Boolean flag: true if the input tensor is sparse, false otherwise.
	"""

	M = X.ndim
	S = X.size
	I = X.nonzero()[0].size

	return S > (I + 1) * M


def sptensor_from_dense_array(X):
	"""
		Create an sptensor from a ndarray or dtensor.
		Parameters
		----------
		X : ndarray
			Input data.

		Returns
		-------
		sptensor from a ndarray or dtensor.
	"""

	subs = X.nonzero()
	vals = X[subs] 

	return skt.sptensor(subs, vals, shape=X.shape, dtype=X.dtype)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
	"""
		Check if a matrix a is symmetric.
	"""

	return np.allclose(a, a.T, rtol=rtol, atol=atol)

def transpose_tensor(A):
	'''
	Assuming the first index is for the layer, it transposes the second and third
	'''
	return np.einsum('aij->aji',A)

def plot_L(values, indices = None, k_i = 5, figsize=(5, 5), int_ticks=False, xlab='Iterations'):

	fig, ax = plt.subplots(1,1, figsize=figsize)
	#print('\n\nL: \n\n',values[k_i:])

	if indices is None:
		ax.plot(values[k_i:])
	else:
		ax.plot(indices[k_i:], values[k_i:])
	ax.set_xlabel(xlab)
	ax.set_ylabel('ELBO')
	if int_ticks:
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.grid()

	plt.tight_layout()
	plt.show()

