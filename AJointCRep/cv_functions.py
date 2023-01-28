"""
	Functions used in the k-fold cross-validation procedure.
"""

import AJointCRep_AnomalyDetection as AJCRep
# import AnomalyDetection as AJCRep
import numpy as np
from sklearn import metrics
from collections import Counter
import yaml
from scipy.stats import poisson
import pandas as pd



def lambda0_full(u, v, w):
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

	if w.ndim == 2:
		M = np.einsum('ik,jk->ijk', u, v)
		M = np.einsum('ijk,ak->aij', M, w)
	else:
		M = np.einsum('ik,jq->ijkq', u, v)
		M = np.einsum('ijkq,akq->aij', M, w)

	return M


def calculate_accuracy_joint(pred, data0, mask=None):
	"""
		Return the accuracy of the link prediction for the pair of edges.

		Parameters
		----------
		pred : list
			   Inferred probabilities.
		data0 : ndarray
				Given values.
		mask : ndarray
			   Mask for selecting a subset of the adjacency tensor.

		Returns
		-------
		rp : float
			 Random probability.
		mrf : float
			  Maximum relative frequency.
		acc : float
			  Accuracy.
	"""

	p00, p01, p10, p11 = pred
	idx_upper = np.triu(mask).nonzero()

	true_label, pred_label = [], []

	for a, i, j in zip(*idx_upper):
		if any((data0[a, i, j], data0[a, j, i])):
			true_label.append(data0[a, i, j] + data0[a, j, i] - 1)
			ps = [p01[a, i, j], p10[a, i, j], p11[a, i, j]]
			pps = [np.max(ps[:2]), p11[a, i, j]]
			pps /= sum(pps)
			pred_label.append(np.argmax(pps))

	freq = np.array(list(dict(Counter(true_label)).values()))
	mrf = np.max(freq / sum(freq))
	rp = 1 / len(freq)
	acc = metrics.accuracy_score(true_label, pred_label)

	return rp, mrf, acc


def calculate_Q_dense(A,M,eta,pi,mu,mask=None,EPS=1e-12):
	AT = transpose_tensor(A)
	MT = transpose_tensor(M)
	lambdalambdaT = np.einsum('aij,aji->aij', M, M)
	ZZ = M + MT + eta * lambdalambdaT + 1
	AAT = A + AT
	num = mu * np.exp((A + AT) * np.log(pi+EPS)- 2 * np.log(pi+1+EPS))   
	den = num + (1-mu) * np.exp(A * M + AT * MT + np.log(eta+EPS) * A * AT - np.log(ZZ+EPS))    
	if mask is None:
		return num / den
	else:
		return num[mask.nonzero()] / den[mask.nonzero()]


def expected_computation(B,Q, U, V, W, eta, mu, pi):

	lambda0_aij = lambda0_full(U, V, W)

	M_marginal = compute_M_marginal(lambda0_aij,Q, eta, mu, pi)

	M_conditional = compute_M_conditional(B, lambda0_aij,Q, eta, mu, pi)

	return M_marginal, M_conditional


def compute_M_marginal(lambda0_aij,Q, eta, mu, pi):

	L = lambda0_aij.shape[0]  
	Z = calculate_Z(lambda0_aij, eta)
	# M = (lambda0_aij + eta * lambda0_aij * transpose_tensor(lambda0_aij)) / Z
	M = ((lambda0_aij + eta * lambda0_aij * transpose_tensor(lambda0_aij)) / Z)*(1-Q[0])
	# M *= (1-mu)#*(1-Q) 
	# M += mu* pi/(1+pi)#*Q
	M += pi/(1+pi)*Q[0]
	for l in np.arange(L):
		np.fill_diagonal(M[l], 0.)

	return M


def compute_M_conditional(B, lambda0_aij,Q, eta, mu, pi,EPS=1e-12):

	L = lambda0_aij.shape[0] 
 
	M =  ( (eta ** transpose_tensor(B) * lambda0_aij) / (eta ** transpose_tensor(B) * lambda0_aij + 1))**(1-Q[0]) 
	M *= ( pi/(1+pi))**Q[0]
	print('M_conditional:',M.sum())
	for l in np.arange(L):
		np.fill_diagonal(M[l], 0.) 

	return M


def compute_M_joint(U, V, W, eta, mu):

	lambda0_aij = lambda0_full(U, V, W)

	Z = calculate_Z(lambda0_aij, eta)

	p00 = mu * 1 / Z
	p10 = mu * lambda0_aij / Z
	p01 = mu * transpose_tensor(p10)
	p11 = mu * (eta * lambda0_aij * transpose_tensor(lambda0_aij)) / Z

	return [p00, p01, p10, p11]


def calculate_Z(lambda0_aij, eta):
	"""
		Compute the normalization constant of the Bivariate Bernoulli distribution.

		Returns
		-------
		Z : ndarray
			Normalization constant Z of the Bivariate Bernoulli distribution.
	"""

	Z = lambda0_aij + transpose_tensor(lambda0_aij) + eta * np.einsum('aij,aji->aij', lambda0_aij, lambda0_aij) + 1
	for l in range(len(Z)):
		assert check_symmetric(Z[l])

	return Z


def transpose_tensor(M):
	"""
		Given M tensor, it returns its transpose: for each dimension a, compute the transpose ij->ji.

		Parameters
		----------
		M : ndarray
			Tensor with the mean lambda for all entries.

		Returns
		-------
		Transpose version of M_aij, i.e. M_aji.
	"""

	return np.einsum('aij->aji', M)

def calculate_expectation(U,V,W,Q,eta,pi=1):    
	lambda0 = lambda0_full(U,V,W)
	lambda0T = transpose_tensor(lambda0)

	lambdalambdaT = np.einsum('aij,aji->aij', lambda0, lambda0)
	ZZ = lambda0 + lambda0T + eta * lambdalambdaT + 1

	return (1-Q) * lambda0 *((1+eta*lambda0T)/ZZ) + Q * pi / ((1+pi)*(1+pi))

def calculate_conditional_expectation(B, u, v, w, mean=None):
	"""
		Compute the conditional expectations, e.g. the parameters of the conditional distribution lambda_{ij}.

		Parameters
		----------
		B : ndarray
			Graph adjacency tensor.
		u : ndarray
			Out-going membership matrix.
		v : ndarray
			In-coming membership matrix.
		w : ndarray
			Affinity tensor.
		eta : float
			  Reciprocity coefficient.
		mean : ndarray
			   Matrix with mean entries.

		Returns
		-------
		Matrix whose elements are lambda_{ij}.
	"""

	if mean is None:
		return lambda0_full(u, v, w)  # conditional expectation (knowing A_ji)
	else:
		return lambda0_full(u, v, w)


def calculate_AUC(pred, data0, mask=None):
	"""
		Return the AUC of the link prediction. It represents the probability that a randomly chosen missing connection
		(true positive) is given a higher score by our method than a randomly chosen pair of unconnected vertices
		(true negative).

		Parameters
		----------
		pred : ndarray
			   Inferred values.
		data0 : ndarray
				Given values.
		mask : ndarray
			   Mask for selecting a subset of the adjacency tensor.

		Returns
		-------
		AUC value.
	"""

	data = (data0 > 0).astype('int')
	if mask is None:
		fpr, tpr, thresholds = metrics.roc_curve(data.flatten(), pred.flatten())
	else: 
		fpr, tpr, thresholds = metrics.roc_curve(data[mask > 0], pred[mask > 0])

	return metrics.auc(fpr, tpr)

def calculate_f1_score(pred, data0, mask=None,threshold=0.0):
	
	Z_pred = np.copy(pred[0])
	Z_pred[Z_pred<threshold] = 0
	Z_pred[Z_pred>=threshold] = 1

	data = (data0 > 0).astype('int')
	if mask is None:
		return metrics.f1_score(data.flatten(), Z_pred.flatten())
	else:
		return metrics.f1_score(data[mask], Z_pred[mask])

def shuffle_indices_all_matrix(N, L, rseed=10):
	"""
		Shuffle the indices of the adjacency tensor.

		Parameters
		----------
		N : int
			Number of nodes.
		L : int
			Number of layers.
		rseed : int
				Random seed.

		Returns
		-------
		indices : ndarray
				  Indices in a shuffled order.
	"""

	n_samples = int(N * N)
	indices = [np.arange(n_samples) for _ in range(L)]
	rng = np.random.RandomState(rseed)
	for l in range(L):
		rng.shuffle(indices[l])

	return indices


def extract_mask_kfold(indices, N, fold=0, NFold=5):
	"""
		Extract a non-symmetric mask using KFold cross-validation. It contains pairs (i,j) but possibly not (j,i).
		KFold means no train/test sets intersect across the K folds.

		Parameters
		----------
		indices : ndarray
				  Indices of the adjacency tensor in a shuffled order.
		N : int
			Number of nodes.
		fold : int
			   Current fold.
		NFold : int
				Number of total folds.

		Returns
		-------
		mask : ndarray
			   Mask for selecting the held out set in the adjacency tensor.
	"""

	L = len(indices)
	mask = np.zeros((L, N, N), dtype=bool)
	for l in range(L):
		n_samples = len(indices[l])
		test = indices[l][fold * (n_samples // NFold):(fold + 1) * (n_samples // NFold)]
		mask0 = np.zeros(n_samples, dtype=bool)
		mask0[test] = 1
		mask[l] = mask0.reshape((N, N))

	return mask


def fit_model(B,nodes, N, L, K,mask=None, **conf):
	"""
		Model directed networks by using a probabilistic generative model that assume community parameters and
		reciprocity coefficient. The inference is performed via EM algorithm.

		Parameters
		----------
		B : ndarray
			Graph adjacency tensor.
		B_T : None/sptensor
			  Graph adjacency tensor (transpose).
		data_T_vals : None/ndarray
					  Array with values of entries A[j, i] given non-zero entry (i, j).
		nodes : list
				List of nodes IDs.
		N : int
			Number of nodes.
		L : int
			Number of layers.
		algo : str
			   Configuration to use (CRep, CRepnc, CRep0).
		K : int
			Number of communities.

		Returns
		-------
		u_f : ndarray
			  Out-going membership matrix.
		v_f : ndarray
			  In-coming membership matrix.
		w_f : ndarray
			  Affinity tensor.
		eta_f : float
				Reciprocity coefficient.
		maxPSL : float
				 Maximum pseudo log-likelihood.
		mod : obj
			  The CRep object.
	"""

	# setting to run the algorithm
	with open(conf['out_folder'] + '/setting.yaml', 'w') as f:
		yaml.dump(conf, f)

	mod = AJCRep.AnomalyDetection(N=N, L=L, K=K, **conf)
	uf, vf, wf, etaf, pibrf, muprf, maxPSL = mod.fit(data=B,mask=mask, nodes=nodes) 
	return uf, vf, wf, etaf, pibrf, muprf, maxPSL, mod

def CalculatePermuation(U_infer,U0):  
	"""
	Permuting the overlap matrix so that the groups from the two partitions correspond
	U0 has dimension NxK, reference memebership
	"""
	N,RANK=U0.shape
	M=np.dot(np.transpose(U_infer),U0)/float(N);   #  dim=RANKxRANK
	rows=np.zeros(RANK);
	columns=np.zeros(RANK);
	P=np.zeros((RANK,RANK));  # Permutation matrix
	for t in range(RANK):
	# Find the max element in the remaining submatrix,
	# the one with rows and columns removed from previous iterations
		max_entry=0.;c_index=1;r_index=1;
		for i in range(RANK):
			if columns[i]==0:
				for j in range(RANK):
					if rows[j]==0:
						if M[j,i]>max_entry:
							max_entry=M[j,i];
							c_index=i;
							r_index=j;
	 
		P[r_index,c_index]=1;
		columns[c_index]=1;
		rows[r_index]=1;

	return P

def cosine_similarity(U_infer,U0):
	"""
	I'm assuming row-normalized matrices 
	"""
	P=CalculatePermuation(U_infer,U0) 
	U_infer=np.dot(U_infer,P);      # Permute infered matrix
	N,K=U0.shape
	U_infer0=U_infer.copy()
	U0tmp=U0.copy()
	cosine_sim=0.
	norm_inf=np.linalg.norm(U_infer,axis=1)
	norm0=np.linalg.norm(U0,axis=1  )
	for i in range(N):
		if(norm_inf[i]>0.):U_infer[i,:]=U_infer[i,:]/norm_inf[i]
		if(norm0[i]>0.): U0[i,:]=U0[i,:]/norm0[i]
	   
	for k in range(K):
		cosine_sim+=np.dot(np.transpose(U_infer[:,k]),U0[:,k])
	U0=U0tmp.copy()
	return U_infer0,cosine_sim/float(N) 

def normalize_nonzero_membership(U):
	"""
		Given a matrix, it returns the same matrix normalized by row.

		Parameters
		----------
		U: ndarray
		   Numpy Matrix.

		Returns
		-------
		The matrix normalized by row.
	"""

	den1 = U.sum(axis=1, keepdims=True)
	nzz = den1 == 0.
	den1[nzz] = 1.

	return U / den1
	
def evalu(U_infer, U0, metric='f1', com=False):
	"""
		Compute an evaluation metric.

		Compare a set of ground-truth communities to a set of detected communities. It matches every detected
		community with its most similar ground-truth community and given this matching, it computes the performance;
		then every ground-truth community is matched with a detected community and again computed the performance.
		The final performance is the average of these two metrics.

		Parameters
		----------
		U_infer : ndarray
				  Inferred membership matrix (detected communities).
		U0 : ndarray
			 Ground-truth membership matrix (ground-truth communities).
		metric : str
				 Similarity measure between the true community and the detected one. If 'f1', it used the F1-score,
				 if 'jaccard', it uses the Jaccard similarity.
		com : bool
			  Flag to indicate if U_infer contains the communities (True) or if they have to be inferred from the
			  membership matrix (False).

		Returns
		-------
		Evaluation metric.
	"""

	if metric not in {'f1', 'jaccard'}:
		raise ValueError('The similarity measure can be either "f1" to use the F1-score, or "jaccard" to use the '
						 'Jaccard similarity!')

	K = U0.shape[1]

	gt = {}
	d = {}
	threshold = 1 / U0.shape[1]
	for i in range(K):
		gt[i] = list(np.argwhere(U0[:, i] > threshold).flatten())
		if com:
			try:
				d[i] = U_infer[i]
			except:
				pass
		else:
			d[i] = list(np.argwhere(U_infer[:, i] > threshold).flatten())
	# First term
	R = 0
	for i in np.arange(K):
		ground_truth = set(gt[i])
		_max = -1
		M = 0
		for j in d.keys():
			detected = set(d[j])
			if len(ground_truth & detected) != 0:
				precision = len(ground_truth & detected) / len(detected)
				recall = len(ground_truth & detected) / len(ground_truth)
				if metric == 'f1':
					M = 2 * (precision * recall) / (precision + recall)
				elif metric == 'jaccard':
					M = len(ground_truth & detected) / len(ground_truth.union(detected))
			if M > _max:
				_max = M
		R += _max
	# Second term
	S = 0
	for j in d.keys():
		detected = set(d[j])
		_max = -1
		M = 0
		for i in np.arange(K):
			ground_truth = set(gt[i])
			if len(ground_truth & detected) != 0:
				precision = len(ground_truth & detected) / len(detected)
				recall = len(ground_truth & detected) / len(ground_truth)
				if metric == 'f1':
					M = 2 * (precision * recall) / (precision + recall)
				elif metric == 'jaccard':
					M = len(ground_truth & detected) / len(ground_truth.union(detected))
			if M > _max:
				_max = M
		S += _max

	return np.round(R / (2 * len(gt)) + S / (2 * len(d)), 4)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
	"""
		Check if a matrix a is symmetric.
	"""

	return np.allclose(a, a.T, rtol=rtol, atol=atol)



def output_adjacency(G, outfile = None):
	"""
		Output the adjacency matrix. Default format is space-separated .csv
		with 3 columns: node1 node2 weight
		INPUT
		----------
		G: Digraph
		   DiGraph NetworkX object.
		outfile: str
				 Name of the adjacency matrix.
	"""  
	edges = list(G.edges(data=True)) 
	try:
		data = [[u, v, d['weight']] for u, v, d in edges]
	except:
		data = [[u, v, 1] for u, v, d in edges]  
	df = pd.DataFrame(data, columns=['source', 'target', 'w'], index=None)
	df.to_csv(outfile, index=False, sep=' ')
	print(f'anomalous network saved in: {outfile}') 
	return  df