"""Functions for handling the data."""

import networkx as nx
import numpy as np
import pandas as pd
import sktensor as skt


def import_data(dataset, undirected=False, ego='source', alter='target', force_dense=True, noselfloop=True, verbose=True,
				binary=True):
	"""
		Import data, i.e. the adjacency tensor, from a given folder.

		Return the NetworkX graph and its numpy adjacency tensor.

		Parameters
		----------
		dataset : str
				  Path of the input file.
		undirected : bool
					 If set to True, the algorithm considers an undirected graph.
		ego : str
			  Name of the column to consider as source of the edge.
		alter : str
				Name of the column to consider as target of the edge.
		force_dense : bool
					  If set to True, the algorithm is forced to consider a dense adjacency tensor.
		noselfloop : bool
					 If set to True, the algorithm removes the self-loops.
		verbose : bool
				  Flag to print details.
		binary : bool
				 Flag to force the matrix to be binary.

		Returns
		-------
		A : list
			List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
		B : ndarray
			Graph adjacency tensor.
		nodes : list
				List of nodes IDs.
	"""

	# read adjacency file
	df_adj = pd.read_csv(dataset, sep='\s+', header = 0) 
	print('{0} shape: {1}'.format(dataset, df_adj.shape))

	# create the graph adding nodes and edges
	A = read_graph(df_adj=df_adj, ego=ego, alter=alter, undirected=undirected, noselfloop=noselfloop, verbose=verbose,
				   binary=binary)

	nodes = list(A[0].nodes)
	print('\nNumber of nodes =', len(nodes))
	print('Number of layers =', len(A))
	if verbose:
		print_graph_stat(A)

	# save the multilayer network in a tensor with all layers
	if force_dense:
		B, rw = build_B_from_A(A, nodes=nodes)
		B_T, data_T_vals = None, None
	else:
		B, B_T, data_T_vals, rw = build_sparse_B_from_A(A)

	return A, B, B_T, data_T_vals


def read_graph(df_adj, ego='source', alter='target', undirected=False, noselfloop=True, verbose=True, binary=True):
	"""
		Create the graph by adding edges and nodes.

		Return the list MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.

		Parameters
		----------
		df_adj : DataFrame
				 Pandas DataFrame object containing the edges of the graph.
		ego : str
			  Name of the column to consider as source of the edge.
		alter : str
				Name of the column to consider as target of the edge.
		undirected : bool
					 If set to True, the algorithm considers an undirected graph.
		noselfloop : bool
					 If set to True, the algorithm removes the self-loops.
		verbose : bool
				  Flag to print details.
		binary : bool
				 If set to True, read the graph with binary edges.

		Returns
		-------
		A : list
			List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
	"""

	# build nodes
	egoID = df_adj[ego].unique()
	alterID = df_adj[alter].unique()
	nodes = list(set(egoID).union(set(alterID)))
	nodes.sort()

	L = df_adj.shape[1] - 2  # number of layers
	# build the multilayer NetworkX graph: create a list of graphs, as many graphs as there are layers
	if undirected:
		A = [nx.MultiGraph() for _ in range(L)]
	else:
		A = [nx.MultiDiGraph() for _ in range(L)]

	if verbose:
		print('Creating the network ...', end=' ')
	# set the same set of nodes and order over all layers
	for l in range(L):
		A[l].add_nodes_from(nodes)

	for index, row in df_adj.iterrows():
		v1 = row[ego]
		v2 = row[alter]
		for l in range(L):
			if row[l + 2] > 0:
				if binary:
					if A[l].has_edge(v1, v2):
						A[l][v1][v2][0]['weight'] = 1
					else:
						A[l].add_edge(v1, v2, weight=1)
				else:
					if A[l].has_edge(v1, v2):
						A[l][v1][v2][0]['weight'] += int(row[l + 2])  # the edge already exists, no parallel edge created
					else:
						A[l].add_edge(v1, v2, weight=int(row[l + 2]))
	if verbose:
		print('done!')

	# remove self-loops
	if noselfloop:
		if verbose:
			print('Removing self loops')
		for l in range(L):
			A[l].remove_edges_from(list(nx.selfloop_edges(A[l])))

	return A


def print_graph_stat(G):
	"""
		Print the statistics of the graph A.

		Parameters
		----------
		G : list
			List of MultiDiGraph NetworkX objects.
	"""

	L = len(G)
	N = G[0].number_of_nodes()

	print('Number of edges and average degree in each layer:')
	for l in range(L):
		E = G[l].number_of_edges()
		k = 2 * float(E) / float(N)
		print(f'{np.round(k, 3)}')
		print(f'E[{l}] = {E} - <k> = {np.round(k, 3)}')

		weights = [d['weight'] for u, v, d in list(G[l].edges(data=True))]
		if not np.array_equal(weights, np.ones_like(weights)):
			M = np.sum([d['weight'] for u, v, d in list(G[l].edges(data=True))])
			kW = 2 * float(M) / float(N)
			print(f'M[{l}] = {M} - <k_weighted> = {np.round(kW, 3)}')

		print(f'Sparsity [{l}] = {np.round(E / (N * N), 3)}')

		print(f'Reciprocity (networkX) = {np.round(nx.reciprocity(G[l]), 3)}')
		print(f'Reciprocity (intended as the proportion of bi-directional edges over the unordered pairs) = '
			  f'{np.round(reciprocal_edges(G[l]), 3)}\n')


def build_B_from_A(A, nodes=None):
	"""
		Create the numpy adjacency tensor of a networkX graph.

		Parameters
		----------
		A : list
			List of MultiDiGraph NetworkX objects.
		nodes : list
				List of nodes IDs.

		Returns
		-------
		B : ndarray
			Graph adjacency tensor.
		rw : list
			 List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
	"""

	N = A[0].number_of_nodes()
	if nodes is None:
		nodes = list(A[0].nodes())
	B = np.empty(shape=[len(A), N, N])
	rw = []
	for l in range(len(A)):
		B[l, :, :] = nx.to_numpy_matrix(A[l], weight='weight', dtype=int, nodelist=nodes)
		rw.append(np.multiply(B[l], B[l].T).sum() / B[l].sum())

	return B, rw


def build_sparse_B_from_A(A):
	"""
		Create the sptensor adjacency tensor of a networkX graph.

		Parameters
		----------
		A : list
			List of MultiDiGraph NetworkX objects.

		Returns
		-------
		data : sptensor
			   Graph adjacency tensor.
		data_T : sptensor
				 Graph adjacency tensor (transpose).
		v_T : ndarray
			  Array with values of entries A[j, i] given non-zero entry (i, j).
		rw : list
			 List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
	"""

	N = A[0].number_of_nodes()
	L = len(A)
	rw = []

	d1 = np.array((), dtype='int64')
	d2, d2_T = np.array((), dtype='int64'), np.array((), dtype='int64')
	d3, d3_T = np.array((), dtype='int64'), np.array((), dtype='int64')
	v, vT, v_T = np.array(()), np.array(()), np.array(())
	for l in range(L):
		b = nx.to_scipy_sparse_matrix(A[l])
		b_T = nx.to_scipy_sparse_matrix(A[l]).transpose()
		rw.append(np.sum(b.multiply(b_T))/np.sum(b))
		nz = b.nonzero()
		nz_T = b_T.nonzero()
		d1 = np.hstack((d1, np.array([l] * len(nz[0]))))
		d2 = np.hstack((d2, nz[0]))
		d2_T = np.hstack((d2_T, nz_T[0]))
		d3 = np.hstack((d3, nz[1]))
		d3_T = np.hstack((d3_T, nz_T[1]))
		v = np.hstack((v, np.array([b[i, j] for i, j in zip(*nz)])))
		vT = np.hstack((vT, np.array([b_T[i, j] for i, j in zip(*nz_T)])))
		v_T = np.hstack((v_T, np.array([b[j, i] for i, j in zip(*nz)])))
	subs_ = (d1, d2, d3)
	subs_T_ = (d1, d2_T, d3_T)
	data = skt.sptensor(subs_, v, shape=(L, N, N), dtype=v.dtype)
	data_T = skt.sptensor(subs_T_, vT, shape=(L, N, N), dtype=vT.dtype)

	return data, data_T, v_T, rw


def reciprocal_edges(G):
	"""
		Compute the proportion of bi-directional edges, by considering the unordered pairs.

		Parameters
		----------
		G: MultiDigraph
		   MultiDiGraph NetworkX object.

		Returns
		-------
		reciprocity: float
					 Reciprocity value, intended as the proportion of bi-directional edges over the unordered pairs.
	"""

	n_all_edge = G.number_of_edges()
	n_undirected = G.to_undirected().number_of_edges()  # unique pairs of edges, i.e. edges in the undirected graph
	n_overlap_edge = (n_all_edge - n_undirected)  # number of undirected edges reciprocated in the directed network

	if n_all_edge == 0:
		raise nx.NetworkXError("Not defined for empty graphs.")

	reciprocity = float(n_overlap_edge) / float(n_undirected)

	return reciprocity


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