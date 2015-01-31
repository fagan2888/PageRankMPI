/*
 *@description - this file implements the PageRank algorithm using
 	 	 	 	 MPI. It assumes the web graph file is in coordinate
 	 	 	 	 format representing a column stochastic matrix, i.e., each line
 	 	 	 	 of the input file should be in the format of
 	 	 	 	 <j> <i> <Pji>
 	 	 	 	 where j is the destination node, i is the source node, and Pji
 	 	 	 	 is the transition probability P(j|i).
 	 	 	 	 The node numbers are continuous intergers starting from 0.
 *@author: Yao Zhu (yzhucs@gmail.com).
 */

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

using namespace std;

// some constants definition.
#define INVALID_PROC_RANK 	-1		// indicate an invalid process rank.
#define INVALID_NODE_NUMBER	-1		// indicate an invalid node number.

#define DEPENDENT_LIST_MSG	0		// tag for send and recv dependent lists.
#define IN_OUT_INDEX_MSG	1		// tag for send and recv in/out indices.
#define IN_OUT_ZVAL_MSG		2		// tag for send and recv in/out z values.

// initialize a pointer array to be NULL for each element.
template<typename T>
void init_pointer_array(T **ptr, int len) {
	if (ptr == NULL) {
		cerr << "ptr is NULL in init_pointer_array()." << endl;
		exit(-1);
	}
	for (int i = 0; i < len; i++) {
		ptr[i] = NULL;
	}
}

// deallocate a pointer array.
template<typename T>
void deallocate_pointer_array(T **ptr, int len) {
	if (ptr == NULL) {
		return;
	}
	for (int i = 0; i < len; i++) {
		if (ptr[i] != NULL) {
			delete [] ptr[i];
			ptr[i] = NULL;
		}
	}
}

// get the effective size of a terminated integer array.
int get_effective_size(int *ptr, int terminator) {
	if (ptr == NULL) {
		return 0;
	}
	int size = 0;
	while (ptr[size] != terminator) {
		size++;
	}
	return size;
}

// output a vector in column format to a stream.
void output_vector(ostream &out, const double* const z, const int len) {
	if (z == NULL) {
		cerr << "z cannot be NULL in output_vector()." << endl;
		exit(-1);
	}

	out.unsetf(ios::floatfield);
	out.precision(17);
	for (int i = 0; i < len; i++) {
		out << z[i] << endl;
	}
}

// let process 0 print out some message for showing the progress.
void print_msg(string msg, int rank) {
	if (rank == 0) {
		cerr << msg << endl;
	}
}

/*
 *@description - read in the webgraph data. (src[m], dest[m]) is a directed edge
 	 	 	 	 in the webgraph. This function also returns the maximum node
 	 	 	 	 number, which is number of nodes - 1 (we assume the node
 	 	 	 	 number starts from 0).
 *@param - webgraph_file, name of the webgraph file.
 *@param[out] - src, the array of source nodes.
 *@param[out] - dest, the array of the destination nodes.
 *@param[out] - val, the array of transition probability.
 *@param[out] - nnz, number of edges.
 *@return - the maximum node number.
 */
int read_webgraph(const char *webgraph_file, int **src, int **dest,
		 	 	  double **val, int *nnz) {
	std::ifstream wbfile(webgraph_file);
	string line;
	*nnz = 0;
	// count the number of edges.
	while (std::getline(wbfile, line)) {
		// check the 1st character of this line.
		if (line[0] >= '0' && line[0] <= '9') {
			(*nnz)++;
		}
	}
	// allocate storage in coordinate format.
	*src = new int[*nnz];
	*dest = new int[*nnz];
	// allocate val array.
	*val = new double[*nnz];
	if (*src == NULL || *dest == NULL) {
		cerr << "cannot allocate src or dest in read_webgraph()." << endl;
		exit(-1);
	}
	if (*val == NULL) {
		cerr << "cannot allocate val in read_webgraph()." << endl;
		exit(-1);
	}
	// return to the beginning of the input file.
	wbfile.clear();
	wbfile.seekg(0, ios::beg);
	int i = 0;
	int max_nn = -1;	// maximum node number.
	while (std::getline(wbfile, line)) {
		// check the 1st character of this line.
		if (line[0] >= '0' && line[0] <= '9') {
			std::istringstream iss(line);
			// 1st column is dest node, and 2nd column is src node.
			iss >> (*dest)[i] >> (*src)[i];
			iss >> (*val)[i];			// read in the val.
			if ((*src)[i] > max_nn) {
				max_nn = (*src)[i];
			}
			if ((*dest)[i] > max_nn) {
				max_nn = (*dest)[i];
			}
			i++;
		}
	}
	wbfile.close();
	return max_nn;
}

/*
 *@description - construct the CSR format of the sparse PageRank matrix
 	 	 	 	 (dest, src) from the coordinate format given by (src, dest).
 *@param - (src, dest, coord_val) defines the web graph in coordinate format.
 *@param - nnz, number of nonzero elements. it's the length of
 	 	   src, dest, val, and col_ind.
 *@param - N, number of nodes, it's the length of row_ptr and out_degree.
 *@param - (val, col_ind, row_ptr) represents the CSR format of the PageRank
 	 	   matrix (dest, src).
 *@param - out_degree, the array of out degree of each node. will be used to
 	 	   determine the dangling nodes.
 */
void coord2csr(const int* const src, const int* const dest,
			   const double* const coord_val,
			   const int nnz, const int N,
			   double *val, int *col_ind, int *row_ptr, int *out_degree) {
	if (src == NULL || dest == NULL) {
		cerr << "none of src and dest can be NULL in coord2csr()." << endl;
		exit(-1);
	}
	if (coord_val == NULL) {
		cerr << "coord_val cannot be NULL in coord2csr()." << endl;
		exit(-1);
	}
	if (val == NULL || col_ind == NULL || row_ptr == NULL) {
		cerr << "none of val, col_ind, and row_ptr can be NULL in "
				"coord2csr()." << endl;
		exit(-1);
	}
	if (out_degree == NULL) {
		cerr << "out_degree cannot be NULL in coord2csr()." << endl;
		exit(-1);
	}
	std::fill(out_degree, out_degree + N, 0);
	// we need the in_degree to construct the CSR format.
	int *in_degree = new int[N];
	std::fill(in_degree, in_degree + N, 0);
	for (int l = 0; l < nnz; l++) {
		int j = src[l];
		out_degree[j]++;
		int i = dest[l];
		in_degree[i]++;
	}
	// compute row_ptr as the cumsum of in_degree.
	// note row_ptr[N] = nnz. the node numbers are in [0..N-1].
	row_ptr[0] = 0;
	for (int i = 1; i < N+1; i++) {
		row_ptr[i] = row_ptr[i-1] + in_degree[i-1];
	}
	// construct val and col_ind according to row_ptr.
	for (int l = 0; l < nnz; l++) {
		int j = src[l];
		int i = dest[l];
		col_ind[row_ptr[i]] = j;
		val[row_ptr[i]] = coord_val[l];		// supplied by coord_val.
		row_ptr[i]++;
	}
	// recompute row_ptr as the cumsum of in_degree.
	row_ptr[0] = 0;
	for (int i = 1; i < N+1; i++) {
		row_ptr[i] = row_ptr[i-1] + in_degree[i-1];
	}
	// deallocate in_degree.
	if (in_degree != NULL) {
		delete [] in_degree;
	}
}

/*
 *@description - get the rank of the process that has the node with the given
 	 	 	 	 node number. This partition assumes all the remnant elements
 	 	 	 	 are given to the last process.
 *@param - nn, node number.
 *@param - N, total number of nodes.
 *@param - nproc, number of processes.
 */
int nn2rank(const int nn, const int N, const int nproc) {
	int quota = int(N / nproc);
	if (nn >= quota * nproc) {
		return (nproc - 1);
	} else {
		return int(nn / quota);
	}
}

/*
 *@description - get the number of rows a process p has from the row parition
 				 scheme.
 *@param - N, total number of rows (one row correspond to one node).
 *@param - nproc, total number of processes.
 *@param - rank, of the process.
 */
int get_Nlocal(const int N, const int nproc, const int rank) {
	int quota = int(N / nproc);
	if (rank == nproc - 1) {	// the last process.
		return (N - (nproc - 1) * quota);
	} else {
		return quota;
	}
}

/*
 *@description - get the first node number that belongs to a process p.
 */
int get_start_nn(const int N, const int nproc, const int rank) {
	int quota = int(N / nproc);
	return (quota * rank);
}

/*
 *@description - construct the dependent lists for each process from the
 	 	 	 	 data (src, dest), and the scheme of allocating nodes to
 	 	 	 	 processes.
 *@param - nnz, the length of src and dest.
 *@param - N, total number of nodes.
 *@param - nproc, number of processes.
 *@param[out] - all_dependent_list[p] will store the processes that depdend
 	 	 	 	on process p.
 *@param[out] - all_dependent_list_size[p] stores the length of
 	 	 	    all_dependent_list[p].
 */
void get_all_dependent_list(const int* const src, const int* const dest,
							const int nnz, const int N, const int nproc,
							int **all_dependent_list,
							int *all_dependent_list_size) {
	if (src == NULL || dest == NULL) {
		cerr << "none of src and dest can be NULL in "
				"get_all_dependent_list()." << endl;
		exit(-1);
	}
	if (all_dependent_list == NULL || all_dependent_list_size == NULL) {
		cerr << "none of all_dependent_list and all_dependent_list_size can be "
				"NULL in get_all_dependent_list()." << endl;
		exit(-1);
	}
	set<int> dependent_set[nproc];
	for (int l = 0; l < nnz; l++) {
		int j = src[l];
		int i = dest[l];
		int p1 = nn2rank(j, N, nproc);
		int p2 = nn2rank(i, N, nproc);
		if (p1 == p2) {		// no need for self dependency.
			continue;
		}
		// p2 is a dependent of p1.
		if (dependent_set[p1].find(p2) == dependent_set[p1].end()) {
			dependent_set[p1].insert(p2);
		}
	}
	for (int p = 0; p < nproc; p++) {
		if (dependent_set[p].size() > 0) {
			all_dependent_list_size[p] = dependent_set[p].size();
			all_dependent_list[p] = new int[all_dependent_list_size[p]];
			int index = 0;
			for (set<int>::iterator it = dependent_set[p].begin();
				 it != dependent_set[p].end(); ++it) {
				all_dependent_list[p][index] = *it;
				index++;
			}
		}
	}
}

/*
 *@description - construct the in_index array of a process p, where in_index[p']
 	 	 	 	 is the list of node numbers that p needs from p'. the sparse
 	 	 	 	 matrix is passed in CSR format. note we only need the col_ind.
 *@param - (col_ind, row_ptr) represents the CSR format.
 *@param - nnz, the length of col_ind.
 *@param - rank, rank of the process calling this function.
 *@param - N, total number of nodes.
 *@param - nproc, number of processes.
 *@param[out] - in_index[p'] will store the node numbers that process p needs
 				from p'.
 */
void get_in_index(const int* const col_ind, const int nnz, const int rank,
				  const int N, const int nproc, int **in_index) {
	if (col_ind == NULL) {
		cerr << "col_ind cannot be NULL in get_in_index()." << endl;
		exit(-1);
	}
	if (in_index == NULL) {
		cerr << "in_index cannot be NULL in get_in_index()." << endl;
		exit(-1);
	}
	set<int> in_index_set[nproc];
	// scan col_ind to determine the refined dependency structure.
	for (int l = 0; l < nnz; l++) {
		int j = col_ind[l];		// note j is still globally numbered after
								// after row parition.
		int p = nn2rank(j, N, nproc);
		if (rank == p) {		// no need for self dependency.
			continue;
		}
		// current process is a dependent of p for j.
		if (in_index_set[p].find(j) == in_index_set[p].end()) {
			in_index_set[p].insert(j);
		}
	}
	for (int p = 0; p < nproc; p++) {
		if (in_index_set[p].size() > 0) {
			// we store a terminator at last to determine the effective length
			// of in_index[p] later without an extra array for size information.
			in_index[p] = new int[in_index_set[p].size() + 1];
			//std::fill(in_index[p], in_index[p] + (in_index_set[p].size() + 1),
			//		  INVALID_NODE_NUMBER);
			int index = 0;
			for (set<int>::iterator it = in_index_set[p].begin();
				 it != in_index_set[p].end(); ++it) {
				in_index[p][index] = *it;
				index++;
			}
			in_index[p][in_index_set[p].size()] = INVALID_NODE_NUMBER;
		}
	}
}

/*
 *@description - compute the sparse MatVec multiplication.
 *@param - (val, col_ind, row_ptr) represents the sparse row block the process
 	 	   has.
 *@param - Nlocal is the length of row_ptr.
 *@param - start_nn is the first node number the process has.
 *@param - N, total number of nodes.
 *@param - nproc, total number of processes.
 *@param - z_cur, current z vector, in full storage.
 *@param[out] - z_new, the new z vector containing the updated elements from
 	 	 	 	the calling process.
 */
void sparse_matvec(const double* const val, const int* const col_ind,
				   const int* const row_ptr, const int Nlocal,
				   const int start_nn, const int N, const int nproc,
				   double *z_cur, double *z_new) {
	if (val == NULL || col_ind == NULL || row_ptr == NULL) {
		cerr << "none of val, col_ind and row_ptr can be NULL in "
				"sparse_matvec()." << endl;
		exit(-1);
	}
	if (z_cur == NULL || z_new == NULL) {
		cerr << "none of z_cur, and z_new can be NULL in "
				"sparse_matvec()." << endl;
		exit(-1);
	}
	for (int i = 0; i < Nlocal; i++) {
		z_new[start_nn + i] = 0.0;
		for (int l = row_ptr[i]; l < row_ptr[i+1]; l++) {
			z_new[start_nn + i] += val[l] * z_cur[col_ind[l]];
		}
	}
}

/*
 *@description - compute the sum of the ranks of dangling nodes.
 *@param - out_degree, the vector storing the out degrees of the nodes belong
 	 	   to this process.
 *@param - Nlocal, the length of out_degree.
 *@param - start_nn is the first node number the process has.
 *@param - z, the rank vector.
 */
double get_dangle_rank_sum(const int* const out_degree, int Nlocal,
						   int start_nn, const double* const z) {
	if (out_degree == NULL || z == NULL) {
		cerr << "none of out_degree, and z can be NULL in "
				"get_dangle_rank_sum()." << endl;
		exit(-1);
	}
	double dangle_rank_sum = 0.0;
	for (int i = 0; i < Nlocal; i++) {
		if (out_degree[i] == 0) {
			dangle_rank_sum += z[start_nn + i];
		}
	}
	return dangle_rank_sum;
}

/*
 *@descriptioni - it assumes the node number starts from 0.
 *@param - argv[1] - name of the file storing web graph.
 *@param - argv[2] - alpha, the damping factor.
 *@param - argv[3] - tol, the convergence criterion.
 *@param - argv[4] - file to store the solution vector.
 */
int parallel_pagerank(int argc, char *argv[]) {
	int nproc;							// total number of processes.
	int N;								// total number of nodes.
	MPI_Status recv_status;
	int rank;							// rank of the process.

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (argc != 5) {
		if (rank == 0) {
			cerr << "to run this program must supply the following command "
					"line arguments (in order)" << endl;
			cerr << "argv[1]---web graph file." << endl;
			cerr << "argv[2]---damping factor." << endl;
			cerr << "argv[3]---tol for convergence." << endl;
			cerr << "argv[4]---file to save the solution." << endl;
		}
		exit(-1);
	}

	//--------process 0 reads in the web graph, change it to CSR format
	//--------compute information like out degree, and dependent lists,
	//--------and then partition the nodes and distributes them to each process.

	// data only used by process 0 for reading in the webgraph.
	double *val = NULL;
	int *col_ind = NULL;
	int *row_ptr = NULL;
	int *out_degree = NULL;
	int **all_dependent_list = NULL; // dependent list of all processes.
	int *all_dependent_list_size = NULL; // record the length of each
										 // all_dependent_list[p].
	if (rank == 0) {
		print_msg("process 0 reads in the web graph data......", rank);
		// process 0 reads in the web graph from file in coordinate format.
		int *src = NULL;
		int *dest = NULL;
		int nnz = 0;
		double *coord_val = NULL;
		N = read_webgraph(argv[1], &src, &dest, &coord_val, &nnz) + 1;
		cerr << "N = " << N << endl;
		// record the out degree of each node. it will be used to determine
		// the dangling nodes.
		out_degree = new int[N];
		// construct the CSR format of the sparse matrix P^{T} from the
		// coordinate format.
		val = new double[nnz];
		col_ind = new int[nnz];
		row_ptr = new int[N+1];
		coord2csr(src, dest, coord_val, nnz, N, val, col_ind, row_ptr,
				  out_degree);

		// process 0 constructs the dependent lists of all processes.
		all_dependent_list = new int* [nproc];
		for (int p = 0; p < nproc; p++) {
			all_dependent_list[p] = NULL;
		}
		all_dependent_list_size = new int[nproc];
		std::fill(all_dependent_list_size, all_dependent_list_size + nproc, 0);
		get_all_dependent_list(src, dest, nnz, N, nproc, all_dependent_list,
							   all_dependent_list_size);
		// we no longer need src and dest.
		if (src != NULL) {
			delete [] src;
			src = NULL;
		}
		if (dest != NULL) {
			delete [] dest;
			dest = NULL;
		}
		nnz = 0;
		// we no longer need coord_val
		if (coord_val != NULL) {
			delete [] coord_val;
			coord_val = NULL;
		}
	}

	print_msg("read in the webgraph is done.", rank);

	/***collective communications for scattering necessary information.***/
	// broadcast the total number of nodes.
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// process 0 prepare the information for scattering P^{T}.
	// nnz of the row block local to each process. it's also the sendcounts
	// buffer used to scatter val and col_ind.
	int *local_nnz = NULL;
	// the data sent to process p start at location displs_csr[p] when
	// scattering val and col_ind.
	int *displs_csr = NULL;
	// the sendcounts buffer used to scatter row_ptr and out_degree.
	int *sendcounts_node = NULL;
	// the data sent to process p start at location displs_row_ptr[p] when
	// scattering row_ptr and out_degree.
	int *displs_node = NULL;
	if (rank == 0) {
		int quota = int(N / nproc);
		local_nnz = new int[nproc];
		displs_csr = new int[nproc];
		sendcounts_node = new int[nproc];
		displs_node = new int[nproc];
		for (int p = 0; p < nproc; p++) {
			int sp = p * quota;
			int tp;
			if (p != nproc - 1) {
				tp = (p + 1) * quota - 1;
			} else {
				tp = N - 1;
			}
			local_nnz[p] = row_ptr[tp+1] - row_ptr[sp];
			displs_csr[p] = row_ptr[sp];
			sendcounts_node[p] = get_Nlocal(N, nproc, p);
			displs_node[p] = sp;
		}
	}
	// (local_val, local_col_ind, local_row_ptr) represents the CSR format
	// of the row block local to this process.
	double *local_val = NULL;
	int *local_col_ind = NULL;
	int *local_row_ptr = NULL;
	int *local_out_degree = NULL;
	// allocate space for local_row_ptr according to the row parition scheme.
	int start_nn = get_start_nn(N, nproc, rank);	// first node belong to the
													// process.
	// this process will possess node numbers [start_nn..Nlocal]
	// assuming the contiguous row partition.
	int Nlocal = get_Nlocal(N, nproc, rank);
	local_row_ptr = new int[Nlocal + 1];	// local_row_ptr[Nlocal] stores nnz
											// local to this process from
											// row partition.
	// process 0 scatter local_nnz to each process.
	MPI_Scatter(local_nnz, 1, MPI_INT, local_row_ptr+Nlocal, 1, MPI_INT,
				0, MPI_COMM_WORLD);
	// allocate space for local_val and local_col_ind with the size given by
	// local_row_ptr[Nlocal].
	local_val = new double[local_row_ptr[Nlocal]];
	local_col_ind = new int[local_row_ptr[Nlocal]];
	/***process 0 scatterv the sparse matrix P^{T} in CSR format.***/
	// process 0 scatterv val and col_ind to each process.
	MPI_Scatterv(val, local_nnz, displs_csr, MPI_DOUBLE, local_val,
				 local_row_ptr[Nlocal], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(col_ind, local_nnz, displs_csr, MPI_INT, local_col_ind,
				 local_row_ptr[Nlocal], MPI_INT, 0, MPI_COMM_WORLD);
	// process 0 scatterv row_ptr to each process.
	MPI_Scatterv(row_ptr, sendcounts_node, displs_node, MPI_INT,
				 local_row_ptr, Nlocal, MPI_INT, 0, MPI_COMM_WORLD);
	// adjust local_row_ptr to make it start from 0.
	for (int i = Nlocal - 1; i >= 0; i--) {
		local_row_ptr[i] -= local_row_ptr[0];
	}
	// process 0 scatterv out_degree to each process.
	local_out_degree = new int[Nlocal];
	MPI_Scatterv(out_degree, sendcounts_node, displs_node, MPI_INT,
				 local_out_degree, Nlocal, MPI_INT, 0, MPI_COMM_WORLD);
	// process 0 sends dependent_list to each process.
	int *my_dependent_list = NULL;		// list of the processes that depend
										// on this process for z values.
	my_dependent_list = new int[nproc];
	std::fill(my_dependent_list, my_dependent_list + nproc, INVALID_PROC_RANK);
	if (rank == 0) {
		for (int p = 1; p < nproc; p++) {
			if (all_dependent_list[p] != NULL) {
				MPI_Send(all_dependent_list[p], all_dependent_list_size[p],
						 MPI_INT, p, DEPENDENT_LIST_MSG, MPI_COMM_WORLD);
			}
		}
		// local copy all_dependent_list[0] if it is not NULL.
		if (all_dependent_list[0] != NULL) {
			std::copy(all_dependent_list[0],
					  all_dependent_list[0]+all_dependent_list_size[0],
					  my_dependent_list);
		}
	} else {
		MPI_Recv(my_dependent_list, nproc, MPI_INT, 0, DEPENDENT_LIST_MSG,
				 MPI_COMM_WORLD, &recv_status);
	}

	print_msg("distribute sparse matrix is done.", rank);

	// process 0 cleans up some space no longer needed.
	if (rank == 0) {
		if (val != NULL) {
			delete [] val;
			val = NULL;
		}
		if (col_ind != NULL) {
			delete [] col_ind;
			col_ind = NULL;
		}
		if (row_ptr != NULL) {
			delete [] row_ptr;
			row_ptr = NULL;
		}
		if (out_degree != NULL) {
			delete [] out_degree;
			out_degree = NULL;
		}
		if (all_dependent_list != NULL) {
			for (int p = 0; p < nproc; p++) {
				if (all_dependent_list[p] != NULL) {
					delete [] all_dependent_list[p];
					all_dependent_list[p] = NULL;
				}
			}
			delete [] all_dependent_list;
			all_dependent_list = NULL;
		}
		if (local_nnz != NULL) {
			delete [] local_nnz;
			local_nnz = NULL;
		}
		if (displs_csr != NULL) {
			delete [] displs_csr;
			displs_csr = NULL;
		}
		// sendcounts_node and displs_node will be reused when gathering
		// the solution back.
	}

	//--------related processes exchange in/out index information pairwisely.

	// recv and send buffers to exchange z values with other processes.
	// the index buffer will provide the node index of the values in z buffer.
	int **in_index = new int* [nproc];
	init_pointer_array(in_index, nproc);
	int *in_index_len = new int[nproc];		// in_index_len[p] is the effective
											// size of in_index[p].
	std::fill(in_index_len, in_index_len + nproc, 0);
	double **in_z = new double* [nproc];
	init_pointer_array(in_z, nproc);
	int	**out_index = new int* [nproc];
	init_pointer_array(out_index, nproc);
	double **out_z = new double* [nproc];
	init_pointer_array(out_z, nproc);

	// we store the solution vector in full. but each process only needs to
	// operate on the elements it need.
	double *z1 = new double[N];
	double *z2 = new double[N];
	double *z_cur = z1;		// point to the current z vector.
	double *z_new = z2;		// point to the new z vector.
	// init z_cur to be 1/N.
	std::fill(z_cur + start_nn, z_cur + start_nn + Nlocal, 1.0/N);

	// get the parameters from command line.
	double alpha = atof(argv[2]);
	double tol = atof(argv[3]);
	double diff;				// difference between z_cur and z_new as
								// measured by l2-norm.
	// the MatVec iteration.
	int iter = 0;				// the iteration number.
	// prepare the precision for display the difference.
	cerr.unsetf(ios::floatfield);
	cerr.precision(17);

	// construct the in_index buffer.
	get_in_index(local_col_ind, local_row_ptr[Nlocal], rank, N,
				 nproc, in_index);

	MPI_Request request[nproc];			// used by MPI_Isend.

	// send in_index to each possessing process.
	for (int p = 0; p < nproc; p++) {
		if (in_index[p] != NULL) {
			in_index_len[p] =
					get_effective_size(in_index[p], INVALID_NODE_NUMBER);
			MPI_Isend(in_index[p], in_index_len[p], MPI_INT, p,
					  IN_OUT_INDEX_MSG, MPI_COMM_WORLD, &request[p]);
			// allocate in_z accordingly.
			in_z[p] = new double[in_index_len[p]];
		}
	}

	// receive messages containing out_index from each dependent process.
	int *recv_buf = new int[Nlocal];	// a buffer to receive message
										// containing out index.
	for (int ii = 0; ii < nproc; ii++) {
		int p = my_dependent_list[ii];
		if (p != INVALID_PROC_RANK) {
			MPI_Recv(recv_buf, Nlocal, MPI_INT, p, IN_OUT_INDEX_MSG,
					 MPI_COMM_WORLD, &recv_status);
			int out_index_p_size;
			MPI_Get_count(&recv_status, MPI_INT, &out_index_p_size);
			// we store a terminator for indicating the effective size
			// of out_index[p] later.
			out_index[p] = new int[out_index_p_size + 1];
			std::copy(recv_buf, recv_buf + out_index_p_size, out_index[p]);
			out_index[p][out_index_p_size] = INVALID_NODE_NUMBER;
			// allocate out_z accordingly.
			out_z[p] = new double[out_index_p_size];
		}
	}
	if (recv_buf != NULL) {		// no longer need recv_buf.
		delete [] recv_buf;
		recv_buf = NULL;
	}

	// variables for using MPI_Alltoallv().
	int all_sendcounts[nproc];
	int all_sdispls[nproc];
	double *all_sendbuf = NULL;
	int all_recvcounts[nproc];
	int all_rdispls[nproc];
	double *all_recvbuf = NULL;

	// prepare all_sendcounts, all_sdispls according to out_index.
	int all_sendbuf_size = 0;
	for (int p = 0; p < nproc; p++) {
		int out_index_len_p = 0;
		if (out_index[p] != NULL) {
			out_index_len_p =
					get_effective_size(out_index[p], INVALID_NODE_NUMBER);
		}
		all_sdispls[p] = all_sendbuf_size;
		all_sendcounts[p] = out_index_len_p;
		all_sendbuf_size += out_index_len_p;
	}
	// allocate all_sendbuf.
	all_sendbuf = new double[all_sendbuf_size];

	// prepare all_recvcounts, all_rdispls according to in_index.
	int all_recvbuf_size = 0;
	for (int p = 0; p < nproc; p++) {
		all_rdispls[p] = all_recvbuf_size;
		all_recvcounts[p] = in_index_len[p];
		all_recvbuf_size += in_index_len[p];
	}
	// allocate all_sendbuf.
	all_recvbuf = new double[all_recvbuf_size];

	// time variables for profiling. we need to include the time for
	// constructing the dependency among processes.
	struct timeval start_tv, end_tv;
	// barrier for time profiling.
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		gettimeofday(&start_tv, NULL);
	}

	//--------parallel MatVec multiplication through MPI.

	do {
		for (int p = 0; p < nproc; p++) {
			if (out_index[p] != NULL) {
				for (int ii = 0; out_index[p][ii] != INVALID_NODE_NUMBER;
					 ii++) {
					all_sendbuf[all_sdispls[p]+ii] = z_cur[out_index[p][ii]];
				}
			}
		}
		MPI_Alltoallv(all_sendbuf, all_sendcounts, all_sdispls,
					  MPI_DOUBLE, all_recvbuf, all_recvcounts,
					  all_rdispls, MPI_DOUBLE, MPI_COMM_WORLD);
		for (int p = 0; p < nproc; p++) {
			if (in_index[p] != NULL) {
				// retrieve the received z values to z_cur.
				for (int ii = 0; in_index[p][ii] != INVALID_NODE_NUMBER;
					 ii++) {
					z_cur[in_index[p][ii]] = all_recvbuf[all_rdispls[p]+ii];
				}
			}
		}

		// conduct the sparse MatVec multiplication using
		// (local_val, local_col_ind, local_row_ptr) and z_cur.
		sparse_matvec(local_val, local_col_ind, local_row_ptr, Nlocal,
					  start_nn, N, nproc, z_cur, z_new);

		// compute the sum of ranks of dangling nodes.
		double local_dangle_rank_sum =
				get_dangle_rank_sum(local_out_degree, Nlocal, start_nn, z_cur);
		double total_dangle_rank_sum = 0;
		// all reduce to get the total dangle sum.
		MPI_Allreduce(&local_dangle_rank_sum, &total_dangle_rank_sum, 1,
					  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// damping update z_new.
		for (int i = 0; i < Nlocal; i++) {
			z_new[start_nn + i] = alpha * z_new[start_nn + i] +
					(alpha * total_dangle_rank_sum + 1 - alpha) / N;
		}

		// normalize z_new to have unit l1 norm.
		double partial_l1_norm = 0.0;
		for (int i = 0; i < Nlocal; i++) {
			if (z_new[start_nn + i] < 0) {
				cerr << "error: z_new has negative elements." << endl;
				exit(-1);
			}
			partial_l1_norm += z_new[start_nn + i];
		}
		double l1_norm;
		// all reduce to get the l1_norm.
		MPI_Allreduce(&partial_l1_norm, &l1_norm, 1,
					  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		if (l1_norm == 0) {
			cerr << "error: l1_norm of z_new is zero in parallel_pagerank()."
				 << endl;

			exit(-1);
		}
		// conduct l1 normalization.
		for (int i = 0; i < Nlocal; i++) {
			z_new[start_nn + i] /= l1_norm;
		}

		// check for convergence using l2 norm.
		double partial_l2_norm2 = 0.0;
		for (int i = 0; i < Nlocal; i++) {
			partial_l2_norm2 += (z_new[start_nn + i] - z_cur[start_nn + i]) *
								(z_new[start_nn + i] - z_cur[start_nn + i]);
		}
		// all reduce to get the l2 norm square.
		MPI_Allreduce(&partial_l2_norm2, &diff, 1,
					  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		diff = sqrt(diff);		// get the l2 norm.

		// process prints out iteration information.
		iter++;
		if (rank == 0) {
			//cerr << "Iteration " << iter << " ||z_{t+1} - z_t||: "
			//	 << diff << endl;
		}

		if (diff < tol) {
			break;
		}

		// exchange z_cur and z_new.
		double *temp_ptr = z_cur;
		z_cur = z_new;
		z_new = temp_ptr;
	} while (true);

	// barrier for time profiling.
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		gettimeofday(&end_tv, NULL);
		double tElapsed = (end_tv.tv_sec + end_tv.tv_usec/1000000.0) -
						  (start_tv.tv_sec + start_tv.tv_usec/1000000.0);
		cout << "Time: " << tElapsed << " seconds when using "
			 << nproc << " processes." << endl;
		cerr << "Iteration " << iter << " ||z_{t+1} - z_t||: "
			 << diff << endl;
	}

	// process 0 gather the z vector and stores it.
	double *z_sol = NULL;	// the solution vector.
	if (rank == 0) {
		z_sol = new double[N];
	}
	MPI_Gatherv(z_new + start_nn, Nlocal, MPI_DOUBLE, z_sol,
			    sendcounts_node, displs_node, MPI_DOUBLE, 0,
			    MPI_COMM_WORLD);
	if (rank == 0) {
		// get the highest ranked node id and its pagerank value.
		double max_pgrank = -1.0;
		int max_id = -1;
		for (int i = 0; i < N; i++) {
			if (z_sol[i] > max_pgrank) {
				max_id = i;
				max_pgrank = z_sol[i];
			}
		}
		cerr << "the node with the highest pagerank value is node: " << max_id
			 << ", and its pagerank value is: " << max_pgrank << endl;

		// save the vector.
		ofstream out(argv[4], ios::trunc);
		output_vector(out, z_sol, N);
		cerr << "the solution vector has been saved in file "
			 << argv[4] << endl;
		// we no longer need sendcounts_node and displs_node.
		if (sendcounts_node != NULL) {
			delete [] sendcounts_node;
			sendcounts_node = NULL;
		}
		if (displs_node != NULL) {
			delete [] displs_node;
			displs_node = NULL;
		}
		// we no longer need z_sol.
		if (z_sol != NULL) {
			delete [] z_sol;
			z_sol = NULL;
		}
	}

	// z_cur and z_new are no longer needed.
	z_cur = NULL;
	z_new = NULL;

	// clean up all the space.
	if (local_val != NULL) {
		delete [] local_val;
		local_val = NULL;
	}
	if (local_col_ind != NULL) {
		delete [] local_col_ind;
		local_col_ind = NULL;
	}
	if (local_row_ptr != NULL) {
		delete [] local_row_ptr;
		local_row_ptr = NULL;
	}
	if (local_out_degree != NULL) {
		delete [] local_out_degree;
		local_out_degree = NULL;
	}
	if (in_index != NULL) {
		deallocate_pointer_array(in_index, nproc);
		delete [] in_index;
		in_index = NULL;
	}
	if (in_index_len != NULL) {
		delete [] in_index_len;
		in_index_len = NULL;
	}
	if (in_z != NULL) {
		deallocate_pointer_array(in_z, nproc);
		delete [] in_z;
		in_z = NULL;
	}
	if (out_index != NULL) {
		deallocate_pointer_array(out_index, nproc);
		delete [] out_index;
		out_index = NULL;
	}
	if (out_z != NULL) {
		deallocate_pointer_array(out_z, nproc);
		delete [] out_z;
		out_z = NULL;
	}
	if (z1 != NULL) {
		delete [] z1;
		z1 = NULL;
	}
	if (z2 != NULL) {
		delete [] z2;
		z2 = NULL;
	}

	return 0;
}

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);

	parallel_pagerank(argc, argv);

	MPI_Finalize();
}
