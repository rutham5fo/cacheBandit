import math
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import matplotlib as plt
import mat73
import sys
from contextlib import redirect_stdout

def load_matfile (src_path, file_name, rows, cols, dtype=np.int8):
    # Loader params
    mat_r = rows
    mat_c = cols
    mat_dtype = dtype
    src = src_path + file_name
    mat_vars = {}

    # Print contents
    try:
        mat_cont = sio.loadmat(file_name=src, mdict=mat_vars, squeeze_me=True)
    except Exception as e:
        print (f'{load_matfile.__name__} ||| {e}')
        fname = src + '.mat'
        mat_vars = mat73.loadmat(fname)
        # Fix this flow for CollegeMsg Matrix
    
    prob = mat_vars['Problem']          # LHS is a numpy.0d array
    #print (f'{load_matfile.__name__} ||| Problem: {prob}')
    mat_arr = prob['A']
    #print (f'{load_matfile.__name__} ||| mat_Array: {mat_arr}')
    csc_mat = sp.csc_matrix(mat_arr.all(), shape=(mat_r, mat_c), dtype=mat_dtype)
    csr_mat = csc_mat.tocsr()
    
    return csr_mat

def sparse_correl(mat):

    # Binarize matrix
    mat[mat != 0] = 1
    # Give higher weights (vals) to elements further away from the diagonal
    for r in range(mat.shape[0]):
        for ind in range(mat.indptr[r], mat.indptr[r+1]):
            col = mat.indices[ind]
            diag_diff = abs(r-col)/10           # divide by arbitrary scaling factor to limit range
            mat.data[ind] += diag_diff

    # Start pearson correlation computation
    d = mat.shape[0]
    j = np.array([1 for _ in range(d)], dtype=np.float64)
    r = np.array([x+1 for x in range(d)], dtype=np.float64)
    r2 = r**2
    # Convert to dense
    A = mat.todense()
    n = A.sum()
    # Means and the rest
    sx = np.dot(np.dot(r, A), j.T)
    sy = np.dot(np.dot(j, A), r.T)
    sx2 = np.dot(np.dot(r2, A), j.T)
    sy2 = np.dot(np.dot(j, A), r2.T)
    sxy = np.dot(np.dot(r, A), r.T)
    sdx = np.sqrt(n*sx2 - sx**2)
    sdy = np.sqrt(n*sy2 - sy**2)
    
    # Pearson Correlation
    pc = (n*sxy-sx*sy)/(sdx*sdy)

    return pc

"""
Main function
"""
def main():

    # Run parameters
    run_name = 'cb_spmat_correl'
    log_path = '.\\logs\\'
    # Load matrix path
    matPath = '.\\spmat\\'

    matName = ['bcsstk17', 'shock-9', 'fv1', 'airfoil1', 'diag', 'big_dual', 'Chem97ZtZ', 'crack']
    matRows = [10974, 36476, 9604, 4253, 2559, 30269, 2541, 10240]
    matCols = [10974, 36476, 9604, 4253, 2559, 30269, 2541, 10240]
    matDtype = np.float64      # Not used by trace/stream generator, choose the smallest possible representation to keep runtime memory requirements small

    # Check if python is in 64bit version
    #print(f'{sys.maxsize:02x}, {sys.maxsize > 2**32}')

    cor_fname = log_path + run_name + '.txt'
    with open(cor_fname, 'w') as f:
        with redirect_stdout(f):
            for mname, mrow, mcol in list(zip(matName, matRows, matCols)):
                matA = load_matfile(matPath, mname, mrow, mcol, matDtype)
                corA = sparse_correl(matA)
                print (f'{main.__name__} ||| Matrix {mname} Pearson correlation = {np.mean(corA)}')

if __name__ == "__main__":
    main()
