import ssgetpy as ssg
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from contextlib import redirect_stdout

def load_matfile (src_path, file_name, rows, cols, dtype=np.int8):
    # Loader params
    mat_r = rows
    mat_c = cols
    mat_dtype = dtype
    src = src_path + file_name
    mat_vars = {}

    # Print contents
    mat_cont = sio.loadmat(file_name=src, mdict=mat_vars, squeeze_me=True)
    prob = mat_vars['Problem']          # LHS is a numpy.0d array
    #print (f'{load_matfile.__name__} ||| Problem: {prob}')
    mat_arr = prob['A']
    #print (f'{load_matfile.__name__} ||| mat_Array: {mat_arr}')
    csc_mat = sp.csc_matrix(mat_arr.all(), shape=(mat_r, mat_c), dtype=mat_dtype)
    csr_mat = csc_mat.tocsr()
    
    return csr_mat

def plot_spmat (src, name, rows, cols, dpi=300):
    mat = load_matfile(src, name, rows, cols)
    figName = src + name + '.png'
    fig, ax = plt.subplots(figsize=(100, 100))
    ax.spy(mat, markersize=1, color='blue')
    ax.set_title("Sparse CSR Matrix Visualization")
    #plt.show()
    plt.savefig(figName, format='png', dpi=dpi)

def main ():

    matName = ['bcsstk10', 'bcsstk13', 'bcsstk17', 'c8_mat11', 'cq9', 'fv1', 'kl02', 'lhr34c', 'pdb1HYS', 'psmigr_1', 'ca-HepTh', 'p2p-Gnutella04', 'as-735', 'amazon0312']
    matDest = '.\\spmat\\'
    matFormat = 'MAT'

    log_name = matDest + 'sspget_log.txt'

    # Get a matrix from the collection
    with open(log_name, 'w') as f:
        for mid, mname in enumerate(matName):
            print (f'{main.__name__} ||| Getting matrix[{mid}]: {mname}')
            with redirect_stdout(f):
                mat = ssg.search(name_or_id=mname)[0]
                density = mat.nnz / (mat.rows * mat.cols)
                # Print Deets
                print (f'{main.__name__} ||| Matrix Deets[{mid}]: Name = {mat.name}, ID = {mat.id}, Group = {mat.group}, Problem Kind = {mat.kind} | Rows = {mat.rows}, Cols = {mat.cols}, NNZ = {mat.nnz}, Density = {density} | DataType = {mat.dtype} \n')
                # Donwload matrix to matDest
                mat.download(format='MAT', destpath=matDest)
                # Plot matrix
                #plot_spmat(matDest, mname, mat.rows, mat.cols)

if __name__ == "__main__":
    main()