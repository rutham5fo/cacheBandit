import math
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import matplotlib as plt
import mat73
from queue import Queue as Q
import sys
import csv
import more_itertools as mit
import random

# Search Space (BLOCK_D) = 128 lines/frames | 128B per line (8*2 BRAMs) => 16KiB => 4KiB per core (8 threads)
bytes_per_word = 8
bytes_per_line = 128
shmem_line = int(bytes_per_line/bytes_per_word)     # No. of elements/words per cache line

workers = 4

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

def insert_uframe(uframe, frame):
    s = set(uframe)
    if frame not in s:
        #print (f'{insert_uframe.__name__} ||| Inserting new frame = {frame} into list', flush=True)
        uframe.append(frame)

def gen_strm (strm_ctxt_fname, interleaved_ctxt_fname, matRandom='random', matName='Matrix name', matPath='Path to matrix', matRows=10, matCols=10, matDensity=1.0, matDtype=np.int8, en_uniqueFrames=1):

    K = matRows
    M = matCols
    density = matDensity

    if (matRandom):
        # Generate matA
        matA = sp.random(K, M, density=density, format='csr', dtype=matDtype)
    else:
        # Grab matrix from file
        matA = load_matfile(matPath, matName, matRows, matCols, matDtype)
        density = matA.nnz / (matA.shape[0] * matA.shape[1])

    #Construct row-wise context and stream
    unique_frames = []
    streamA = []
    thread_ctxt = [[] for _ in range(workers)]
    worker_sel = 0
    for r in range(matA.shape[0]):
        for ind in range(matA.indptr[r], matA.indptr[r+1]):
            col = matA.indices[ind]
            frame = int(col/shmem_line)
            # Using sets to count unique frames visited takes an eternity.
            # Turn it (insert_uframe) off for very large matrices
            if (en_uniqueFrames):
                insert_uframe(unique_frames, frame)
            streamA.append(col)
            thread_ctxt[worker_sel].append(col)
        worker_sel = 0 if (worker_sel == workers-1) else worker_sel+1
    #print (f'{gen_strm.__name__} ||| streamA = {streamA}| len = {len(streamA)}')
    #print (f'{gen_strm.__name__} ||| thread_ctxt = {thread_ctxt}')

    # Generate Interleaved context
    interleaved_ctxt = []
    interleaved_ctxt_tid = []
    worker_sel = 0
    elem_sel = [0 for _ in range(workers)]
    elem_iter = 0
    while (elem_iter < matA.nnz):
        #print (f'{gen_mat_random.__name__} ||| inter_itr = {elem_iter}, worker_sel = {worker_sel}, elem_sel = {elem_sel[worker_sel]}, ctxt_len = {len(thread_context[worker_sel])}')
        if (elem_sel[worker_sel] < len(thread_ctxt[worker_sel])):
            interleaved_ctxt.append(thread_ctxt[worker_sel][elem_sel[worker_sel]])
            interleaved_ctxt_tid.append(worker_sel)
            elem_sel[worker_sel] += 1
            elem_iter += 1
        worker_sel = 0 if (worker_sel == workers-1) else worker_sel+1
    #print (f'{gen_mat_random.__name__} ||| interleaved_ctxt = {interleaved_ctxt}, len = {len(interleaved_ctxt)}')
    #print (f'{gen_mat_random.__name__} ||| interleaved_ctxt_tid = {interleaved_ctxt_tid}, len = {len(interleaved_ctxt_tid)}')

    # Create matrix header
    matA_header = [K, M, matA.nnz, density, len(unique_frames)]

    # Write streams to file to avoid memory overflow
    with open (strm_ctxt_fname, 'w', newline='') as wrf:
        writer = csv.writer(wrf, quoting=csv.QUOTE_NONE)
        writer.writerow(matA_header)
        writer.writerow(streamA)
    with open (interleaved_ctxt_fname, 'w', newline='') as wrf:
        writer = csv.writer(wrf, quoting=csv.QUOTE_NONE)
        writer.writerow(matA_header)
        writer.writerow(interleaved_ctxt)
        writer.writerow(interleaved_ctxt_tid)

    return (matA.nnz, K, M, density, len(unique_frames))

"""
Main function
"""
def main():

    # Run parameters
    run_name = 'cb_test'
    ctxt_path = '.\\context\\'
    gen_random = 0

    # Load matrix path
    matPath = '.\\spmat\\'

    if (gen_random):
        matName = ['KM_2000_d100', 'KM_3000_d100', 'KM_4000_d100', 'KM_5000_d100']
        matRows = [2000, 3000, 4000, 5000]
        matCols = [2000, 3000, 4000, 5000]
        matDensity = [1.0, 1.0, 1.0, 1.0]
        matEnFrames = [1, 1, 1, 1]
        matDtype = np.int8      # Not used by trace/stream generator, choose the smallest possible representation to keep runtime memory requirements small
    else:
        #matName = ['bcsstk10', 'bcsstk13', 'bcsstk17', 'c8_mat11', 'cq9', 'fv1', 'kl02', 'lhr34c', 'pdb1HYS', 'psmigr_1', 'wiki-Vote', 'ca-HepTh', 'p2p-Gnutella04', 'as-735', 'amazon0312']
        #matRows = [1086, 2003, 10974, 4562, 9278, 9604, 71, 35152, 36417, 3140, 8297, 9877, 10879, 7716, 400727]
        #matCols = [1086, 2003, 10974, 5761, 21534, 9604, 36699, 35152, 36417, 3140, 8297, 9877, 10879, 7716, 400727]
        #matEnFrames = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        
        #matName = ['airfoil1', 'diag', 'crack', 'shock-9']
        #matRows = [4253, 2559, 10240, 36476]
        #matCols = [4253, 2559, 10240, 36476]
        #matEnFrames = [1, 1, 1, 1]
        
        matName = ['sinc15']
        matRows = [11532]
        matCols = [11532]
        matEnFrames = [1]

        matDtype = np.int8      # Not used by trace/stream generator, choose the smallest possible representation to keep runtime memory requirements small

    # Check if python is in 64bit version
    #print(f'{sys.maxsize:02x}, {sys.maxsize > 2**32}')

    for id, elem in enumerate(list(zip(matName, matRows, matCols, matEnFrames))):
        mname, mrow, mcol, menFrames = elem
        strmA_fname = ctxt_path + 'streamA_' + run_name + '_' + mname + '.csv'
        interleavedA_fname = ctxt_path + 'interleavedA_' + run_name + '_' + mname + '.csv'
        mdensity = matDensity[id] if (gen_random) else 0
        matA_nnz, matA_rows, matA_cols, den, unique_frames = gen_strm(strmA_fname, interleavedA_fname, matRandom=gen_random, matName=mname, matPath=matPath, matRows=mrow, matCols=mcol, matDensity=mdensity, matDtype=matDtype, en_uniqueFrames=menFrames)
        print (f'{main.__name__} ||| Matrix A dim K = {matA_rows}, M = {matA_cols}, NNZ = {matA_nnz}, Density = {den} | Unique frames visited = {unique_frames}')

if __name__ == "__main__":
    main()
