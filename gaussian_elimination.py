import numpy as np
import argparse
import sys
# parser = argparse.ArgumentParser()
# parser.add_argument('--array',default = np.asarray([[1,2,3],[2,3,4]]),type = np.ndarray,
# help = 'Input Matrix to be reduced')
# args = parser.parse_args()
# a = args.array


# M = [[0,1,-1,2,0],[1,0,1,2,-1],[0,0,0,1,0],[1,1,0,5,-1]]
# M = [[1,2,-1,0],[0,3,1,4],[-1,1,2,4],[2,3,1,5]]
M = [[2,2,3]]
M = np.asarray(M,dtype=np.float32)

''' Lets put down the algorithm first!
    1) Start at (0,0)  look for a non zero element in 0th column, swap rows if neccessary
    2) normalize to make the (0,0)th element = 1. Now, subtract row1 from other rows so that the
     0th column is = e1 (ie. it looks  like 1,0,0.....)
    3) Move to (1,1). Repeat above procedure. Similarly for the rest
    4) If no nonzero elemment is present at and below the (i,i)th element, skip the column
'''
def swap_rows(M,a_ind,b_ind):
    '''' Swaps two rows of a numpy array'''
    M[[a_ind,b_ind]] = M[[b_ind,a_ind]]
    return M

def check_zero_rows(M):
    ''' Check if any row is a zero vector '''
    row_inds = []
    for row_ind in range(M.shape[0]):
        row = M[row_ind]
        if np.count_nonzero(row) == 0:
            row_inds.append(row_ind)
    return row_inds

def check_nz_col(M, col_ind,row_ind):
    ''' Checks if column col indexed with col_ind of matrix M has a non zero value at index row_ind 
        Swaps rows from below incase  col[row_ind] = 0
        returns None matrixif no non zero value at index ind exists
        Else, returns  M with entry 1 at desired index and the corresponding column
    '''
    col = M[:,col_ind]
    if col[row_ind] != 0 : 
        print('diving by %d'%col[row_ind])
        M[row_ind] = M[row_ind]*1.0/float(col[row_ind])
        print('Column directly given out')
        print('After normalizing :',M)
        return M,col_ind
    elif col[row_ind:].any() != 0:
        print('Rows swapped')
        nz_inds = np.asarray(np.nonzero(col))
        # print(nz_inds)
        nz_ind = nz_inds[nz_inds > row_ind][0]
        M = swap_rows(M,row_ind,nz_ind)
        # print(M)
        M[row_ind] = M[row_ind]*1.0/float(M[row_ind,col_ind])
        return M,col_ind
    else :
        #Case when the col has no non zero entry at and beyond row_ind
        if col_ind == M.shape[1]-1:
            print('Last column reached')
            return M,None
        print('Skipping one column')
        return check_nz_col(M,col_ind+1,row_ind)
        
def reduce_rows_below(M,col_ind,row_ind):
    # print(np.transpose(M[row_ind+1:,col_ind]).shape)
    M[row_ind+1:,:] = M[row_ind+1:,:] - (np.expand_dims(M[row_ind+1:,col_ind],axis=1))*M[row_ind]
    return M


def GE(M):
    '''
    Performs gaussian elimination for a matrix M.
    It is assumed that the columns of the transformation matrix 
    have been transposed so that row reduction can be performed.
    returns : array in Row Reduced Echelon Form
    '''
    col_ind = 0
    row_ind = 0
    while row_ind < M.shape[0] :
        z_inds = check_zero_rows(M) 
        if len(z_inds) > 0:
            # This means M[z_ind] is a zero vector
            rows = [n for n in range(M.shape[0]) if n not in z_inds]
            M = M[rows,:]
        if row_ind >= M.shape[1] or row_ind >= M.shape[0]:
            return M
        M,col_ind = check_nz_col(M,col_ind,row_ind)
        print('After column choosing :\n' ,M)
        if col_ind == None :
            return M
        M = reduce_rows_below(M,col_ind,row_ind)
        print('Row reduction : \n',M)
        col_ind +=1
        row_ind += 1
    print('This')
    return M


# print('Input array :\n {}'.format(M))
print(GE(M))



    