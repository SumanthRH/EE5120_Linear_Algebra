import numpy as np
import argparse
import sys


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
        Skips a column if no non zero value exists at and below row_ind. If such a column is the last one,
        then it returns the matrix unchanged.
        Else, returns  M with entry 1 at desired index and the corresponding column index
    '''
    col = M[:,col_ind]
    if col[row_ind] != 0 : 
        #Normalize by diving by the value at row_ind
        M[row_ind] = M[row_ind]/col[row_ind]
        return M,col_ind
    elif col[row_ind:].any() != 0:
        #If any non zero value exists below given index row_ind, swap rows
        nz_inds = np.asarray(np.nonzero(col))
        nz_ind = nz_inds[nz_inds > row_ind][0]
        M = swap_rows(M,row_ind,nz_ind)
        M[row_ind] = M[row_ind]/M[row_ind,col_ind]
        return M,col_ind
    else :
        #Case when the col has no non zero entry at and beyond row_ind
        if col_ind == M.shape[1]-1:
            return M,None # case when no further simplification is possible
        # Skip one column 
        return check_nz_col(M,col_ind+1,row_ind)
        
def reduce_rows_below(M,col_ind,row_ind):
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
        # print('After column choosing :\n' ,M)
        if col_ind == None :
            return M
        M = reduce_rows_below(M,col_ind,row_ind)
        # print('Row reduction : \n',M)
        col_ind +=1
        row_ind += 1
    return M

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--array',default = '1,2,3.2,3,4',type = str,
    help = 'Input Matrix to be reduced in a11,a12,a13.a21,a22,a23.... format')
    args = parser.parse_args()
    array = args.array 
    array = array.split('.')
    array = [a.split(',') for a in array]
    # M = [[0,1,-1,2,0],[1,0,1,2,-1],[0,0,0,1,0],[1,1,0,5,-1]]
    # M = [[1,2,-1,0],[0,3,1,4],[-1,1,2,4],[2,3,1,5]]
    # M = [[2,2,3]]
    M = np.asarray(array,dtype=np.float32)
    print('Input matrix :\n {}'.format(M))
    print('Matrix after Gaussian Elimination :\n {}' .format(GE(M)))



    