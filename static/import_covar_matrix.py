import numpy as np

def import_covar_matrix(dir,dim):
    """import covariance matrix from csv file"""
    
    try:
        file = open(dir,'rU')
    except IOError:
        # if covariance matrix doesn't exist, initialize to be id
        covar_matrix = np.identity(dim)
    else:
        file_lines = file.readlines()
        file.close()
        covar_matrix = np.zeros((dim,dim))
        for i in range(0,dim):
            temp = temp.rstrip()
            temp = file_lines[i].split(',')
            for j in range(0,dim):
                covar_matrix[i,j] = float(temp[j])
                
    return covar_matrix