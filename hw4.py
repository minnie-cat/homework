# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 07:35:13 2024

@author: htchen
"""
import numpy as np
import numpy.linalg as la

def scale_to_range(X: np.ndarray, to_range=(0,1), byrow = False):
    """
    Parameters
    ----------
    X: 
        1D or 2D array
    
    to_range: default to (0,1).
        Desired range of transformed data.
        
    byrow: default to False
        When working with a 2D array, true to perform row mapping; 
        otherwise, column mapping. Ignore if X is 1D. 
    
    ----------
    
    """
    a, b = to_range
    Y = np.zeros(X.shape)
    # write your code here
    if X.ndim == 1:
        X_min, X_max = np.min(X), np.max(X)
        Y = a + (X - X_min) * (b - a) / (X_max - X_min)
    
    elif X.ndim == 2:
        if byrow:
            for i in range(X.shape[0]):
                X_min, X_max = np.min(X[i, :]), np.max(X[i, :])
                Y[i, :] = a + (X[i, :] - X_min) * (b - a) / (X_max - X_min)
        else:
            for j in range(X.shape[1]):
                X_min, X_max = np.min(X[:, j]), np.max(X[:, j])
                Y[:, j] = a + (X[:, j] - X_min) * (b - a) / (X_max - X_min)
    return Y

print('test case 1:')
A = np.array([1, 2.5, 6, 4, 5])
print(f'A => \n{A}')
print(f'scale_to_range(A) => \n{scale_to_range(A)}\n\n')

print('test case 2:')
A = np.array([[1,12,3,7,8],
              [5,14,1,5,5],
              [4,11,4,1,2],
              [3,13,2,3,5],
              [2,15,6,3,2]])
print(f'A => \n{A}')
print(f'scale_to_range(A) => \n{scale_to_range(A)}\n\n')

print('test case 3:')
A = np.array([[1,2,3,4,5],
              [5,4,1,2,3],
              [3,5,4,1,2]])
print(f'A => \n{A}')
print(f'scale_to_range(A, byrow=True) => \n{scale_to_range(A, byrow=True)}\n\n')


"""
Expected output:
------------------
test case 1:
A => 
[1.  2.5 6.  4.  5. ]
scale_to_range(A) => 
[0.  0.3 1.  0.6 0.8]


test case 2:
A => 
[[ 1 12  3  7  8]
 [ 5 14  1  5  5]
 [ 4 11  4  1  2]
 [ 3 13  2  3  5]
 [ 2 15  6  3  2]]
scale_to_range(A) => 
[[0.   0.25 0.4  1.   1.  ]
 [1.   0.75 0.   0.67 0.5 ]
 [0.75 0.   0.6  0.   0.  ]
 [0.5  0.5  0.2  0.33 0.5 ]
 [0.25 1.   1.   0.33 0.  ]]


test case 3:
A => 
[[1 2 3 4 5]
 [5 4 1 2 3]
 [3 5 4 1 2]]
scale_to_range(A, byrow=True) => 
[[0.   0.25 0.5  0.75 1.  ]
 [1.   0.75 0.   0.25 0.5 ]
 [0.5  1.   0.75 0.   0.25]]   
S 
"""    

