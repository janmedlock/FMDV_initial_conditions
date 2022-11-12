'''Tools for sparse matrices.'''

import scipy.sparse


class csr_matrix(scipy.sparse.csr_matrix):
    '''A subclass of `scipy.sparse.csr_matrix()` with a `matvecs()`
    method.'''

    def matvecs(self, B, C):
        '''Compute the matrix multiplication `C += A @ B`, where
        `A` is a `scipy.sparse.csr_matrix()`, and
        `B` & `C` are `numpy.ndarray()`s.'''
        # Use the private function
        # `scipy.sparse._sparsetools.csr_matvecs()` so we can specify
        # the output array `C` to avoid the building of a new matrix
        # for the output.
        n_row, n_col = self.shape
        n_vecs = B.shape[1]
        scipy.sparse._sparsetools.csr_matvecs(
            n_row, n_col, n_vecs,
            self.indptr, self.indices, self.data,
            B.ravel(), C.ravel())
