import numpy as np


def dense_to_coo(A):
    """
    Convert a dense matrix to COO triplets.
    """
    A = np.asarray(A)
    rows, cols = np.nonzero(A)
    data = A[rows, cols]
    return rows, cols, data


def coo_to_dense(rows, cols, data, shape):
    """
    Convert COO triplets back to a dense matrix.
    """
    A = np.zeros(shape, dtype=np.asarray(data).dtype)
    for row, col, value in zip(rows, cols, data):
        A[row, col] += value
    return A


def dense_to_csr(A):
    """
    Convert a dense matrix to a simple CSR representation.
    """
    A = np.asarray(A)
    rows, cols, data = dense_to_coo(A)
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    data = data[order]

    indptr = np.zeros(A.shape[0] + 1, dtype=int)
    for row in rows:
        indptr[row + 1] += 1
    indptr = np.cumsum(indptr)

    return data, cols.astype(int), indptr


def dense_to_csc(A):
    """
    Convert a dense matrix to a simple CSC representation.
    """
    A = np.asarray(A)
    rows, cols, data = dense_to_coo(A)
    order = np.lexsort((rows, cols))
    rows = rows[order]
    cols = cols[order]
    data = data[order]

    indptr = np.zeros(A.shape[1] + 1, dtype=int)
    for col in cols:
        indptr[col + 1] += 1
    indptr = np.cumsum(indptr)

    return data, rows.astype(int), indptr


def csr_matvec(data, indices, indptr, x):
    """
    Sparse matrix-vector multiply for CSR matrices.
    """
    x = np.asarray(x)
    y = np.zeros(len(indptr) - 1, dtype=np.result_type(data, x))

    for row in range(len(y)):
        start = indptr[row]
        end = indptr[row + 1]
        y[row] = np.dot(data[start:end], x[indices[start:end]])

    return y


def csc_matvec(data, indices, indptr, x):
    """
    Sparse matrix-vector multiply for CSC matrices.
    """
    x = np.asarray(x)
    y = np.zeros(np.max(indices) + 1 if len(indices) else 0, dtype=np.result_type(data, x))

    for col in range(len(indptr) - 1):
        start = indptr[col]
        end = indptr[col + 1]
        y[indices[start:end]] += data[start:end] * x[col]

    return y


def storage_efficiency(dense_shape, nnz):
    """
    Compare dense storage to COO / CSR / CSC storage counts.
    """
    m, n = dense_shape
    dense_entries = m * n
    return {
        "dense": dense_entries,
        "coo": 3 * nnz,
        "csr": 2 * nnz + m + 1,
        "csc": 2 * nnz + n + 1,
    }
