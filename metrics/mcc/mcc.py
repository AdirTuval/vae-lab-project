import numpy as np
from munkres import Munkres


def mcc(x : np.array, y : np.array) -> tuple:
    """Evaluate MCC
    Args:
        x: data to be sorted
        y: target data
    Returns:
        corr_sort: correlation matrix between x and y (after sorting)
        sort_idx: sorting index
        x_sort: x after sorting
    """

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    corr = np.corrcoef(y, x)
    corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i, :] = x[indexes[i][1], :]

    # Re-calculate correlation --------------------------------
    corr_sort = np.corrcoef(y, x_sort)
    corr_sort = corr_sort[0:dim, dim:]

    # return corr_sort, sort_idx, x_sort
    mcc = np.mean(np.abs(np.diag(corr_sort)))
    return mcc, corr_sort, sort_idx, x_sort

# x1 = 2 * y2

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    # x = rng.random((10, 2))
    x = np.array([[1, 2, 6, 8], [1, -4, -3, -4]])
    y = np.array([[-1, 4, 3, 4], [2, 4, 12, 16]])
    mcc(x, y)