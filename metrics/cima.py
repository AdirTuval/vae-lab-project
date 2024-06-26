import torch

def cima_kl_diagonality(jacobian: torch.Tensor):
    """
    Calculates the IMA constrast. Able to handle jax and Pytorch objects as well.

    Source: https://www.sciencedirect.com/science/article/pii/S0024379516303834
    Note that the trace term in Eq. (10) of the above paper cancels since the \hat{A} matrix in the paper
    (which corresponds to A=jacobian_t_jacobian rescaled as diag(A)^{-1/2}@A@diag(A)^{-1/2})
    is symmetric and positive definite with 1's on the main diagonal, so tr(\hat{A}-I)=0.
    Thus, it is not included in the code

    :param jacobian: jacobian matrix (jax or Pytorch)
    :return:
    """
    jacobian_t_jacobian = jacobian.T @ jacobian

    return 0.5 * (
        torch.linalg.slogdet(torch.diag(torch.diag(jacobian_t_jacobian)))[1]
        - torch.linalg.slogdet(jacobian_t_jacobian)[1]
    )