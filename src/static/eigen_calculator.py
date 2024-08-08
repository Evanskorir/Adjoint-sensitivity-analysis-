import torch


class EigenCalculator:
    def __init__(self, ngm_small_tensor: torch.Tensor) -> None:
        self.ngm_small_tensor = ngm_small_tensor

        self.dominant_eig_vec = None
        self.dominant_eig_val = None

    def run(self):
        # Compute eigenvalues and eigenvectors
        eig_val, eig_vec = torch.linalg.eig(self.ngm_small_tensor)

        # Find the index of the eigenvalue with the largest magnitude
        max_eigval_idx = torch.abs(eig_val).argmax()

        # Extract the dominant eigenvalue and its corresponding eigenvector
        self.dominant_eig_val = eig_val[max_eigval_idx].real.item()
        self.dominant_eig_vec = eig_vec[:, max_eigval_idx].real
