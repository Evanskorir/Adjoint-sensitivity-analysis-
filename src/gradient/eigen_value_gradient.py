import torch


class EigenValueGradient:
    def __init__(self, ngm_small_tensor: torch.Tensor, dominant_eig_vec: torch.Tensor):
        self.ngm_small_tensor = ngm_small_tensor
        self.dominant_eig_vec = dominant_eig_vec

        self.eig_val_cm_grad = None

    def run(self, ngm_small_grads: torch.Tensor):
        # Reshape self.dominant_eig_vec to be a column vector
        eig_vec = self.dominant_eig_vec.view(-1, 1)  # Shape: [16, 1]

        # Compute A @ x: ngm_small_grads @ eig_vec for each slice
        a_dot_x = torch.matmul(ngm_small_grads, eig_vec).squeeze(dim=2)  # Shape: [16, 136]

        # Compute x.T @ A @ x
        # Reshape x.T to match the multiplication with B
        # get the eig_vec_transpose
        eig_vec_transpose = self.dominant_eig_vec.T
        self.eig_val_cm_grad = torch.matmul(eig_vec_transpose, a_dot_x)

