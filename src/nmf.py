import torch


class ContactMatrixFactorizer:
    def __init__(self, n_age: int, pop: torch.Tensor,
                 transformed_total_orig_cm: torch.Tensor, n_components: int = 1,
                 max_iter: int = 1000, tol: float = 1e-8):
        self.pop = pop
        self.n_age = n_age
        self.transformed_total_orig_cm = transformed_total_orig_cm
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

        self.upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)

        self.W = None  # reduced matrix, 16 * 1
        self.H = None  # Coefficients matrix, 1 * 16
        self.contact_input_16 = None
        self.contact_input_136 = None
        self.scale_value = None

    def _scale_matrix(self, matrix: torch.Tensor, scale: str) -> torch.Tensor:
        if scale == "pop_sum":
            self.scale_value = torch.sum(self.pop)
            return matrix / self.scale_value
        elif scale == "contact_sum":
            self.scale_value = torch.sum(matrix)
            return matrix / self.scale_value
        elif scale == "no_scale":
            return matrix
        else:
            raise ValueError(f"Unrecognized scale method: {scale}")

    def _initialize_matrices(self) -> None:
        """Initialize W and H matrices randomly and set requires_grad=True."""
        self.W = torch.rand(self.transformed_total_orig_cm.shape[0],
                            self.n_components,
                            requires_grad=True)
        self.H = torch.rand(self.n_components,
                            self.transformed_total_orig_cm.shape[1],
                            requires_grad=True)

    def _perform_nmf(self, scale) -> None:
        """Perform the NMF algorithm to factorize the contact matrix."""
        for _ in range(self.max_iter):
            # Update H
            self.H.data = self.H.data * (self.W.T @ self.transformed_total_orig_cm) / \
                          (self.W.T @ self.W @ self.H + 1e-10)

            # Update W
            self.W.data = self.W.data * (self.transformed_total_orig_cm @ self.H.T) / \
                          (self.W @ self.H @ self.H.T + 1e-10)

            # Check for convergence
            if torch.norm(self.W @ self.H - self.transformed_total_orig_cm) < self.tol:
                break

        self.contact_input_16 = self.W

    def _reconstruct_matrix(self, scale: str) -> torch.Tensor:
        """Reconstruct the original matrix using W and H."""
        reconstructed_input = self.W @ self.H
        self.contact_input_136 = reconstructed_input[self.upper_tri_idx[0],
                                                     self.upper_tri_idx[1]]
        return self._scale_matrix(self.contact_input_136, scale)

    def run(self, scale: str) -> None:
        self._initialize_matrices()
        self._perform_nmf(scale)
        self.contact_input_136 = self._reconstruct_matrix(scale)

    def get_contact_inputs(self) -> torch.Tensor:
        return self.contact_input_16, self.contact_input_136



