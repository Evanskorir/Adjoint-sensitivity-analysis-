import torch


class SymmetricContactMatrixInput:
    def __init__(self, n_age: int, pop: torch.Tensor):
        """
        Initialize the class with the number of age groups, population data, and
        transformed original contact matrix.
        Args:
            n_age (int): The number of age groups.
            pop (torch.Tensor): A tensor representing the population data.
        """
        self.n_age = n_age
        self.pop = pop

    def create_symmetric_matrix(self, contact_matrix) -> torch.Tensor:
        """
        Create a symmetric matrix from the upper triangular elements of the contact matrix.
        Returns: torch.Tensor: A tensor containing the symmetric matrix derived from the
        upper triangular elements.
        """
        # Create a new matrix filled with zeros
        new_sym_contact_mtx = torch.zeros((self.n_age, self.n_age))

        # Get indices of upper triangular part
        upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)

        # Fill the upper triangular part with the extracted elements
        new_sym_contact_mtx[upper_tri_idx[0], upper_tri_idx[1]] = contact_matrix

        # Transpose the upper triangular matrix to fill the lower triangular part
        new_sym_contact_mtx_transposed = new_sym_contact_mtx + new_sym_contact_mtx.T - \
            torch.diag(new_sym_contact_mtx.diag())

        # Divide by the population to get the full symmetric contact matrix
        symmetric_contact_matrix = new_sym_contact_mtx_transposed / self.pop

        return symmetric_contact_matrix
