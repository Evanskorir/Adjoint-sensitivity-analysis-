from typing import Union

import torch


class CMCreator:
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

        self.cm = None

    def run(self, contact_matrix: torch.Tensor,
            scale_value: Union[torch.Tensor, None],
            scale: str):
        """
        Create a symmetric matrix from the upper triangular elements of the contact matrix.
        Returns: torch.Tensor: A tensor containing the symmetric matrix derived from the
        upper triangular elements.
        """
        if scale == "pop_sum" or scale == "contact_sum":
            cm_elements = contact_matrix * scale_value
        elif scale == "no_scale":
            cm_elements = contact_matrix
        else:
            raise ValueError(f"Unknown scale: {scale}")

        # Create a new matrix filled with zeros
        new_sym_contact_mtx = torch.zeros((self.n_age, self.n_age))

        # Get indices of upper triangular part
        upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)

        # Fill the upper triangular part with the extracted elements
        new_sym_contact_mtx[upper_tri_idx[0], upper_tri_idx[1]] = cm_elements

        # Transpose the upper triangular matrix to fill the lower triangular part
        new_sym_contact_mtx_transposed = (
                new_sym_contact_mtx +
                new_sym_contact_mtx.T -
                torch.diag(new_sym_contact_mtx.diag())
        )

        # Divide by the population to get the full symmetric contact matrix
        self.cm = new_sym_contact_mtx_transposed / self.pop


