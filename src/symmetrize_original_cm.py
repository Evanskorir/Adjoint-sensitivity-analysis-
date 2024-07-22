import torch


class OriginalCMSymmetrization:
    def __init__(self, data):
        """
        Initialize the class with data.
        Args: data: An instance of DataLoader containing contact and age data.
        """
        self.data = data

    def calculate_full_transformed_cm(self) -> torch.Tensor:
        """
        Calculate and return the full transformed contact matrix.
        The full contact matrix is derived from the sum of various contact matrices
        (Home, School, Work, Other) and then symmetrized and transformed.
        Returns: torch.Tensor: The symmetrized and transformed full contact matrix.
        """
        full_orig_cm = self._calculate_full_orig_cm()
        transformed_orig_cm = self._transform_orig_cm(full_orig_cm)
        return transformed_orig_cm

    def _calculate_full_orig_cm(self) -> torch.Tensor:
        """
        Calculate the full contact matrix by summing up different contact matrices.
        Returns: torch.Tensor: The full contact matrix.
        """
        contact_data = self.data.contact_data
        full_orig_cm = (
            contact_data["Home"] +
            contact_data["School"] +
            contact_data["Work"] +
            contact_data["Other"]
        )
        return full_orig_cm

    def _transform_orig_cm(self, contact_matrix: torch.Tensor) -> torch.Tensor:
        """
        Transform and symmetrize the given contact matrix.
        The transformation involves multiplying by the age distribution and symmetrizing.
        Args: contact_matrix (torch.Tensor): The contact matrix to transform.
        Returns: torch.Tensor: The transformed and symmetrized contact matrix.
        """
        age_distribution = self.data.age_data.reshape((-1, 1))  # (16, 1) column vector
        symmetrized_orig_cm = ((contact_matrix * age_distribution) +
                              (contact_matrix * age_distribution).T) / 2
        return symmetrized_orig_cm


