import torch
from src.static.dataloader import DataLoader
from src.static.cm_data import CMData


class CGLeafPreparator:
    def __init__(self, data: DataLoader, model: str):
        """
        Initialize the class with data and model.
        Args:
            data (DataLoader): An instance of DataLoader containing contact and age data.
            model (str): The model to be used.
        """
        self.data = data
        self.model = model
        self.cm_data = CMData(data, model)
        self.transformed_total_orig_cm = None

    def run(self):
        """
        Calculate and return the full transformed contact matrix.
        The full contact matrix is derived from the sum of various contact matrices
        (Home, School, Work, Other) and then symmetrized and transformed.
        Returns: torch.Tensor: The symmetrized and transformed full contact matrix.
        """
        full_orig_cm = self.cm_data.calculate_full_contact_matrix()
        self.transformed_total_orig_cm = self._transform_orig_total_cm(full_orig_cm)

    def _transform_orig_total_cm(self, contact_matrix: torch.Tensor) -> torch.Tensor:
        """
        Transform and symmetrize the given contact matrix.
        The transformation involves multiplying by the age distribution and symmetrizing.
        Args: contact_matrix (torch.Tensor): The contact matrix to transform.
        Returns: torch.Tensor: The transformed and symmetrized contact matrix.
        """
        age_distribution = self.data.age_data.reshape((-1, 1))  # (16, 1) column vector
        symmetrized_orig_total_cm = ((contact_matrix * age_distribution) +
                                     (contact_matrix * age_distribution).T
                                    ) / 2
        return symmetrized_orig_total_cm
