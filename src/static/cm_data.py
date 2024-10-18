import torch
from src.static.dataloader import DataLoader


class CMData:
    def __init__(self, data: DataLoader, model: str):
        """
        Initialize the class with data and the specific model type.
        Args:
            data (DataLoader): An instance of DataLoader containing contact and age data.
            model (str): The model type that determines how the contact matrix is loaded.
        """
        self.data = data
        self.model = model

    def load_contact_matrix(self) -> torch.Tensor:
        """
        Load the appropriate contact matrix based on the model.
        For certain models (moghadas, seir, etc.), it uses the 'All' contact matrix,
        otherwise it sums Home, School, Work, and Other contact matrices.
        Returns: torch.Tensor: The loaded full contact matrix.
        """
        contact_data = self.data.contact_data
        if self.model in ["moghadas", "seir", "italy", "british_columbia"]:
            full_orig_cm = contact_data["All"]
        else:
            full_orig_cm = (
                    contact_data["Home"] +
                    contact_data["School"] +
                    contact_data["Work"] +
                    contact_data["Other"]
            )
        return full_orig_cm

    def calculate_full_contact_matrix(self) -> torch.Tensor:
        """
        Calculate and return the full contact matrix.

        Returns:
            torch.Tensor: The full contact matrix for the model.
        """
        full_orig_cm = self.load_contact_matrix()
        return full_orig_cm
