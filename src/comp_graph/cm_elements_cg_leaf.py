import torch


class CMElementsCGLeaf:
    def __init__(self, n_age: int, pop: torch.Tensor,
                 transformed_total_orig_cm: torch.Tensor):
        """
        Initialize the class with the number of age groups, population data,
        and contact matrix.
        Args:
            n_age (int): The number of age groups.
            pop (torch.Tensor): A tensor representing the population data.
            transformed_total_orig_cm (torch.Tensor): The original transformed total cm.
        """

        self.pop = pop
        self.n_age = n_age
        self.transformed_total_orig_cm = transformed_total_orig_cm

        # Get the indices of the upper triangular part
        self.upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)
        self.contact_input = None
        self.scale_value = None

    def run(self, scale: str):
        """
        Extract the upper triangular elements of the contact matrix and
        create a symmetric matrix.
        Returns: torch.Tensor: A tensor containing the symmetric matrix derived from the
            upper tri elements.
        """
        # Extract the upper tri elements of orig_cm and set requires_grad=True
        self.contact_input = self.transformed_total_orig_cm[
            self.upper_tri_idx[0], self.upper_tri_idx[1]
        ]
        if scale == "pop_sum":
            self.scale_value = torch.sum(self.pop)
            self.contact_input /= self.scale_value
        elif scale == "contact_sum":
            self.scale_value = torch.sum(self.contact_input)
            self.contact_input /= self.scale_value
        elif scale == "no_scale":
            pass
        else:
            raise ValueError(f"Unrecognized scale {scale}")
