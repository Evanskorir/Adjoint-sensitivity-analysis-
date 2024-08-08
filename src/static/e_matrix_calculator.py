from scipy.linalg import block_diag
import torch


class EMatrixCalculator:
    def __init__(self, n_states: int, n_age: int) -> None:
        self.n_states = n_states
        self.n_age = n_age

        self._get_e()

    def _get_e(self):
        block = torch.zeros(self.n_states)
        block[0] = 1
        self.e = block
        for _ in range(1, self.n_age):
            self.e = block_diag(self.e, block)
            self.e = torch.tensor(self.e, dtype=torch.float32)
