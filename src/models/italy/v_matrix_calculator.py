import torch


class VMatrixCalculator:
    def __init__(self, param: dict, n_states: int, n_age: int, states) -> None:
        self.n_states = n_states
        self.states = states
        self.n_age = n_age
        self.parameters = param

        self.i = {self.states[index]: index for index in range(0, self.n_states)}

        self._get_v()

    def _idx(self, state: str) -> torch.Tensor:
        return torch.arange(self.n_age * self.n_states) % self.n_states == self.i[state]

    def _get_v(self):
        idx = self._idx
        v = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))
        # e -> e
        v[idx("e"), idx("e")] = self.parameters["d_e"]
        # e -> ip
        v[idx("ip"), idx("e")] = -self.parameters["d_e"] * self.parameters["y_i"]
        # e -> is
        v[idx("is"), idx("e")] = -self.parameters["d_e"] * (1 - self.parameters["y_i"])
        # ip -> ip
        v[idx("ip"), idx("ip")] = self.parameters["d_p"]
        # ip -> ic
        v[idx("ic"), idx("ip")] = -self.parameters["d_p"]
        # ic -> ic
        v[idx("ic"), idx("ic")] = -self.parameters["d_c"]
        # is -> is
        v[idx("is"), idx("is")] = self.parameters["d_s"]

        self.v_inv = torch.linalg.inv(v)
