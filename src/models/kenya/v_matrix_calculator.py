import torch
from src.static.v_matrix_calculator_base import VMatrixCalculatorBase


class VMatrixCalculator(VMatrixCalculatorBase):
    def __init__(self, param: dict, n_age: int, states) -> None:
        super().__init__(param=param, n_age=n_age, states=states)

        self._get_v()

    def _get_v(self):
        idx = self._idx
        v = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))

        # E -> E (exposed to exposed)
        v[idx("e"), idx("e")] = self.parameters["omega"]
        # E -> A (exposed to asymptomatic)
        v[idx("a"), idx("e")] = -self.parameters["theta_i"] * self.parameters["omega"]
        # E -> M (exposed to symptomatic)
        v[idx("m"), idx("e")] = -(1 - self.parameters["theta_i"]) * self.parameters["omega"]
        # A -> A (asymptomatic to asymptomatic)
        v[idx("a"), idx("a")] = self.parameters["gamma_a"]
        # M -> M (symptomatic to symptomatic)
        v[idx("m"), idx("m")] = self.parameters["gamma_m"]
        # M -> H (symptomatic to hospitalized)
        v[idx("h"), idx("m")] = -self.parameters["kappa_i"]
        # H -> H (hospitalized to hospitalized)
        v[idx("h"), idx("h")] = self.parameters["zeta_i"]
        # H -> C (hospitalized to critical care)
        v[idx("c"), idx("h")] = -self.parameters["zeta_i"]
        # C -> H (critical care back to hospitalized)
        v[idx("h"), idx("c")] = -self.parameters["phi_i"]
        # C -> C (critical care to critical care)
        v[idx("c"), idx("c")] = self.parameters["lambda_i"]

        # Compute the inverse of the V matrix
        self.v_inv = torch.linalg.inv(v)

