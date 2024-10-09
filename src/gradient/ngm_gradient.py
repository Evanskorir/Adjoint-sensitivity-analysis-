import torch


class NGMGradient:
    def __init__(self, ngm_small_tensor: torch.Tensor,
                 contact_input_16: torch.Tensor,
                 contact_input_136: torch.Tensor):
        self.ngm_small_tensor = ngm_small_tensor
        self.contact_input_16 = contact_input_16
        self.contact_input_136 = contact_input_136

        self.ngm_small_grads_16 = None  # Gradients for contact_input_16
        self.ngm_small_grads_136 = None  # Gradients for contact_input_136

    def compute_gradients(self):
        self.ngm_small_grads_16 = self._compute_grad(self.contact_input_16)
        self.ngm_small_grads_136 = self._compute_grad(self.contact_input_136)

    def _compute_grad(self, contact_input: torch.Tensor):
        ngm_small_grads = torch.zeros(
            (self.ngm_small_tensor.size(0),
             contact_input.size(0),
             self.ngm_small_tensor.size(1)),
            dtype=torch.float32)

        for i in range(self.ngm_small_tensor.size(0)):
            for j in range(self.ngm_small_tensor.size(1)):
                grad = torch.autograd.grad(outputs=self.ngm_small_tensor[i, j],
                                           inputs=contact_input,
                                           retain_graph=True,
                                           create_graph=True)[0]

                # Apply grad.view(-1) only if contact_input is self.contact_input_16
                if contact_input.size(0) == 16 and grad is not None:
                    grad = grad.view(-1)  # Flatten for compatibility
                if grad is not None:
                    ngm_small_grads[i, :, j] = grad

        return ngm_small_grads

    def run(self):
        self.compute_gradients()
