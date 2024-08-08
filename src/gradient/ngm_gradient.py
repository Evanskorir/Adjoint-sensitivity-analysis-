import torch


class NGMGradient:
    def __init__(self, ngm_small_tensor: torch.Tensor,
                 contact_input: torch.Tensor):
        self.ngm_small_tensor = ngm_small_tensor
        self.contact_input = contact_input

        self.ngm_small_grads = None

    def run(self):
        ngm_small_grads = torch.zeros(
            (self.ngm_small_tensor.size(0),
             self.contact_input.size(0),
             self.ngm_small_tensor.size(1)),
            dtype=torch.float32)
        for i in range(self.ngm_small_tensor.size(0)):
            for j in range(self.ngm_small_tensor.size(1)):
                grad = torch.autograd.grad(outputs=self.ngm_small_tensor[i, j],
                                           inputs=self.contact_input,
                                           retain_graph=True,
                                           create_graph=True)[0]
                ngm_small_grads[i, :, j] = grad
        self.ngm_small_grads = ngm_small_grads    # 16 * 136 * 16
        