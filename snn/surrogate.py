import torch


class _SurrogateSpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor, threshold: float, beta: float):
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        ctx.beta = beta
        return (membrane_potential >= threshold).to(membrane_potential.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane_potential,) = ctx.saved_tensors
        threshold = ctx.threshold
        beta = ctx.beta

        x = beta * (membrane_potential - threshold)
        grad = beta * torch.sigmoid(x) * (1.0 - torch.sigmoid(x))
        return grad_output * grad, None, None


class SurrogateSpike(torch.nn.Module):
    def __init__(self, threshold: float = 1.0, beta: float = 10.0):
        super().__init__()
        self.threshold = float(threshold)
        self.beta = float(beta)

    def forward(self, membrane_potential: torch.Tensor) -> torch.Tensor:
        return _SurrogateSpikeFn.apply(membrane_potential, self.threshold, self.beta)
