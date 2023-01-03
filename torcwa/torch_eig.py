import torch

'''
Pytorch 1.10.1
Complex domain eigen-decomposition with numerical stability
'''

class Eig(torch.autograd.Function):
    broadening_parameter = 1e-10

    @staticmethod
    def forward(ctx,x):
        ctx.input = x
        eigval, eigvec = torch.linalg.eig(x)
        ctx.eigval = eigval.cpu()
        ctx.eigvec = eigvec.cpu()
        return eigval, eigvec

    @staticmethod
    def backward(ctx,grad_eigval,grad_eigvec):
        eigval = ctx.eigval.to(grad_eigval)
        eigvec = ctx.eigvec.to(grad_eigvec)

        grad_eigval = torch.diag(grad_eigval)
        s = eigval.unsqueeze(-2) - eigval.unsqueeze(-1)

        # Lorentzian broadening: get small error but stabilizing the gradient calculation
        if Eig.broadening_parameter is not None:
            F = torch.conj(s)/(torch.abs(s)**2 + Eig.broadening_parameter)
        elif s.dtype == torch.complex64:
            F = torch.conj(s)/(torch.abs(s)**2 + 1.4e-45)
        elif s.dtype == torch.complex128:
            F = torch.conj(s)/(torch.abs(s)**2 + 4.9e-324)

        diag_indices = torch.linspace(0,F.shape[-1]-1,F.shape[-1],dtype=torch.int64)
        F[diag_indices,diag_indices] = 0.
        XH = torch.transpose(torch.conj(eigvec),-2,-1)
        tmp = torch.conj(F) * torch.matmul(XH, grad_eigvec)

        grad = torch.matmul(torch.matmul(torch.inverse(XH), grad_eigval + tmp), XH)
        if not torch.is_complex(ctx.input):
            grad = torch.real(grad)

        return grad