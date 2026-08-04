import unittest

import torch

import torcwa
from torcwa.torch_eig import Eig


class LinearSolveOptimizationTests(unittest.TestCase):
    def test_redheffer_product_matches_explicit_inverse_formula(self):
        dtype = torch.complex64
        device = torch.device("cpu")
        sim = torcwa.rcwa(
            freq=1 / 500.0,
            order=[0, 0],
            L=[300.0, 300.0],
            dtype=dtype,
            device=device,
        )

        torch.manual_seed(0)

        def small_matrix(scale):
            real = torch.randn(2, 2, dtype=torch.float32, device=device) * scale
            imag = torch.randn(2, 2, dtype=torch.float32, device=device) * scale
            return torch.complex(real, imag).to(dtype)

        Sm = [small_matrix(0.05) for _ in range(4)]
        Sn = [small_matrix(0.05) for _ in range(4)]
        Cm = [[small_matrix(0.05)], [small_matrix(0.05)]]
        Cn = [[small_matrix(0.05)], [small_matrix(0.05)]]

        actual_s, actual_c = sim._RS_prod(Sm, Sn, Cm, Cn)

        eye = torch.eye(2, dtype=dtype, device=device)
        tmp1 = torch.linalg.inv(eye - Sm[2] @ Sn[1])
        tmp2 = torch.linalg.inv(eye - Sn[1] @ Sm[2])
        expected_s = [
            Sn[0] @ tmp1 @ Sm[0],
            Sm[1] + Sm[3] @ tmp2 @ Sn[1] @ Sm[0],
            Sn[2] + Sn[0] @ tmp1 @ Sm[2] @ Sn[3],
            Sm[3] @ tmp2 @ Sn[3],
        ]
        expected_c = [
            [
                Cm[0][0] + Cm[1][0] @ tmp2 @ Sn[1] @ Sm[0],
                Cn[0][0] @ tmp1 @ Sm[0],
            ],
            [
                Cm[1][0] @ tmp2 @ Sn[3],
                Cn[1][0] + Cn[0][0] @ tmp1 @ Sm[2] @ Sn[3],
            ],
        ]

        for actual, expected in zip(actual_s, expected_s):
            self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))
        for actual_group, expected_group in zip(actual_c, expected_c):
            for actual, expected in zip(actual_group, expected_group):
                self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))

    def test_eig_backward_matches_explicit_inverse_formula(self):
        dtype = torch.complex128
        device = torch.device("cpu")
        torch.manual_seed(1)

        real = torch.randn(4, 4, dtype=torch.float64, device=device)
        imag = torch.randn(4, 4, dtype=torch.float64, device=device)
        x = torch.complex(real, imag).to(dtype).requires_grad_(True)

        eigval, eigvec = Eig.apply(x)
        grad_eigval = torch.randn_like(eigval)
        grad_eigvec = torch.randn_like(eigvec)
        actual = torch.autograd.grad(
            outputs=(eigval, eigvec),
            inputs=x,
            grad_outputs=(grad_eigval, grad_eigvec),
        )[0]

        s = eigval.detach().unsqueeze(-2) - eigval.detach().unsqueeze(-1)
        F = torch.conj(s) / (torch.abs(s) ** 2 + Eig.broadening_parameter)
        diag_indices = torch.arange(F.shape[-1], device=F.device)
        F[diag_indices, diag_indices] = 0.0
        XH = torch.transpose(torch.conj(eigvec.detach()), -2, -1)
        tmp = torch.conj(F) * torch.matmul(XH, grad_eigvec)
        expected = torch.matmul(
            torch.matmul(torch.inverse(XH), torch.diag(grad_eigval) + tmp),
            XH,
        )

        self.assertTrue(torch.allclose(actual, expected, rtol=1e-10, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
