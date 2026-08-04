import argparse
import time

import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(fn, reps, device):
    synchronize(device)
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / reps


def random_matrix(size, dtype, device, scale=0.05):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    real = torch.randn(size, size, dtype=real_dtype, device=device) * scale
    imag = torch.randn(size, size, dtype=real_dtype, device=device) * scale
    return torch.complex(real, imag).to(dtype)


def benchmark_redheffer(size, reps, dtype, device):
    Sm = [random_matrix(size, dtype, device) for _ in range(4)]
    Sn = [random_matrix(size, dtype, device) for _ in range(4)]
    identity = torch.eye(size, dtype=dtype, device=device)

    def explicit_inverse():
        tmp1 = torch.linalg.inv(identity - Sm[2] @ Sn[1])
        tmp2 = torch.linalg.inv(identity - Sn[1] @ Sm[2])
        return (
            Sn[0] @ tmp1 @ Sm[0],
            Sm[1] + Sm[3] @ tmp2 @ Sn[1] @ Sm[0],
            Sn[2] + Sn[0] @ tmp1 @ Sm[2] @ Sn[3],
            Sm[3] @ tmp2 @ Sn[3],
        )

    def batched_solve():
        A1 = identity - Sm[2] @ Sn[1]
        A2 = identity - Sn[1] @ Sm[2]
        A1_sol = torch.linalg.solve(A1, torch.hstack((Sm[0], Sm[2] @ Sn[3])))
        A2_sol = torch.linalg.solve(A2, torch.hstack((Sn[1] @ Sm[0], Sn[3])))
        return (
            Sn[0] @ A1_sol[:, :size],
            Sm[1] + Sm[3] @ A2_sol[:, :size],
            Sn[2] + Sn[0] @ A1_sol[:, size:],
            Sm[3] @ A2_sol[:, size:],
        )

    explicit_inverse()
    batched_solve()
    inverse_time = time_call(explicit_inverse, reps, device)
    solve_time = time_call(batched_solve, reps, device)
    print(f"size={size} reps={reps} dtype={dtype} device={device}")
    print(f"explicit inverse: {inverse_time:.6f} s/call")
    print(f"batched solve:    {solve_time:.6f} s/call")
    print(f"speedup:          {inverse_time / solve_time:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.complex128 if args.dtype == "complex128" else torch.complex64
    device = torch.device(args.device)
    benchmark_redheffer(args.size, args.reps, dtype, device)


if __name__ == "__main__":
    main()
