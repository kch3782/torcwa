import numpy as np
import torch
from scipy.interpolate import interp1d

class aSiH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wavelength, dl = 0.005):
        # open material data
        open_name = 'Materials_data/aSiH.txt'
        f = open(open_name)
        data = f.readlines()
        f.close()
        nk_data = []
        for i in range(len(data)):
            _lamb0, _n, _k = data[i].split()
            nk_data.append([float(_lamb0), float(_n), float(_k)])
        nk_data = np.array(nk_data)

        n_interp = interp1d(nk_data[:,0],nk_data[:,1],kind='cubic')
        k_interp = interp1d(nk_data[:,0],nk_data[:,2],kind='cubic')

        wavelength_np = wavelength.detach().cpu().numpy()

        if wavelength_np < nk_data[0,0]:
            nk_value = nk_data[0,1]+1.j*nk_data[0,2]
        elif wavelength_np > nk_data[-1,0]:
            nk_value = nk_data[-1,1]+1.j*nk_data[-1,2]
        else:
            nk_value = n_interp(wavelength_np)+1.j*k_interp(wavelength_np)

        if wavelength_np-dl < nk_data[0,0]:
            nk_value_m = nk_data[0,1]+1.j*nk_data[0,2]
        elif wavelength_np-dl > nk_data[-1,0]:
            nk_value_m = nk_data[-1,1]+1.j*nk_data[-1,2]
        else:
            nk_value_m = n_interp(wavelength_np-dl)+1.j*k_interp(wavelength_np-dl)

        if wavelength_np+dl < nk_data[0,0]:
            nk_value_p = nk_data[0,1]+1.j*nk_data[0,2]
        elif wavelength_np+dl > nk_data[-1,0]:
            nk_value_p = nk_data[-1,1]+1.j*nk_data[-1,2]
        else:
            nk_value_p = n_interp(wavelength_np+dl)+1.j*k_interp(wavelength_np+dl)

        ctx.dnk_dl = (nk_value_p - nk_value_m) / (2*dl)
        
        return torch.tensor(nk_value,dtype=torch.complex128 if ((wavelength.dtype is torch.float64) or\
            (wavelength.dtype is torch.complex128)) else torch.complex64, device=wavelength.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad = 2*torch.real(torch.conj(grad_output)*ctx.dnk_dl)
        return grad