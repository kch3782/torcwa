import torch
from .torch_eig import Eig

pi = 3.141592652589793

class rcwa:
    # Simulation setting
    def __init__(self,freq,order,L,*,
            dtype=torch.complex64,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            stable_eig_grad=True,
            avoid_Pinv_instability=False,
            max_Pinv_instability=0.005
        ):

        '''
            Rigorous Coupled Wave Analysis
            - Lorentz-Heaviside units
            - Speed of light: 1
            - Time harmonics notation: exp(-jωt)

            Parameters
            - freq: simulation frequency (unit: length^-1)
            - order: Fourier order [x_order (int), y_order (int)]
            - L: Lattice constant [Lx, Ly] (unit: length)

            Keyword Parameters
            - dtype: simulation data type (only torch.complex64 and torch.complex128 are allowed.)
            - device: simulation device (only torch.device('cpu') and torch.device('cuda') are allowed.)
            - stable_eig_grad: stabilize gradient calculation of eigendecompsition (default as True)
            - avoid_Pinv_instability: avoid instability of P inverse (P: H to E) (default as False)
            - max_Pinv_instability: allowed maximum instability value for P inverse (default as 0.005 if avoid_Pinv_instability is True)
        '''

        # Hardware
        if dtype != torch.complex64 and dtype != torch.complex128:
            print('Invalid simulation data type. Set as torch.complex64.')
            self._dtype = torch.complex64
        else:
            self._dtype = dtype
        self._device = device

        # Stabilize the gradient of eigendecomposition
        if stable_eig_grad is False:
            self.stable_eig_grad = False
        else:
            self.stable_eig_grad = True

        # Stability setting for inverse matrix of P and Q
        if avoid_Pinv_instability is True:
            self.avoid_Pinv_instability = True
            self.max_Pinv_instability = max_Pinv_instability
            self.Pinv_instability = []
            self.Qinv_instability = []
        else:
            self.avoid_Pinv_instability = False
            self.max_Pinv_instability = None
            self.Pinv_instability = None
            self.Qinv_instability = None

        # Simulation parameters
        self.freq = torch.as_tensor(freq,dtype=self._dtype,device=self._device) # unit^-1
        self.omega = 2*pi*freq # same as k0a
        self.L = torch.as_tensor(L,dtype=self._dtype,device=self._device)

        # Fourier order
        self.order = order
        self.order_x = torch.linspace(-self.order[0],self.order[0],2*self.order[0]+1,dtype=torch.int64,device=self._device)
        self.order_y = torch.linspace(-self.order[1],self.order[1],2*self.order[1]+1,dtype=torch.int64,device=self._device)
        self.order_N = len(self.order_x)*len(self.order_y)

        # Lattice vector
        self.L = L  # unit
        self.Gx_norm = 1/(L[0]*self.freq)
        self.Gy_norm = 1/(L[1]*self.freq)

        # Input and output layer (Default: free space)
        self.eps_in = torch.tensor(1.,dtype=self._dtype,device=self._device)
        self.mu_in = torch.tensor(1.,dtype=self._dtype,device=self._device)
        self.eps_out = torch.tensor(1.,dtype=self._dtype,device=self._device)
        self.mu_out = torch.tensor(1.,dtype=self._dtype,device=self._device)

        # Internal layers
        self.layer_N = 0  # total number of layers
        self.thickness = []

        self.eps_conv = []
        self.mu_conv = []

        # Internal layer eigenmodes
        self.P = []
        self.Q = []
        self.kz_norm = []
        self.E_eigvec = []
        self.H_eigvec = []

        # Internal layer mode coupling coefficiencts
        self.Cf = []
        self.Cb = []

        # Single layer scattering matrices
        self.layer_S11 = []
        self.layer_S21 = []
        self.layer_S12 = []
        self.layer_S22 = []

    def add_input_layer(self,eps=1.,mu=1.):
        '''
            Add input layer
            - If this function is not used, simulation will be performed under free space input layer.

            Parameters
            - eps: relative permittivity
            - mu: relative permeability
        '''

        self.eps_in = torch.as_tensor(eps,dtype=self._dtype,device=self._device)
        self.mu_in = torch.as_tensor(mu,dtype=self._dtype,device=self._device)
        self.Sin = []

    def add_output_layer(self,eps=1.,mu=1.):
        '''
            Add output layer
            - If this function is not used, simulation will be performed under free space output layer.

            Parameters
            - eps: relative permittivity
            - mu: relative permeability
        '''

        self.eps_out = torch.as_tensor(eps,dtype=self._dtype,device=self._device)
        self.mu_out = torch.as_tensor(mu,dtype=self._dtype,device=self._device)
        self.Sout = []

    def set_incident_angle(self,inc_ang,azi_ang,angle_layer='input'):
        '''
            Set incident angle

            Parameters
            - inc_ang: incident angle (unit: radian)
            - azi_ang: azimuthal angle (unit: radian)
            - angle_layer: reference layer to calculate angle ('input' / 'output')
        '''

        self.inc_ang = torch.as_tensor(inc_ang,dtype=self._dtype,device=self._device)
        self.azi_ang = torch.as_tensor(azi_ang,dtype=self._dtype,device=self._device)

        if angle_layer != 'input' and angle_layer != 'output':
            print('Invalid angle layer. Set as input layer.')
            angle_layer = 'input'

        self.angle_layer = angle_layer

        self._kvectors()

    def add_layer(self,thickness,eps=1.,mu=1.):
        '''
            Add internal layer

            Parameters
            - thickness: layer thickness (unit: length)
            - eps: relative permittivity
            - mu: relative permeability
        '''

        is_eps_homogenous = (type(eps) == float) or (type(eps) == complex) or (eps.dim() == 0) or ((eps.dim() == 1) and eps.shape[0] == 1)
        is_mu_homogenous = (type(mu) == float) or (type(mu) == complex) or (mu.dim() == 0) or ((mu.dim() == 1) and mu.shape[0] == 1)
        
        self.eps_conv.append(eps*torch.eye(self.order_N,dtype=self._dtype,device=self._device) if is_eps_homogenous else self._material_conv(eps))
        self.mu_conv.append(mu*torch.eye(self.order_N,dtype=self._dtype,device=self._device) if is_mu_homogenous else self._material_conv(mu))

        self.layer_N += 1
        self.thickness.append(thickness)

        if is_eps_homogenous and is_mu_homogenous:
            self._eigen_decomposition_homogenous(eps,mu)
        else:
            self._eigen_decomposition()

        self._solve_layer_smatrix()

    # Solve simulation
    def solve_global_smatrix(self):
        '''
            Solve global S-matrix
        '''

        # Initialization
        if self.layer_N > 0:
            S11 = self.layer_S11[0]
            S21 = self.layer_S21[0]
            S12 = self.layer_S12[0]
            S22 = self.layer_S22[0]
            C = [[self.Cf[0]], [self.Cb[0]]]
        else:
            S11 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
            S21 = torch.zeros(2*self.order_N,dtype=self._dtype,device=self._device)
            S12 = torch.zeros(2*self.order_N,dtype=self._dtype,device=self._device)
            S22 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
            C = [[], []]

        # Connection
        for i in range(self.layer_N-1):
            [S11, S21, S12, S22], C = self._RS_prod(Sm=[S11, S21, S12, S22],
                Sn=[self.layer_S11[i+1], self.layer_S21[i+1], self.layer_S12[i+1], self.layer_S22[i+1]],
                Cm=C, Cn=[[self.Cf[i+1]], [self.Cb[i+1]]])

        if hasattr(self,'Sin'):
            # input layer coupling
            [S11, S21, S12, S22], C = self._RS_prod(Sm=[self.Sin[0], self.Sin[1], self.Sin[2], self.Sin[3]],
                Sn=[S11, S21, S12, S22],
                Cm=[[],[]], Cn=C)

        if hasattr(self,'Sout'):
            # output layer coupling
            [S11, S21, S12, S22], C = self._RS_prod(Sm=[S11, S21, S12, S22],
                Sn=[self.Sout[0], self.Sout[1], self.Sout[2], self.Sout[3]],
                Cm=C, Cn=[[],[]])

        self.S = [S11, S21, S12, S22]
        self.C = C

    # Returns
    def diffraction_angle(self,orders,*,layer='output',unit='radian'):
        '''
            Diffraction angles for the selected orders

            Parameters
            - orders: selected diffraction orders (Recommended shape: Nx2)
            - layer: selected layer ('input' / 'output')
            - unit: unit of the output angles ('radian' / 'degree')

            Return
            - inclination angle (torch.Tensor), azimuthal angle (torch.Tensor)
        '''

        orders = torch.as_tensor(orders,dtype=torch.int64,device=self._device).reshape([-1,2])

        if layer != 'input' and layer != 'output':
            print('Invalid layer. Set as output layer.')
            layer = 'output'

        if unit != 'radian' and unit != 'degree':
            print('Invalid unit. Set as radian.')
            unit = 'radian'

        # Matching indices
        orders[orders[:,0]<-self.order[0],0] = int(-self.order[0])
        orders[orders[:,0]>self.order[0],0] = int(self.order[0])
        orders[orders[:,1]<-self.order[1],1] = int(-self.order[1])
        orders[orders[:,1]>self.order[1],1] = int(self.order[1])
        order_indices = len(self.order_y)*(orders[:,0]+int(self.order[0])) + orders[:,1]+int(self.order[1])

        eps = self.eps_in if layer == 'input' else self.eps_out
        mu = self.mu_in if layer == 'input' else self.mu_out

        kx_norm = self.Kx_norm_dn[order_indices]
        ky_norm = self.Ky_norm_dn[order_indices]
        kt_norm = torch.sqrt(kx_norm**2 + ky_norm**2)
        inc_angle = torch.real(torch.asin(kt_norm/torch.real(torch.sqrt(eps*mu))))
        azi_angle = torch.atan2(torch.real(ky_norm),torch.real(kx_norm))

        if unit == 'degree':
            inc_angle = (180./pi) * inc_angle
            azi_angle = (180./pi) * azi_angle

        return inc_angle, azi_angle

    def return_layer(self,layer_num,nx=100,ny=100):
        '''
            Return spatial distributions of eps and mu for the selected layer.
            The eps and mu are recovered from the trucated Fourier orders.

            Parameters
            - layer_num: selected layer (int)
            - nx: x-direction grid number (int)
            - ny: y-direction grid number (int)

            Return
            - eps_recover (torch.Tensor), mu_recover (torch.Tensor)
        '''

        eps_fft = torch.zeros([nx,ny],dtype=self._dtype,device=self._device)
        mu_fft = torch.zeros([nx,ny],dtype=self._dtype,device=self._device)
        for i in range(-2*self.order[0],2*self.order[0]+1):
            for j in range(-2*self.order[1],2*self.order[1]+1):
                if i >= 0 and j >= 0:
                    eps_fft[i,j] = self.eps_conv[layer_num][i*(2*self.order[1]+1)+j,0]
                    mu_fft[i,j] = self.mu_conv[layer_num][i*(2*self.order[1]+1)+j,0]
                elif i >= 0 and j < 0:
                    eps_fft[i,j] = self.eps_conv[layer_num][i*(2*self.order[1]+1),-j]
                    mu_fft[i,j] = self.mu_conv[layer_num][i*(2*self.order[1]+1),-j]
                elif i < 0 and j >= 0:
                    eps_fft[i,j] = self.eps_conv[layer_num][j,-i*(2*self.order[1]+1)]
                    mu_fft[i,j] = self.mu_conv[layer_num][j,-i*(2*self.order[1]+1)]
                else:
                    eps_fft[i,j] = self.eps_conv[layer_num][0,-i*(2*self.order[1]+1)-j]
                    mu_fft[i,j] = self.mu_conv[layer_num][0,-i*(2*self.order[1]+1)-j]

        eps_recover = torch.fft.ifftn(eps_fft)*nx*ny
        mu_recover = torch.fft.ifftn(mu_fft)*nx*ny

        return eps_recover, mu_recover
    
    def S_parameters(self,orders,*,direction='forward',port='transmission',polarization='xx',ref_order=[0,0],power_norm=True):
        '''
            Return S-parameters.

            Parameters
            - orders: selected orders (Recommended shape: Nx2)

            - direction: set the direction of light propagation ('forward' / 'backward')
            - port: set the direction of light propagation ('transmission' / 'reflection')
            - polarization: set the input and output polarization of light ('xx' / 'yx' / 'xy' / 'yy' (output,input))
            - ref_order: set the reference for calculating S-parameters (Recommended shape: Nx2)
            - power_norm: if set as True, the absolute square of S-parameters are corresponds to the ratio of power.

            Return
            - S-parameters (torch.Tensor)
        '''

        orders = torch.as_tensor(orders,dtype=torch.int64,device=self._device).reshape([-1,2])

        if not (direction == 'forward' or direction == 'backward'):
            print('Invalid propagation direction. Set as forward.')
            direction = 'forward'

        if not (port == 'transmission' or port == 'reflection'):
            print('Invalid port. Set as tramsmission.')
            port = 'transmission'

        if not (polarization == 'xx' or polarization == 'yx' or polarization == 'xy' or polarization == 'yy'):
            print('Invalid polarization. Set as xx.')
            polarization = 'xx'

        ref_order = torch.as_tensor(ref_order,dtype=torch.int64,device=self._device).reshape([1,2])

        # Matching order indices
        orders[orders[:,0]<-self.order[0],0] = int(-self.order[0])
        orders[orders[:,0]>self.order[0],0] = int(self.order[0])
        orders[orders[:,1]<-self.order[1],1] = int(-self.order[1])
        orders[orders[:,1]>self.order[1],1] = int(self.order[1])
        order_indices = len(self.order_y)*(orders[:,0]+int(self.order[0])) + orders[:,1]+int(self.order[1])

        # Matching reference order index
        ref_order[ref_order[:,0]<-self.order[0],0] = int(-self.order[0])
        ref_order[ref_order[:,0]>self.order[0],0] = int(self.order[0])
        ref_order[ref_order[:,1]<-self.order[1],1] = int(-self.order[1])
        ref_order[ref_order[:,1]>self.order[1],1] = int(self.order[1])
        ref_order_index = len(self.order_y)*(ref_order[:,0]+int(self.order[0])) + ref_order[:,1]+int(self.order[1])

        # Matching order indices with polarization
        if polarization == 'yx' or polarization == 'yy':
            order_indices = order_indices + self.order_N
        if polarization == 'xy' or polarization == 'yy':
            ref_order_index = ref_order_index + self.order_N

        # power normalization factor
        if power_norm:
            Kz_norm_dn_in = torch.real(torch.sqrt(self.eps_in*self.mu_in - self.Kx_norm_dn**2 - self.Ky_norm_dn**2))
            Kz_norm_dn_in = torch.hstack((Kz_norm_dn_in,Kz_norm_dn_in))
            Kz_norm_dn_out = torch.real(torch.sqrt(self.eps_out*self.mu_out - self.Kx_norm_dn**2 - self.Ky_norm_dn**2))
            Kz_norm_dn_out = torch.hstack((Kz_norm_dn_out,Kz_norm_dn_out))
            Kx_norm_dn = torch.hstack((torch.real(self.Kx_norm_dn),torch.real(self.Kx_norm_dn)))
            Ky_norm_dn = torch.hstack((torch.real(self.Ky_norm_dn),torch.real(self.Ky_norm_dn)))

            if polarization == 'xx':
                if direction == 'forward' and port == 'transmission':
                    normalization = torch.sqrt((1+(Kx_norm_dn[order_indices]/Kz_norm_dn_out[order_indices])**2)/(1+(Kx_norm_dn[ref_order_index]/Kz_norm_dn_in[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_out[order_indices]/Kz_norm_dn_in[ref_order_index])
                elif direction == 'forward' and port == 'reflection':
                    normalization = torch.sqrt((1+(Kx_norm_dn[order_indices]/Kz_norm_dn_in[order_indices])**2)/(1+(Kx_norm_dn[ref_order_index]/Kz_norm_dn_in[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_in[order_indices]/Kz_norm_dn_in[ref_order_index])
                elif direction == 'backward' and port == 'reflection':
                    normalization = torch.sqrt((1+(Kx_norm_dn[order_indices]/Kz_norm_dn_out[order_indices])**2)/(1+(Kx_norm_dn[ref_order_index]/Kz_norm_dn_out[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_out[order_indices]/Kz_norm_dn_out[ref_order_index])
                elif direction == 'backward' and port == 'transmission':
                    normalization = torch.sqrt((1+(Kx_norm_dn[order_indices]/Kz_norm_dn_in[order_indices])**2)/(1+(Kx_norm_dn[ref_order_index]/Kz_norm_dn_out[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_in[order_indices]/Kz_norm_dn_out[ref_order_index])
    
            elif polarization == 'xy':
                if direction == 'forward' and port == 'transmission':
                    normalization = torch.sqrt((1+(Kx_norm_dn[order_indices]/Kz_norm_dn_out[order_indices])**2)/(1+(Ky_norm_dn[ref_order_index]/Kz_norm_dn_in[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_out[order_indices]/Kz_norm_dn_in[ref_order_index])
                elif direction == 'forward' and port == 'reflection':
                    normalization = torch.sqrt((1+(Kx_norm_dn[order_indices]/Kz_norm_dn_in[order_indices])**2)/(1+(Ky_norm_dn[ref_order_index]/Kz_norm_dn_in[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_in[order_indices]/Kz_norm_dn_in[ref_order_index])
                elif direction == 'backward' and port == 'reflection':
                    normalization = torch.sqrt((1+(Kx_norm_dn[order_indices]/Kz_norm_dn_out[order_indices])**2)/(1+(Ky_norm_dn[ref_order_index]/Kz_norm_dn_out[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_out[order_indices]/Kz_norm_dn_out[ref_order_index])
                elif direction == 'backward' and port == 'transmission':
                    normalization = torch.sqrt((1+(Kx_norm_dn[order_indices]/Kz_norm_dn_in[order_indices])**2)/(1+(Ky_norm_dn[ref_order_index]/Kz_norm_dn_out[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_in[order_indices]/Kz_norm_dn_out[ref_order_index])

            elif polarization == 'yx':
                if direction == 'forward' and port == 'transmission':
                    normalization = torch.sqrt((1+(Ky_norm_dn[order_indices]/Kz_norm_dn_out[order_indices])**2)/(1+(Kx_norm_dn[ref_order_index]/Kz_norm_dn_in[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_out[order_indices]/Kz_norm_dn_in[ref_order_index])
                elif direction == 'forward' and port == 'reflection':
                    normalization = torch.sqrt((1+(Ky_norm_dn[order_indices]/Kz_norm_dn_in[order_indices])**2)/(1+(Kx_norm_dn[ref_order_index]/Kz_norm_dn_in[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_in[order_indices]/Kz_norm_dn_in[ref_order_index])
                elif direction == 'backward' and port == 'reflection':
                    normalization = torch.sqrt((1+(Ky_norm_dn[order_indices]/Kz_norm_dn_out[order_indices])**2)/(1+(Kx_norm_dn[ref_order_index]/Kz_norm_dn_out[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_out[order_indices]/Kz_norm_dn_out[ref_order_index])
                elif direction == 'backward' and port == 'transmission':
                    normalization = torch.sqrt((1+(Ky_norm_dn[order_indices]/Kz_norm_dn_in[order_indices])**2)/(1+(Kx_norm_dn[ref_order_index]/Kz_norm_dn_out[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_in[order_indices]/Kz_norm_dn_out[ref_order_index])
            
            elif polarization == 'yy':
                if direction == 'forward' and port == 'transmission':
                    normalization = torch.sqrt((1+(Ky_norm_dn[order_indices]/Kz_norm_dn_out[order_indices])**2)/(1+(Ky_norm_dn[ref_order_index]/Kz_norm_dn_in[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_out[order_indices]/Kz_norm_dn_in[ref_order_index])
                elif direction == 'forward' and port == 'reflection':
                    normalization = torch.sqrt((1+(Ky_norm_dn[order_indices]/Kz_norm_dn_in[order_indices])**2)/(1+(Ky_norm_dn[ref_order_index]/Kz_norm_dn_in[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_in[order_indices]/Kz_norm_dn_in[ref_order_index])
                elif direction == 'backward' and port == 'reflection':
                    normalization = torch.sqrt((1+(Ky_norm_dn[order_indices]/Kz_norm_dn_out[order_indices])**2)/(1+(Ky_norm_dn[ref_order_index]/Kz_norm_dn_out[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_out[order_indices]/Kz_norm_dn_out[ref_order_index])
                elif direction == 'backward' and port == 'transmission':
                    normalization = torch.sqrt((1+(Ky_norm_dn[order_indices]/Kz_norm_dn_in[order_indices])**2)/(1+(Ky_norm_dn[ref_order_index]/Kz_norm_dn_out[ref_order_index])**2))
                    normalization = normalization * torch.sqrt(Kz_norm_dn_in[order_indices]/Kz_norm_dn_out[ref_order_index])
        
        else:
            normalization = 1

        # Get S-parameters
        if direction == 'forward' and port == 'transmission':
            S = self.S[0][order_indices,ref_order_index] * normalization
        elif direction == 'forward' and port == 'reflection':
            S = self.S[1][order_indices,ref_order_index] * normalization
        elif direction == 'backward' and port == 'reflection':
            S = self.S[2][order_indices,ref_order_index] * normalization
        elif direction == 'backward' and port == 'transmission':
            S = self.S[3][order_indices,ref_order_index] * normalization

        S = torch.where(torch.isinf(S),torch.zeros_like(S),S)
        S = torch.where(torch.isnan(S),torch.zeros_like(S),S)

        return S

    def source_planewave(self,*,amplitude=[1.,0.],direction='forward'):
        '''
            Generate planewave

            Paramters
            - amplitude: amplitudes at the matched diffraction orders [Ex_amp, Ey_amp] (list / np.ndarray / torch.Tensor) (Recommended shape: 1x2)
            - direction: incident direction ('forward' / 'backward')
        '''
        self.source_fourier(amplitude=amplitude,orders=[0,0],direction=direction)

    def source_fourier(self,*,amplitude,orders,direction='forward'):
        '''
            Generate Fourier source

            Paramters
            - amplitude: amplitudes at the matched diffraction orders [([Ex_amp, Ey_amp] at orders[0]), ..., ...]
                (list / np.ndarray / torch.Tensor) (Recommended shape: Nx2)
            - orders: diffraction orders (list / np.ndarray / torch.Tensor) (Recommended shape: Nx2)
            - direction: incident direction ('forward' / 'backward')
        '''
        amplitude = torch.as_tensor(amplitude,dtype=self._dtype,device=self._device).reshape([-1,2])
        orders = torch.as_tensor(orders,dtype=torch.int64,device=self._device).reshape([-1,2])

        if direction != 'forward' and direction != 'backward':
            print('Invalid source direction. Set as forward.')
            direction = 'forward'

        # Matching indices
        orders[orders[:,0]<-self.order[0],0] = int(-self.order[0])
        orders[orders[:,0]>self.order[0],0] = int(self.order[0])
        orders[orders[:,1]<-self.order[1],1] = int(-self.order[1])
        orders[orders[:,1]>self.order[1],1] = int(self.order[1])
        order_indices = len(self.order_y)*(orders[:,0]+int(self.order[0])) + orders[:,1]+int(self.order[1])

        self.source_direction = direction
        self.E_i = torch.zeros([2*self.order_N,1],dtype=self._dtype,device=self._device)
        self.E_i[order_indices,0] = amplitude[:,0]
        self.E_i[order_indices+self.order_N,0] = amplitude[:,1]

    def field_xz(self,x_axis,z_axis,y):
        '''
            XZ-plane field distribution.
            Returns the field at the specific y point.

            Paramters
            - x_axis: x-direction sampling coordinates (torch.Tensor)
            - z_axis: z-direction sampling coordinates (torch.Tensor)
            - y: selected y point

            Return
            - [Ex, Ey, Ez] (list[torch.Tensor]), [Hx, Hy, Hz] (list[torch.Tensor])
        '''

        if type(x_axis) != torch.Tensor or type(z_axis) != torch.Tensor:
            print("x and z axis must be torch.Tensor type. Return None.")
            return None

        x_axis = x_axis.reshape([-1,1,1])

        Kx_norm = self.Kx_norm
        Ky_norm = self.Ky_norm

        Ex_split = []
        Ey_split = []
        Ez_split = []
        Hx_split = []
        Hy_split = []
        Hz_split = []

        # layer number
        zp = torch.zeros(len(self.thickness),device=self._device)
        zm = torch.zeros(len(self.thickness),device=self._device)
        layer_num = torch.zeros([len(z_axis)],dtype=torch.int64)
        layer_num[z_axis<0.] = -1

        for ti in range(len(self.thickness)):
            zp[ti:] += self.thickness[ti]
        zm[1:] = zp[0:-1]

        for bi in range(len(zp)):
            layer_num[z_axis>zp[bi]] += 1

        prev_layer_num = -2
        for zi in range(len(z_axis)):
            # Input and output layers
            if layer_num[zi] == -1 or layer_num[zi] == self.layer_N:
                Kx_norm_dn = self.Kx_norm_dn
                Ky_norm_dn = self.Ky_norm_dn

                if layer_num[zi] == -1:
                    z_prop = z_axis[zi] if z_axis[zi] <= 0. else 0.
                    if layer_num[zi] != prev_layer_num:
                        eps = self.eps_in if hasattr(self,'eps_in') else 1.
                        mu = self.mu_in if hasattr(self,'mu_in') else 1.
                        Vi = self.Vi if hasattr(self,'Vi') else self.Vf
                        Kz_norm_dn = torch.sqrt(eps*mu - Kx_norm_dn**2 - Ky_norm_dn**2)
                        Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)>0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])
                        Kz_norm_dn = torch.vstack((Kz_norm_dn,Kz_norm_dn))
                elif layer_num[zi] == self.layer_N:
                    if len(zp) == 0:
                        z_prop = z_axis[zi]
                    else:
                        z_prop = z_axis[zi]-zp[-1] if z_axis[zi]-zp[-1] >= 0. else 0.
                    if layer_num[zi] != prev_layer_num:
                        eps = self.eps_out if hasattr(self,'eps_in') else 1.
                        mu = self.mu_out if hasattr(self,'mu_in') else 1.        
                        Vo = self.Vo if hasattr(self,'Vo') else self.Vf
                        Kz_norm_dn = torch.sqrt(eps*mu - Kx_norm_dn**2 - Ky_norm_dn**2)
                        Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])
                        Kz_norm_dn = torch.vstack((Kz_norm_dn,Kz_norm_dn))

                # Phase
                z_phase = torch.exp(1.j*self.omega*Kz_norm_dn*z_prop)

                # Fourier domain fields
                # [diffraction order]
                if layer_num[zi] == -1 and self.source_direction == 'forward':
                    Exy_p = self.E_i*z_phase
                    Hxy_p = torch.matmul(Vi,Exy_p)
                    Exy_m = torch.matmul(self.S[1],self.E_i)*torch.conj(z_phase)
                    Hxy_m = torch.matmul(-Vi,Exy_m)
                elif layer_num[zi] == -1 and self.source_direction == 'backward':
                    Exy_p = torch.zeros_like(self.E_i)
                    Hxy_p = torch.zeros_like(self.E_i)
                    Exy_m = torch.matmul(self.S[3],self.E_i)*torch.conj(z_phase)
                    Hxy_m = torch.matmul(-Vi,Exy_m)
                elif layer_num[zi] == self.layer_N and self.source_direction == 'forward':
                    Exy_p = torch.matmul(self.S[0],self.E_i)*z_phase
                    Hxy_p = torch.matmul(Vo,Exy_p)
                    Exy_m = torch.zeros_like(self.E_i)
                    Hxy_m = torch.zeros_like(self.E_i)
                elif layer_num[zi] == self.layer_N and self.source_direction == 'backward':
                    Exy_p = torch.matmul(self.S[2],self.E_i)*z_phase
                    Hxy_p = torch.matmul(Vo,Exy_p)
                    Exy_m = self.E_i*torch.conj(z_phase)
                    Hxy_m = torch.matmul(-Vo,Exy_m)

                Ex_mn = Exy_p[:self.order_N] + Exy_m[:self.order_N]
                Ey_mn = Exy_p[self.order_N:] + Exy_m[self.order_N:]
                Hz_mn = torch.matmul(Kx_norm,Ey_mn)/mu - torch.matmul(Ky_norm,Ex_mn)/mu
                Hx_mn = Hxy_p[:self.order_N] + Hxy_m[:self.order_N]
                Hy_mn = Hxy_p[self.order_N:] + Hxy_m[self.order_N:]
                Ez_mn = torch.matmul(Ky_norm,Hx_mn)/eps - torch.matmul(Kx_norm,Hy_mn)/eps

                # Spatial domain fields
                xy_phase = torch.exp(1.j * self.omega * (self.Kx_norm_dn*x_axis + self.Ky_norm_dn*y))
                Ex_split.append(torch.sum(Ex_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Ey_split.append(torch.sum(Ey_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Ez_split.append(torch.sum(Ez_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Hx_split.append(torch.sum(Hx_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Hy_split.append(torch.sum(Hy_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Hz_split.append(torch.sum(Hz_mn.reshape(1,1,-1)*xy_phase,dim=2))

            # Internal layers
            else:
                z_prop = z_axis[zi] - zm[layer_num[zi]]
                
                if layer_num[zi] != prev_layer_num:
                    if self.source_direction == 'forward':
                        C = torch.matmul(self.C[0][layer_num[zi]],self.E_i)
                    elif self.source_direction == 'backward':
                        C = torch.matmul(self.C[1][layer_num[zi]],self.E_i)

                    kz_norm = self.kz_norm[layer_num[zi]]
                    E_eigvec = self.E_eigvec[layer_num[zi]]
                    H_eigvec = self.H_eigvec[layer_num[zi]]

                    Cp = torch.diag(C[:2*self.order_N,0])
                    Cm = torch.diag(C[2*self.order_N:,0])

                    eps_conv_inv = torch.linalg.inv(self.eps_conv[layer_num[zi]])
                    mu_conv_inv = torch.linalg.inv(self.mu_conv[layer_num[zi]])

                # Phase
                z_phase_p = torch.diag(torch.exp(1.j*self.omega*kz_norm*z_prop))
                z_phase_m = torch.diag(torch.exp(1.j*self.omega*kz_norm*(self.thickness[layer_num[zi]]-z_prop)))

                # Fourier domain fields
                # [diffraction order, eigenmode number]
                Exy_p = torch.matmul(E_eigvec,z_phase_p)
                Ex_p = Exy_p[:self.order_N,:]
                Ey_p = Exy_p[self.order_N:,:]
                Hz_p = torch.matmul(mu_conv_inv,torch.matmul(Kx_norm,Ey_p)) - torch.matmul(mu_conv_inv,torch.matmul(Ky_norm,Ex_p))
                Exy_m = torch.matmul(E_eigvec,z_phase_m)
                Ex_m = Exy_m[:self.order_N,:]
                Ey_m = Exy_m[self.order_N:,:]
                Hz_m = torch.matmul(mu_conv_inv,torch.matmul(Kx_norm,Ey_m)) - torch.matmul(mu_conv_inv,torch.matmul(Ky_norm,Ex_m))
                Hxy_p = torch.matmul(H_eigvec,z_phase_p)
                Hx_p = Hxy_p[:self.order_N,:]
                Hy_p = Hxy_p[self.order_N:,:]
                Ez_p = torch.matmul(eps_conv_inv,torch.matmul(Ky_norm,Hx_p)) - torch.matmul(eps_conv_inv,torch.matmul(Kx_norm,Hy_p))
                Hxy_m = torch.matmul(-H_eigvec,z_phase_m)
                Hx_m = Hxy_m[:self.order_N,:]
                Hy_m = Hxy_m[self.order_N:,:]
                Ez_m = torch.matmul(eps_conv_inv,torch.matmul(Ky_norm,Hx_m)) - torch.matmul(eps_conv_inv,torch.matmul(Kx_norm,Hy_m))
                
                Ex_mn = torch.sum(torch.matmul(Ex_p,Cp) + torch.matmul(Ex_m,Cm),dim=1)
                Ey_mn = torch.sum(torch.matmul(Ey_p,Cp) + torch.matmul(Ey_m,Cm),dim=1)
                Ez_mn = torch.sum(torch.matmul(Ez_p,Cp) + torch.matmul(Ez_m,Cm),dim=1)
                Hx_mn = torch.sum(torch.matmul(Hx_p,Cp) + torch.matmul(Hx_m,Cm),dim=1)
                Hy_mn = torch.sum(torch.matmul(Hy_p,Cp) + torch.matmul(Hy_m,Cm),dim=1)
                Hz_mn = torch.sum(torch.matmul(Hz_p,Cp) + torch.matmul(Hz_m,Cm),dim=1)

                # Spatial domain fields
                xy_phase = torch.exp(1.j * self.omega * (self.Kx_norm_dn*x_axis + self.Ky_norm_dn*y))
                Ex_split.append(torch.sum(Ex_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Ey_split.append(torch.sum(Ey_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Ez_split.append(torch.sum(Ez_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Hx_split.append(torch.sum(Hx_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Hy_split.append(torch.sum(Hy_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Hz_split.append(torch.sum(Hz_mn.reshape(1,1,-1)*xy_phase,dim=2))

            prev_layer_num = layer_num[zi]

        Ex = torch.cat(Ex_split,dim=1)
        Ey = torch.cat(Ey_split,dim=1)
        Ez = torch.cat(Ez_split,dim=1)
        Hx = torch.cat(Hx_split,dim=1)
        Hy = torch.cat(Hy_split,dim=1)
        Hz = torch.cat(Hz_split,dim=1)

        return [Ex, Ey, Ez], [Hx, Hy, Hz]
    
    def field_yz(self,y_axis,z_axis,x):
        '''
            YZ-plane field distribution.
            Returns the field at the specific x point.

            Parameters
            - y_axis: y-direction sampling coordinates (torch.Tensor)
            - z_axis: z-direction sampling coordinates (torch.Tensor)
            - x: selected x point

            Return
            - [Ex, Ey, Ez] (list[torch.Tensor]), [Hx, Hy, Hz] (list[torch.Tensor])
        '''

        if type(y_axis) != torch.Tensor or type(z_axis) != torch.Tensor:
            print("x and z axis must be torch.Tensor type. Return None.")
            return None

        y_axis = y_axis.reshape([-1,1,1])

        Kx_norm = self.Kx_norm
        Ky_norm = self.Ky_norm

        Ex_split = []
        Ey_split = []
        Ez_split = []
        Hx_split = []
        Hy_split = []
        Hz_split = []

        # layer number
        zp = torch.zeros(len(self.thickness),device=self._device)
        zm = torch.zeros(len(self.thickness),device=self._device)
        layer_num = torch.zeros([len(z_axis)],dtype=torch.int64)
        layer_num[z_axis<0.] = -1

        for ti in range(len(self.thickness)):
            zp[ti:] += self.thickness[ti]

        for bi in range(len(zp)):
            layer_num[z_axis>zp[bi]] += 1
        zm[1:] = zp[0:-1]

        prev_layer_num = -2
        for zi in range(len(z_axis)):
            # Input and output layers
            if layer_num[zi] == -1 or layer_num[zi] == self.layer_N:
                Kx_norm_dn = self.Kx_norm_dn
                Ky_norm_dn = self.Ky_norm_dn

                if layer_num[zi] == -1:
                    z_prop = z_axis[zi] if z_axis[zi] <= 0. else 0.
                    if layer_num[zi] != prev_layer_num:
                        eps = self.eps_in if hasattr(self,'eps_in') else 1.
                        mu = self.mu_in if hasattr(self,'mu_in') else 1.
                        Vi = self.Vi if hasattr(self,'Vi') else self.Vf
                        Kz_norm_dn = torch.sqrt(eps*mu - Kx_norm_dn**2 - Ky_norm_dn**2)
                        Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)>0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])
                        Kz_norm_dn = torch.vstack((Kz_norm_dn,Kz_norm_dn))
                elif layer_num[zi] == self.layer_N:
                    if len(zp) == 0:
                        z_prop = z_axis[zi]
                    else:
                        z_prop = z_axis[zi]-zp[-1] if z_axis[zi]-zp[-1] >= 0. else 0.
                    if layer_num[zi] != prev_layer_num:
                        eps = self.eps_out if hasattr(self,'eps_in') else 1.
                        mu = self.mu_out if hasattr(self,'mu_in') else 1.        
                        Vo = self.Vo if hasattr(self,'Vo') else self.Vf
                        Kz_norm_dn = torch.sqrt(eps*mu - Kx_norm_dn**2 - Ky_norm_dn**2)
                        Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])
                        Kz_norm_dn = torch.vstack((Kz_norm_dn,Kz_norm_dn))

                # Phase
                z_phase = torch.exp(1.j*self.omega*Kz_norm_dn*z_prop)
                
                # Fourier domain fields
                # [diffraction order]
                if layer_num[zi] == -1 and self.source_direction == 'forward':
                    Exy_p = self.E_i*z_phase
                    Hxy_p = torch.matmul(Vi,Exy_p)
                    Exy_m = torch.matmul(self.S[1],self.E_i)*torch.conj(z_phase)
                    Hxy_m = torch.matmul(-Vi,Exy_m)
                elif layer_num[zi] == -1 and self.source_direction == 'backward':
                    Exy_p = torch.zeros_like(self.E_i)
                    Hxy_p = torch.zeros_like(self.E_i)
                    Exy_m = torch.matmul(self.S[3],self.E_i)*torch.conj(z_phase)
                    Hxy_m = torch.matmul(-Vi,Exy_m)
                elif layer_num[zi] == self.layer_N and self.source_direction == 'forward':
                    Exy_p = torch.matmul(self.S[0],self.E_i)*z_phase
                    Hxy_p = torch.matmul(Vo,Exy_p)
                    Exy_m = torch.zeros_like(self.E_i)
                    Hxy_m = torch.zeros_like(self.E_i)
                elif layer_num[zi] == self.layer_N and self.source_direction == 'backward':
                    Exy_p = torch.matmul(self.S[2],self.E_i)*z_phase
                    Hxy_p = torch.matmul(Vo,Exy_p)
                    Exy_m = self.E_i*torch.conj(z_phase)
                    Hxy_m = torch.matmul(-Vo,Exy_m)

                Ex_mn = Exy_p[:self.order_N] + Exy_m[:self.order_N]
                Ey_mn = Exy_p[self.order_N:] + Exy_m[self.order_N:]
                Hz_mn = torch.matmul(Kx_norm,Ey_mn)/mu - torch.matmul(Ky_norm,Ex_mn)/mu
                Hx_mn = Hxy_p[:self.order_N] + Hxy_m[:self.order_N]
                Hy_mn = Hxy_p[self.order_N:] + Hxy_m[self.order_N:]
                Ez_mn = torch.matmul(Ky_norm,Hx_mn)/eps - torch.matmul(Kx_norm,Hy_mn)/eps

                # Spatial domain fields
                xy_phase = torch.exp(1.j * self.omega * (self.Kx_norm_dn*x + self.Ky_norm_dn*y_axis))
                Ex_split.append(torch.sum(Ex_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Ey_split.append(torch.sum(Ey_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Ez_split.append(torch.sum(Ez_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Hx_split.append(torch.sum(Hx_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Hy_split.append(torch.sum(Hy_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Hz_split.append(torch.sum(Hz_mn.reshape(1,1,-1)*xy_phase,dim=2))

            # Internal layers
            else:
                if layer_num[zi] > 0:
                    z_prop = z_axis[zi] - zp[layer_num[zi]-1]
                else:
                    z_prop = z_axis[zi]
                
                if layer_num[zi] != prev_layer_num:
                    if self.source_direction == 'forward':
                        C = torch.matmul(self.C[0][layer_num[zi]],self.E_i)
                    elif self.source_direction == 'backward':
                        C = torch.matmul(self.C[1][layer_num[zi]],self.E_i)

                    kz_norm = self.kz_norm[layer_num[zi]]
                    E_eigvec = self.E_eigvec[layer_num[zi]]
                    H_eigvec = self.H_eigvec[layer_num[zi]]

                    Cp = torch.diag(C[:2*self.order_N,0])
                    Cm = torch.diag(C[2*self.order_N:,0])

                    eps_conv_inv = torch.linalg.inv(self.eps_conv[layer_num[zi]])
                    mu_conv_inv = torch.linalg.inv(self.mu_conv[layer_num[zi]])

                # Phase
                z_phase_p = torch.diag(torch.exp(1.j*self.omega*kz_norm*z_prop))
                z_phase_m = torch.diag(torch.exp(1.j*self.omega*kz_norm*(self.thickness[layer_num[zi]]-z_prop)))

                # Fourier domain fields
                # [diffraction order, eigenmode number]
                Exy_p = torch.matmul(E_eigvec,z_phase_p)
                Ex_p = Exy_p[:self.order_N,:]
                Ey_p = Exy_p[self.order_N:,:]
                Hz_p = torch.matmul(mu_conv_inv,torch.matmul(Kx_norm,Ey_p)) - torch.matmul(mu_conv_inv,torch.matmul(Ky_norm,Ex_p))
                Exy_m = torch.matmul(E_eigvec,z_phase_m)
                Ex_m = Exy_m[:self.order_N,:]
                Ey_m = Exy_m[self.order_N:,:]
                Hz_m = torch.matmul(mu_conv_inv,torch.matmul(Kx_norm,Ey_m)) - torch.matmul(mu_conv_inv,torch.matmul(Ky_norm,Ex_m))
                Hxy_p = torch.matmul(H_eigvec,z_phase_p)
                Hx_p = Hxy_p[:self.order_N,:]
                Hy_p = Hxy_p[self.order_N:,:]
                Ez_p = torch.matmul(eps_conv_inv,torch.matmul(Ky_norm,Hx_p)) - torch.matmul(eps_conv_inv,torch.matmul(Kx_norm,Hy_p))
                Hxy_m = torch.matmul(-H_eigvec,z_phase_m)
                Hx_m = Hxy_m[:self.order_N,:]
                Hy_m = Hxy_m[self.order_N:,:]
                Ez_m = torch.matmul(eps_conv_inv,torch.matmul(Ky_norm,Hx_m)) - torch.matmul(eps_conv_inv,torch.matmul(Kx_norm,Hy_m))
                
                Ex_mn = torch.sum(torch.matmul(Ex_p,Cp) + torch.matmul(Ex_m,Cm),dim=1)
                Ey_mn = torch.sum(torch.matmul(Ey_p,Cp) + torch.matmul(Ey_m,Cm),dim=1)
                Ez_mn = torch.sum(torch.matmul(Ez_p,Cp) + torch.matmul(Ez_m,Cm),dim=1)
                Hx_mn = torch.sum(torch.matmul(Hx_p,Cp) + torch.matmul(Hx_m,Cm),dim=1)
                Hy_mn = torch.sum(torch.matmul(Hy_p,Cp) + torch.matmul(Hy_m,Cm),dim=1)
                Hz_mn = torch.sum(torch.matmul(Hz_p,Cp) + torch.matmul(Hz_m,Cm),dim=1)

                # Spatial domain fields
                xy_phase = torch.exp(1.j * self.omega * (self.Kx_norm_dn*x + self.Ky_norm_dn*y_axis))
                Ex_split.append(torch.sum(Ex_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Ey_split.append(torch.sum(Ey_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Ez_split.append(torch.sum(Ez_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Hx_split.append(torch.sum(Hx_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Hy_split.append(torch.sum(Hy_mn.reshape(1,1,-1)*xy_phase,dim=2))
                Hz_split.append(torch.sum(Hz_mn.reshape(1,1,-1)*xy_phase,dim=2))

            prev_layer_num = layer_num[zi]

        Ex = torch.cat(Ex_split,dim=1)
        Ey = torch.cat(Ey_split,dim=1)
        Ez = torch.cat(Ez_split,dim=1)
        Hx = torch.cat(Hx_split,dim=1)
        Hy = torch.cat(Hy_split,dim=1)
        Hz = torch.cat(Hz_split,dim=1)

        return [Ex, Ey, Ez], [Hx, Hy, Hz]

    def field_xy(self,layer_num,x_axis,y_axis,z_prop=0.):
        '''
            XY-plane field distribution at the selected layer.
            Returns the field at z_prop away from the lower boundary of the layer.
            For the input layer, z_prop is the distance from the upper boundary and should be negative (calculate z_prop=0 if positive value is entered).

            Parameters
            - layer_num: selected layer (int)
            - x_axis: x-direction sampling coordinates (torch.Tensor)
            - y_axis: y-direction sampling coordinates (torch.Tensor)
            - z_prop: z-direction distance from the lower boundary of the layer (layer_num>-1),
                or the distance from the upper boundary of the layer and should be negative (layer_num=-1).

            Return
            - [Ex, Ey, Ez] (list[torch.Tensor]), [Hx, Hy, Hz] (list[torch.Tensor])
        '''

        if type(layer_num) != int:
            print("Parameter 'layer_num' must be int type. Return None.")
            return None

        if layer_num < -1 or layer_num > self.layer_N:
            print('Layer number is out of range. Return None.')
            return None

        if type(x_axis) != torch.Tensor or type(y_axis) != torch.Tensor:
            print("x and y axis must be torch.Tensor type. Return None.")
            return None
        
        # [x, y, diffraction order]
        x_axis = x_axis.reshape([-1,1,1])
        y_axis = y_axis.reshape([1,-1,1])

        Kx_norm = self.Kx_norm
        Ky_norm = self.Ky_norm

        # Input and output layers
        if layer_num == -1 or layer_num == self.layer_N:
            Kx_norm_dn = self.Kx_norm_dn
            Ky_norm_dn = self.Ky_norm_dn

            if layer_num == -1:
                z_prop = z_prop if z_prop <= 0. else 0.
                eps = self.eps_in if hasattr(self,'eps_in') else 1.
                mu = self.mu_in if hasattr(self,'mu_in') else 1.
                Vi = self.Vi if hasattr(self,'Vi') else self.Vf
                Kz_norm_dn = torch.sqrt(eps*mu - Kx_norm_dn**2 - Ky_norm_dn**2)
                Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)>0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])
            elif layer_num == self.layer_N:
                z_prop = z_prop if z_prop >= 0. else 0.
                eps = self.eps_out if hasattr(self,'eps_in') else 1.
                mu = self.mu_out if hasattr(self,'mu_in') else 1.        
                Vo = self.Vo if hasattr(self,'Vo') else self.Vf
                Kz_norm_dn = torch.sqrt(eps*mu - Kx_norm_dn**2 - Ky_norm_dn**2)
                Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])

            # Phase
            Kz_norm_dn = torch.vstack((Kz_norm_dn,Kz_norm_dn))
            z_phase = torch.exp(1.j*self.omega*Kz_norm_dn*z_prop)
            
            # Fourier domain fields
            # [diffraction order, diffraction order]
            if layer_num == -1 and self.source_direction == 'forward':
                Exy_p = self.E_i*z_phase
                Hxy_p = torch.matmul(Vi,Exy_p)
                Exy_m = torch.matmul(self.S[1],self.E_i)*torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vi,Exy_m)
            elif layer_num == -1 and self.source_direction == 'backward':
                Exy_p = torch.zeros_like(self.E_i)
                Hxy_p = torch.zeros_like(self.E_i)
                Exy_m = torch.matmul(self.S[3],self.E_i)*torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vi,Exy_m)
            elif layer_num == self.layer_N and self.source_direction == 'forward':
                Exy_p = torch.matmul(self.S[0],self.E_i)*z_phase
                Hxy_p = torch.matmul(Vo,Exy_p)
                Exy_m = torch.zeros_like(self.E_i)
                Hxy_m = torch.zeros_like(self.E_i)
            elif layer_num == self.layer_N and self.source_direction == 'backward':
                Exy_p = torch.matmul(self.S[2],self.E_i)*z_phase
                Hxy_p = torch.matmul(Vo,Exy_p)
                Exy_m = self.E_i*torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vo,Exy_m)

            Ex_mn = Exy_p[:self.order_N] + Exy_m[:self.order_N]
            Ey_mn = Exy_p[self.order_N:] + Exy_m[self.order_N:]
            Hz_mn = torch.matmul(Kx_norm,Ey_mn)/mu - torch.matmul(Ky_norm,Ex_mn)/mu
            Hx_mn = Hxy_p[:self.order_N] + Hxy_m[:self.order_N]
            Hy_mn = Hxy_p[self.order_N:] + Hxy_m[self.order_N:]
            Ez_mn = torch.matmul(Ky_norm,Hx_mn)/eps - torch.matmul(Kx_norm,Hy_mn)/eps

            # Spatial domain fields
            xy_phase = torch.exp(1.j * self.omega * (self.Kx_norm_dn*x_axis + self.Ky_norm_dn*y_axis))
            Ex = torch.sum(Ex_mn.reshape(1,1,-1)*xy_phase,dim=2)
            Ey = torch.sum(Ey_mn.reshape(1,1,-1)*xy_phase,dim=2)
            Ez = torch.sum(Ez_mn.reshape(1,1,-1)*xy_phase,dim=2)
            Hx = torch.sum(Hx_mn.reshape(1,1,-1)*xy_phase,dim=2)
            Hy = torch.sum(Hy_mn.reshape(1,1,-1)*xy_phase,dim=2)
            Hz = torch.sum(Hz_mn.reshape(1,1,-1)*xy_phase,dim=2)

        # Internal layers
        else:
            if self.source_direction == 'forward':
                C = torch.matmul(self.C[0][layer_num],self.E_i)
            elif self.source_direction == 'backward':
                C = torch.matmul(self.C[1][layer_num],self.E_i)

            kz_norm = self.kz_norm[layer_num]
            E_eigvec = self.E_eigvec[layer_num]
            H_eigvec = self.H_eigvec[layer_num]

            Cp = torch.diag(C[:2*self.order_N,0])
            Cm = torch.diag(C[2*self.order_N:,0])

            eps_conv_inv = torch.linalg.inv(self.eps_conv[layer_num])
            mu_conv_inv = torch.linalg.inv(self.mu_conv[layer_num])

            # Phase
            z_phase_p = torch.diag(torch.exp(1.j*self.omega*kz_norm*z_prop))
            z_phase_m = torch.diag(torch.exp(1.j*self.omega*kz_norm*(self.thickness[layer_num]-z_prop)))

            # Fourier domain fields
            # [diffraction order, eigenmode number]
            Exy_p = torch.matmul(E_eigvec,z_phase_p)
            Ex_p = Exy_p[:self.order_N,:]
            Ey_p = Exy_p[self.order_N:,:]
            Hz_p = torch.matmul(mu_conv_inv,torch.matmul(Kx_norm,Ey_p)) - torch.matmul(mu_conv_inv,torch.matmul(Ky_norm,Ex_p))
            Exy_m = torch.matmul(E_eigvec,z_phase_m)
            Ex_m = Exy_m[:self.order_N,:]
            Ey_m = Exy_m[self.order_N:,:]
            Hz_m = torch.matmul(mu_conv_inv,torch.matmul(Kx_norm,Ey_m)) - torch.matmul(mu_conv_inv,torch.matmul(Ky_norm,Ex_m))
            Hxy_p = torch.matmul(H_eigvec,z_phase_p)
            Hx_p = Hxy_p[:self.order_N,:]
            Hy_p = Hxy_p[self.order_N:,:]
            Ez_p = torch.matmul(eps_conv_inv,torch.matmul(Ky_norm,Hx_p)) - torch.matmul(eps_conv_inv,torch.matmul(Kx_norm,Hy_p))
            Hxy_m = torch.matmul(-H_eigvec,z_phase_m)
            Hx_m = Hxy_m[:self.order_N,:]
            Hy_m = Hxy_m[self.order_N:,:]
            Ez_m = torch.matmul(eps_conv_inv,torch.matmul(Ky_norm,Hx_m)) - torch.matmul(eps_conv_inv,torch.matmul(Kx_norm,Hy_m))
            
            Ex_mn = torch.sum(torch.matmul(Ex_p,Cp) + torch.matmul(Ex_m,Cm),dim=1)
            Ey_mn = torch.sum(torch.matmul(Ey_p,Cp) + torch.matmul(Ey_m,Cm),dim=1)
            Ez_mn = torch.sum(torch.matmul(Ez_p,Cp) + torch.matmul(Ez_m,Cm),dim=1)
            Hx_mn = torch.sum(torch.matmul(Hx_p,Cp) + torch.matmul(Hx_m,Cm),dim=1)
            Hy_mn = torch.sum(torch.matmul(Hy_p,Cp) + torch.matmul(Hy_m,Cm),dim=1)
            Hz_mn = torch.sum(torch.matmul(Hz_p,Cp) + torch.matmul(Hz_m,Cm),dim=1)

            # Spatial domain fields
            xy_phase = torch.exp(1.j * self.omega * (self.Kx_norm_dn*x_axis + self.Ky_norm_dn*y_axis))
            Ex = torch.sum(Ex_mn.reshape(1,1,-1)*xy_phase,dim=2)
            Ey = torch.sum(Ey_mn.reshape(1,1,-1)*xy_phase,dim=2)
            Ez = torch.sum(Ez_mn.reshape(1,1,-1)*xy_phase,dim=2)
            Hx = torch.sum(Hx_mn.reshape(1,1,-1)*xy_phase,dim=2)
            Hy = torch.sum(Hy_mn.reshape(1,1,-1)*xy_phase,dim=2)
            Hz = torch.sum(Hz_mn.reshape(1,1,-1)*xy_phase,dim=2)

        return [Ex, Ey, Ez], [Hx, Hy, Hz]

    # Internal functions
    def _kvectors(self):
        if self.angle_layer == 'input':
            self.kx0_norm = torch.sqrt(self.eps_in*self.mu_in) * torch.sin(self.inc_ang) * torch.cos(self.azi_ang)
            self.ky0_norm = torch.sqrt(self.eps_in*self.mu_in) * torch.sin(self.inc_ang) * torch.sin(self.azi_ang)
        else:
            self.kx0_norm = torch.sqrt(self.eps_out*self.mu_out) * torch.sin(self.inc_ang) * torch.cos(self.azi_ang)
            self.ky0_norm = torch.sqrt(self.eps_out*self.mu_out) * torch.sin(self.inc_ang) * torch.sin(self.azi_ang)

        # Free space k-vectors and E to H transformation matrix
        self.kx_norm = self.kx0_norm + self.order_x * self.Gx_norm
        self.ky_norm = self.ky0_norm + self.order_y * self.Gy_norm

        kx_norm_grid, ky_norm_grid = torch.meshgrid(self.kx_norm,self.ky_norm,indexing='ij')

        self.Kx_norm_dn = torch.reshape(kx_norm_grid,(-1,))
        self.Ky_norm_dn = torch.reshape(ky_norm_grid,(-1,))
        self.Kx_norm = torch.diag(self.Kx_norm_dn)
        self.Ky_norm = torch.diag(self.Ky_norm_dn)

        Kz_norm_dn = torch.sqrt(1. - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn)
        tmp1 = torch.vstack((torch.diag(-self.Ky_norm_dn*self.Kx_norm_dn/Kz_norm_dn), torch.diag(Kz_norm_dn + self.Kx_norm_dn**2/Kz_norm_dn)))
        tmp2 = torch.vstack((torch.diag(-Kz_norm_dn - self.Ky_norm_dn**2/Kz_norm_dn), torch.diag(self.Kx_norm_dn*self.Ky_norm_dn/Kz_norm_dn)))
        self.Vf = torch.hstack((tmp1, tmp2))

        if hasattr(self,'Sin'):
            # Input layer k-vectors and E to H transformation matrix
            Kz_norm_dn = torch.sqrt(self.eps_in*self.mu_in - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
            Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn)
            tmp1 = torch.vstack((torch.diag(-self.Ky_norm_dn*self.Kx_norm_dn/Kz_norm_dn), torch.diag(Kz_norm_dn + self.Kx_norm_dn**2/Kz_norm_dn)))
            tmp2 = torch.vstack((torch.diag(-Kz_norm_dn - self.Ky_norm_dn**2/Kz_norm_dn), torch.diag(self.Kx_norm_dn*self.Ky_norm_dn/Kz_norm_dn)))
            self.Vi = torch.hstack((tmp1, tmp2))

            Vtmp1 = torch.linalg.inv(self.Vf+self.Vi)
            Vtmp2 = self.Vf-self.Vi

            # Input layer S-matrix
            self.Sin.append(2*torch.matmul(Vtmp1,self.Vi))  # Tf S11
            self.Sin.append(-torch.matmul(Vtmp1,Vtmp2))     # Rf S21
            self.Sin.append(torch.matmul(Vtmp1,Vtmp2))      # Rb S12
            self.Sin.append(2*torch.matmul(Vtmp1,self.Vf))  # Tb S22

        if hasattr(self,'Sout'):
            # Output layer k-vectors and E to H transformation matrix
            Kz_norm_dn = torch.sqrt(self.eps_out*self.mu_out - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
            Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn)
            tmp1 = torch.vstack((torch.diag(-self.Ky_norm_dn*self.Kx_norm_dn/Kz_norm_dn), torch.diag(Kz_norm_dn + self.Kx_norm_dn**2/Kz_norm_dn)))
            tmp2 = torch.vstack((torch.diag(-Kz_norm_dn - self.Ky_norm_dn**2/Kz_norm_dn), torch.diag(self.Kx_norm_dn*self.Ky_norm_dn/Kz_norm_dn)))
            self.Vo = torch.hstack((tmp1, tmp2))

            Vtmp1 = torch.linalg.inv(self.Vf+self.Vo)
            Vtmp2 = self.Vf-self.Vo

            # Output layer S-matrix
            self.Sout.append(2*torch.matmul(Vtmp1,self.Vf))  # Tf S11
            self.Sout.append(torch.matmul(Vtmp1,Vtmp2))      # Rf S21
            self.Sout.append(-torch.matmul(Vtmp1,Vtmp2))     # Rb S12
            self.Sout.append(2*torch.matmul(Vtmp1,self.Vo))  # Tb S22

    def _material_conv(self,material):
        material_N = material.shape[0]*material.shape[1]

        # Matching indices
        order_x_grid, order_y_grid = torch.meshgrid(self.order_x,self.order_y,indexing='ij')
        ox = order_x_grid.to(torch.int64).reshape([-1])
        oy = order_y_grid.to(torch.int64).reshape([-1])

        ind = torch.arange(len(self.order_x)*len(self.order_y),device=self._device)
        indx, indy = torch.meshgrid(ind.to(torch.int64),ind.to(torch.int64),indexing='ij')

        material_fft = torch.fft.fft2(material)/material_N

        material_fft_real = torch.real(material_fft)
        material_fft_imag = torch.imag(material_fft)
        
        material_convmat_real = (material_fft_real[ox[indx]-ox[indy],oy[indx]-oy[indy]])
        material_convmat_imag = (material_fft_imag[ox[indx]-ox[indy],oy[indx]-oy[indy]])

        material_convmat = torch.complex(material_convmat_real,material_convmat_imag)
        
        return material_convmat
    
    def _eigen_decomposition_homogenous(self,eps,mu):
        # H to E transformation matirx
        self.P.append(torch.hstack((torch.vstack((torch.zeros_like(self.mu_conv[-1]),-self.mu_conv[-1])),
            torch.vstack((self.mu_conv[-1],torch.zeros_like(self.mu_conv[-1]))))) +
            1/eps * torch.matmul(torch.vstack((self.Kx_norm,self.Ky_norm)), torch.hstack((self.Ky_norm,-self.Kx_norm))))
        # E to H transformation matrix
        self.Q.append(torch.hstack((torch.vstack((torch.zeros_like(self.eps_conv[-1]),self.eps_conv[-1])),
            torch.vstack((-self.eps_conv[-1],torch.zeros_like(self.eps_conv[-1]))))) +
            1/mu * torch.matmul(torch.vstack((self.Kx_norm,self.Ky_norm)), torch.hstack((-self.Ky_norm,self.Kx_norm))))
        
        E_eigvec = torch.eye(self.P[-1].shape[-1],dtype=self._dtype,device=self._device)
        kz_norm = torch.sqrt(eps*mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        kz_norm = torch.where(torch.imag(kz_norm)<0,torch.conj(kz_norm),kz_norm) # Normalized kz for positive mode
        kz_norm = torch.cat((kz_norm,kz_norm))

        self.kz_norm.append(kz_norm) 
        self.E_eigvec.append(E_eigvec)

    def _eigen_decomposition(self):
        # H to E transformation matirx
        P_tmp = torch.matmul(torch.vstack((self.Kx_norm,self.Ky_norm)), torch.linalg.inv(self.eps_conv[-1]))
        self.P.append(torch.hstack((torch.vstack((torch.zeros_like(self.mu_conv[-1]),-self.mu_conv[-1])),
            torch.vstack((self.mu_conv[-1],torch.zeros_like(self.mu_conv[-1]))))) + torch.matmul(P_tmp, torch.hstack((self.Ky_norm,-self.Kx_norm))))
        # E to H transformation matrix
        Q_tmp = torch.matmul(torch.vstack((self.Kx_norm,self.Ky_norm)), torch.linalg.inv(self.mu_conv[-1]))
        self.Q.append(torch.hstack((torch.vstack((torch.zeros_like(self.eps_conv[-1]),self.eps_conv[-1])),
            torch.vstack((-self.eps_conv[-1],torch.zeros_like(self.eps_conv[-1]))))) + torch.matmul(Q_tmp, torch.hstack((-self.Ky_norm,self.Kx_norm))))
        
        # Eigen-decomposition
        if self.stable_eig_grad is True:
            kz_norm, E_eigvec = Eig.apply(torch.matmul(self.P[-1],self.Q[-1]))
        else:
            kz_norm, E_eigvec = torch.linalg.eig(torch.matmul(self.P[-1],self.Q[-1]))
        
        kz_norm = torch.sqrt(kz_norm)
        self.kz_norm.append(torch.where(torch.imag(kz_norm)<0,-kz_norm,kz_norm)) # Normalized kz for positive mode
        self.E_eigvec.append(E_eigvec)

    def _solve_layer_smatrix(self):
        Kz_norm = torch.diag(self.kz_norm[-1])
        phase = torch.diag(torch.exp(1.j*self.omega*self.kz_norm[-1]*self.thickness[-1]))

        Pinv_tmp = torch.linalg.inv(self.P[-1])
        if self.avoid_Pinv_instability == True:
            
            Pinv_ins_tmp1 = torch.max(torch.abs( torch.matmul(self.P[-1].detach(),Pinv_tmp.detach())-torch.eye(self.P[-1].shape[-1]).to(self.P[-1]) ))
            Pinv_ins_tmp2 = torch.max(torch.abs( torch.matmul(Pinv_tmp.detach(),self.P[-1].detach())-torch.eye(self.P[-1].shape[-1]).to(self.P[-1]) ))
            Qinv_ins_tmp1 = torch.max(torch.abs( torch.matmul(self.Q[-1].detach(),torch.linalg.inv(self.Q[-1]).detach())-torch.eye(self.Q[-1].shape[-1]).to(self.Q[-1]) ))
            Qinv_ins_tmp2 = torch.max(torch.abs( torch.matmul(self.Q[-1].detach(),torch.linalg.inv(self.Q[-1]).detach())-torch.eye(self.Q[-1].shape[-1]).to(self.Q[-1]) ))

            self.Pinv_instability.append(torch.maximum(Pinv_ins_tmp1,Pinv_ins_tmp2))
            self.Qinv_instability.append(torch.maximum(Qinv_ins_tmp1,Qinv_ins_tmp2))

            if self.Pinv_instability[-1] < self.max_Pinv_instability:
                self.H_eigvec.append(torch.matmul(Pinv_tmp,torch.matmul(self.E_eigvec[-1],Kz_norm)))
            else:
                self.H_eigvec.append(torch.matmul(self.Q[-1],torch.matmul(self.E_eigvec[-1],torch.linalg.inv(Kz_norm))))
        else:
            self.H_eigvec.append(torch.matmul(Pinv_tmp,torch.matmul(self.E_eigvec[-1],Kz_norm)))

        Ctmp1 = torch.vstack((self.E_eigvec[-1] + torch.matmul(torch.linalg.inv(self.Vf),self.H_eigvec[-1]), torch.matmul(self.E_eigvec[-1] - torch.matmul(torch.linalg.inv(self.Vf),self.H_eigvec[-1]),phase)))
        Ctmp2 = torch.vstack((torch.matmul(self.E_eigvec[-1] - torch.matmul(torch.linalg.inv(self.Vf),self.H_eigvec[-1]),phase), self.E_eigvec[-1] + torch.matmul(torch.linalg.inv(self.Vf),self.H_eigvec[-1])))
        Ctmp = torch.hstack((Ctmp1,Ctmp2))

        # Mode coupling coefficients
        self.Cf.append(torch.matmul( torch.linalg.inv(Ctmp), torch.vstack((2*torch.eye(2*self.order_N,dtype=self._dtype,device=self._device),
            torch.zeros([2*self.order_N,2*self.order_N],dtype=self._dtype,device=self._device))) ))
        self.Cb.append(torch.matmul( torch.linalg.inv(Ctmp), torch.vstack((torch.zeros([2*self.order_N,2*self.order_N],dtype=self._dtype,device=self._device),
            2*torch.eye(2*self.order_N,dtype=self._dtype,device=self._device))) ))

        self.layer_S11.append(torch.matmul(torch.matmul(self.E_eigvec[-1],phase), self.Cf[-1][:2*self.order_N,:]) + torch.matmul(self.E_eigvec[-1],self.Cf[-1][2*self.order_N:,:]))
        self.layer_S21.append(torch.matmul(self.E_eigvec[-1], self.Cf[-1][:2*self.order_N,:]) + torch.matmul(torch.matmul(self.E_eigvec[-1],phase),self.Cf[-1][2*self.order_N:,:])
            - torch.eye(2*self.order_N,dtype=self._dtype,device=self._device))
        self.layer_S12.append(torch.matmul(torch.matmul(self.E_eigvec[-1],phase), self.Cb[-1][:2*self.order_N,:]) + torch.matmul(self.E_eigvec[-1],self.Cb[-1][2*self.order_N:,:])
            - torch.eye(2*self.order_N,dtype=self._dtype,device=self._device))
        self.layer_S22.append(torch.matmul(self.E_eigvec[-1], self.Cb[-1][:2*self.order_N,:]) + torch.matmul(torch.matmul(self.E_eigvec[-1],phase),self.Cb[-1][2*self.order_N:,:]))

    def _RS_prod(self,Sm,Sn,Cm,Cn):
        # S11 = S[0] / S21 = S[1] / S12 = S[2] / S22 = S[3]
        # Cf = C[0] / Cb = C[1]

        tmp1 = torch.linalg.inv(torch.eye(2*self.order_N,dtype=self._dtype,device=self._device) - torch.matmul(Sm[2],Sn[1]))
        tmp2 = torch.linalg.inv(torch.eye(2*self.order_N,dtype=self._dtype,device=self._device) - torch.matmul(Sn[1],Sm[2]))

        # Layer S-matrix
        S11 = torch.matmul(Sn[0],torch.matmul(tmp1,Sm[0]))
        S21 = Sm[1] + torch.matmul(Sm[3],torch.matmul(tmp2,torch.matmul(Sn[1],Sm[0])))
        S12 = Sn[2] + torch.matmul(Sn[0],torch.matmul(tmp1,torch.matmul(Sm[2],Sn[3])))
        S22 = torch.matmul(Sm[3],torch.matmul(tmp2,Sn[3]))

        # Mode coupling coefficients
        C = [[],[]]
        for m in range(len(Cm[0])):
            C[0].append(Cm[0][m] + torch.matmul(Cm[1][m],torch.matmul(tmp2,torch.matmul(Sn[1],Sm[0]))))
            C[1].append(torch.matmul(Cm[1][m],torch.matmul(tmp2,Sn[3])))

        for n in range(len(Cn[0])):
            C[0].append(torch.matmul(Cn[0][n],torch.matmul(tmp1,Sm[0])))
            C[1].append(Cn[1][n] + torch.matmul(Cn[0][n],torch.matmul(tmp1,torch.matmul(Sm[2],Sn[3]))))

        return [S11, S21, S12, S22], C