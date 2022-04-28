import torch
import torch.fft

class rcwa_geo:
    edge_sharpness = 100.   # sharpness of edge
    Lx = 1.                 # x-direction Lattice constant 
    Ly = 1.                 # y-direction Lattice constant 
    nx = 100                # x-axis sampling number
    ny = 100                # y-axis sampling number 
    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self):
        pass
    
    @classmethod
    def grid(cls):
        '''
            Update grid
        '''

        cls.x = (cls.Lx/cls.nx)*(torch.arange(cls.nx,dtype=cls.dtype,device=cls.device)+0.5)
        cls.y = (cls.Ly/cls.ny)*(torch.arange(cls.ny,dtype=cls.dtype,device=cls.device)+0.5)
        cls.x_grid, cls.y_grid = torch.meshgrid(cls.x,cls.y,indexing='ij')

    @classmethod
    def circle(cls,R,Cx,Cy):
        ''' 
            R: radius
            Cx: x center
            Cy: y center
        '''

        cls.grid()
        level = 1. - torch.sqrt(((cls.x_grid-Cx)/R)**2 + ((cls.y_grid-Cy)/R)**2)
        return torch.sigmoid(cls.edge_sharpness*level)

    @classmethod
    def ellipse(cls,Rx,Ry,Cx,Cy,theta=0.):
        '''
            Rx: x direction radius
            Ry: y direction radius
            Cx: x center
            Cy: y center
        '''

        theta = torch.as_tensor(theta,dtype=cls.dtype,device=cls.device)

        cls.grid()
        level = 1. - torch.sqrt((((cls.x_grid-Cx)*torch.cos(theta)+(cls.y_grid-Cy)*torch.sin(theta))/Rx)**2 + ((-(cls.x_grid-Cx)*torch.sin(theta)+(cls.y_grid-Cy)*torch.cos(theta))/Ry)**2)
        return torch.sigmoid(cls.edge_sharpness*level)

    @classmethod
    def square(cls,W,Cx,Cy,theta=0.):
        '''
            W: width
            Cx: x center
            Cy: y center
            theta: rotation angle / center: [Cx, Cy] / axis: x-axis
        '''

        theta = torch.as_tensor(theta,dtype=cls.dtype,device=cls.device)

        cls.grid()
        level = 1. - (torch.maximum(torch.abs(((cls.x_grid-Cx)*torch.cos(theta)+(cls.y_grid-Cy)*torch.sin(theta))/(W/2.)),torch.abs((-(cls.x_grid-Cx)*torch.sin(theta)+(cls.y_grid-Cy)*torch.cos(theta))/(W/2.))))
        return torch.sigmoid(cls.edge_sharpness*level)

    @classmethod
    def rectangle(cls,Wx,Wy,Cx,Cy,theta=0.):
        '''
            Wx: x width
            Wy: y width
            Cx: x center
            Cy: y center
            theta: rotation angle / center: [Cx, Cy] / axis: x-axis
        '''

        theta = torch.as_tensor(theta,dtype=cls.dtype,device=cls.device)

        cls.grid()
        level = 1. - (torch.maximum(torch.abs(((cls.x_grid-Cx)*torch.cos(theta)+(cls.y_grid-Cy)*torch.sin(theta))/(Wx/2.)),torch.abs((-(cls.x_grid-Cx)*torch.sin(theta)+(cls.y_grid-Cy)*torch.cos(theta))/(Wy/2.))))
        return torch.sigmoid(cls.edge_sharpness*level)

    @classmethod
    def rhombus(cls,Wx,Wy,Cx,Cy,theta=0.):
        '''
            Wx: x diagonal
            Wy: y diagonal
            Cx: x center
            Cy: y center
            theta: rotation angle / center: [Cx, Cy] / axis: x-axis
        '''

        theta = torch.as_tensor(theta,dtype=cls.dtype,device=cls.device)

        cls.grid()
        level = 1. - (torch.abs(((cls.x_grid-Cx)*torch.cos(theta)+(cls.y_grid-Cy)*torch.sin(theta))/(Wx/2.)) + torch.abs((-(cls.x_grid-Cx)*torch.sin(theta)+(cls.y_grid-Cy)*torch.cos(theta))/(Wy/2.)))
        return torch.sigmoid(cls.edge_sharpness*level)

    @classmethod
    def super_ellipse(cls,Wx,Wy,Cx,Cy,theta=0.,power=2.):
        '''
            Wx: x width
            Wy: y width
            Cx: x center
            Cy: y center
            theta: rotation angle / center: [Cx, Cy] / axis: x-axis
            power: elliptic power
        '''

        theta = torch.as_tensor(theta,dtype=cls.dtype,device=cls.device)

        cls.grid()
        level = 1. - (torch.abs(((cls.x_grid-Cx)*torch.cos(theta)+(cls.y_grid-Cy)*torch.sin(theta))/(Wx/2.))**power + torch.abs((-(cls.x_grid-Cx)*torch.sin(theta)+(cls.y_grid-Cy)*torch.cos(theta))/(Wy/2.))**power)**(1/power)
        return torch.sigmoid(cls.edge_sharpness*level)

    @classmethod
    def union(cls,A,B):
        '''
            A U B
        '''

        return torch.maximum(A,B)

    @classmethod
    def intersection(cls,A,B):
        '''
            A n B
        '''

        return torch.minimum(A,B)

    @classmethod
    def difference(cls,A,B):
        '''
            A - B = A n Bc
        '''

        return torch.minimum(A,1.-B)