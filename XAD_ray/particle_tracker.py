"""PARTICLE TRACKER
BASED ON: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.61.895

SOLVES: 
$ \frac{d\vec{v}}{dt} = -\nabla \left( \frac{c^2}{2} \frac{n_e}{n_c} \right) $

$ \frac{d\vec{x}}{dt} = \vec{v} $

CODE BY: Aidan CRILLY
REFACTORING: Jack HARE

Taken from https://github.com/jdhare/turbulence_tracing repo

"""

import numpy as np
from scipy.integrate import odeint,solve_ivp
from scipy.interpolate import RegularGridInterpolator
from time import time
import scipy.constants as sc

c = sc.c # honestly, this could be 3e8 *shrugs*
sigma_t = 6.65246e-29 # 1/m^2

class ElectronCube:
    """A class to hold and generate electron density cubes
    """
    
    def __init__(self, x, y, z, extent, absorption = False, phaseshift = False, probing_direction = 'z'):
        """
        Example:
            N_V = 100
            M_V = 2*N_V+1
            ne_extent = 5.0e-3
            ne_x = np.linspace(-ne_extent,ne_extent,M_V)
            ne_y = np.linspace(-ne_extent,ne_extent,M_V)
            ne_z = np.linspace(-ne_extent,ne_extent,M_V)

        Args:
            x (float array): x coordinates, m
            y (float array): y coordinates, m
            z (float array): z coordinates, m
            extent (float): physical size, m
        """
        self.z,self.y,self.x = z, y, x
        self.dx = x[1]-x[0]
        self.XX, self.YY, self.ZZ = np.meshgrid(x,y,z, indexing='ij')
        self.extent = extent
        self.probing_direction = probing_direction
        # Logical switches
        self.absorption = absorption
        self.phaseshift = phaseshift
        
    def external_ne(self, ne):
        """Load externally generated grid

        Args:
            ne ([type]): MxMxM grid of density in m^-3
        """
        self.ne = ne

    def external_absorption(self, opacity, rho, rho_solid):
        self.opa_a = opacity*(rho/rho_solid)

    def calc_dndr(self, xhv):
        """Generate interpolators for derivatives.

        Args:
            xhv (float): X-ray energy in eV
        """

        self.omega = 2*np.pi*sc.e*xhv/sc.h
        nc = 3.14207787e-4*self.omega**2

        self.ne_nc = self.ne/nc #normalise to critical density
        
        #More compact notation is possible here, but we are explicit
        self.dndx = -0.5*c**2*np.gradient(self.ne_nc,self.x,axis=0)
        self.dndy = -0.5*c**2*np.gradient(self.ne_nc,self.y,axis=1)
        self.dndz = -0.5*c**2*np.gradient(self.ne_nc,self.z,axis=2)
        
        self.dndx_interp = RegularGridInterpolator((self.x, self.y, self.z), self.dndx, bounds_error = False, fill_value = 0.0)
        self.dndy_interp = RegularGridInterpolator((self.x, self.y, self.z), self.dndy, bounds_error = False, fill_value = 0.0)
        self.dndz_interp = RegularGridInterpolator((self.x, self.y, self.z), self.dndz, bounds_error = False, fill_value = 0.0)

    # Plasma refractive index
    def n_refrac(self):
        def omega_pe(ne):
            '''Calculate electron plasma freq. Output units are rad/sec. From nrl pp 28'''
            return 5.64e4*np.sqrt(ne)
        ne_cc = self.ne*1e-6
        o_pe  = omega_pe(ne_cc)
        o_pe[o_pe > self.omega] = self.omega
        return np.sqrt(1.0-(o_pe/self.omega)**2)

    def absorp(self):
        return (sigma_t*self.ne+self.opa_a)*c

    def set_up_interps(self):
        # Electron density
        self.ne_interp = RegularGridInterpolator((self.x, self.y, self.z), self.ne, bounds_error = False, fill_value = 0.0)
        # Inverse Bremsstrahlung
        if(self.absorption):
            self.absorp_interp = RegularGridInterpolator((self.x, self.y, self.z), self.absorp(), bounds_error = False, fill_value = 0.0)
        # Phase shift
        if(self.phaseshift):
            self.refractive_index_interp = RegularGridInterpolator((self.x, self.y, self.z), self.n_refrac(), bounds_error = False, fill_value = 1.0)
    
    def dndr(self,x):
        """returns the gradient at the locations x

        Args:
            x (3xN float): N [x,y,z] locations

        Returns:
            3 x N float: N [dx,dy,dz] electron density gradients
        """
        grad = np.zeros_like(x)
        grad[0,:] = self.dndx_interp(x.T)
        grad[1,:] = self.dndy_interp(x.T)
        grad[2,:] = self.dndz_interp(x.T)
        return grad

    # Attenuation due to inverse bremsstrahlung
    def atten(self,x):
        if(self.absorption):
            return -self.absorp_interp(x.T)
        else:
            return 0.0

    # Phase shift introduced by refractive index
    def phase(self,x):
        if(self.phaseshift):
            return self.omega*self.refractive_index_interp(x.T)
        else:
            return 0.0

    def get_ne(self,x):
        return self.ne_interp(x.T)

    def solve(self, s0, method = 'RK45'):
        # Need to make sure all rays have left volume
        # Conservative estimate of diagonal across volume
        # Then can backproject to surface of volume

        t  = np.linspace(0.0,np.sqrt(8.0)*self.extent/c,2)

        s0 = s0.flatten() #odeint insists

        start = time()
        dsdt_ODE = lambda t, y: dsdt(t, y, self)
        sol = solve_ivp(dsdt_ODE, [0,t[-1]], s0, t_eval=t, method = method, rtol=1e-4)
        finish = time()
        print("Ray trace completed in:\t",finish-start,"s")

        Np = s0.size//9
        self.sf = sol.y[:,-1].reshape(9,Np)
        # Fix amplitudes
        self.sf[6,self.sf[6,:] < 0.0] = 0.0
        self.rf,self.Jf = ray_to_Jonesvector(self.sf, self.extent, probing_direction = self.probing_direction)

    def clear_memory(self):
        """
        Clears variables not needed by solve method, saving memory

        Can also use after calling solve to clear ray positions - important when running large number of rays

        """
        self.dndx = None
        self.dndx = None
        self.dndx = None
        self.ne = None
        self.ne_nc = None
        self.sf = None
        self.rf = None
    
# ODEs of photon paths
def dsdt(t, s, ElectronCube):
    """Returns an array with the gradients and velocity per ray for ode_int

    Args:
        t (float array): I think this is a dummy variable for ode_int - our problem is time invarient
        s (9N float array): flattened 9xN array of rays used by ode_int
        ElectronCube (ElectronCube): an ElectronCube object which can calculate gradients

    Returns:
        9N float array: flattened array for ode_int
    """
    Np     = s.size//9
    s      = s.reshape(9,Np)
    sprime = np.zeros_like(s)
    # Velocity and position
    v = s[3:6,:]
    x = s[:3,:]
    # Amplitude, phase
    a = s[6,:]
    p = s[7,:]

    sprime[3:6,:] = ElectronCube.dndr(x)
    sprime[:3,:]  = v
    sprime[6,:]   = ElectronCube.atten(x)*a
    sprime[7,:]   = ElectronCube.phase(x)
    sprime[8,:]   = 0.0

    return sprime.flatten()

def init_beam(Np, beam_size, divergence, ne_extent, probing_direction = 'z'):
    """[summary]

    Args:
        Np (int): Number of photons
        beam_size (float): beam radius, m
        divergence (float): beam divergence, radians
        ne_extent (float): size of electron density cube, m. Used to back propagate the rays to the start
        probing_direction (str): direction of probing. I suggest 'z', the best tested

    Returns:
        s0, 9 x N float: N rays with (x, y, z, vx, vy, vz) in m, m/s and amplitude, phase and polarisation (a, p, r) 
    """
    s0 = np.zeros((9,Np))

    # position, uniformly within a square
    t  = 2*np.random.rand(Np)-1.0
    u  = 2*np.random.rand(Np)-1.0
    # angle
    ϕ = np.pi*np.random.rand(Np) #azimuthal angle of velocity
    χ = divergence*np.random.randn(Np) #polar angle of velocity

    beam_size_1 = beam_size[0]
    beam_size_2 = beam_size[1]

    if(probing_direction == 'x'):
        # Initial velocity
        s0[3,:] = c * np.cos(χ)
        s0[4,:] = c * np.sin(χ) * np.cos(ϕ)
        s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
        # Initial position
        s0[0,:] = -ne_extent
        s0[1,:] = beam_size_1*u
        s0[2,:] = beam_size_2*t
    elif(probing_direction == 'y'):
        # Initial velocity
        s0[4,:] = c * np.cos(χ)
        s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
        s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
        # Initial position
        s0[0,:] = beam_size_1*u
        s0[1,:] = -ne_extent
        s0[2,:] = beam_size_2*t
    elif(probing_direction == 'z'):
        # Initial velocity
        s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
        s0[4,:] = c * np.sin(χ) * np.sin(ϕ)
        s0[5,:] = c * np.cos(χ)
        # Initial position
        s0[0,:] = beam_size_1*u
        s0[1,:] = beam_size_2*t
        s0[2,:] = -ne_extent
    else: # Default to y
        print("Default to y")
        # Initial velocity
        s0[4,:] = c * np.cos(χ)
        s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
        s0[5,:] = c * np.sin(χ) * np.sin(ϕ)        
        # Initial position
        s0[0,:] = beam_size_1*u
        s0[1,:] = -ne_extent
        s0[2,:] = beam_size_2*t

    # Initialise amplitude, phase and polarisation

    phase = np.zeros(Np)

    s0[6,:] = 1.0
    s0[7,:] = phase
    s0[8,:] = 0.0
    return s0

# Need to backproject to ne volume, then find angles
def ray_to_Jonesvector(ode_sol, ne_extent, probing_direction = 'z'):
    """Takes the output from the 9D solver and returns 6D rays for ray-transfer matrix techniques.
    Effectively finds how far the ray is from the end of the volume, returns it to the end of the volume.

    Args:
        ode_sol (6xN float): N rays in (x,y,z,vx,vy,vz) format, m and m/s and amplitude, phase and polarisation
        ne_extent (float): edge length of cube, m
        probing_direction (str): x, y or z.

    Returns:
        [type]: [description]
    """
    Np = ode_sol.shape[1] # number of photons
    ray_p = np.zeros((4,Np))
    ray_J = np.zeros((2,Np),dtype=np.complex)

    x, y, z, vx, vy, vz = ode_sol[0], ode_sol[1], ode_sol[2], ode_sol[3], ode_sol[4], ode_sol[5]

    # Resolve distances and angles
    # YZ plane
    if(probing_direction == 'x'):
        t_bp = (x-ne_extent)/vx
        # Positions on plane
        ray_p[0] = y-vy*t_bp
        ray_p[2] = z-vz*t_bp
        # Angles to plane
        ray_p[1] = np.arctan(vy/vx)
        ray_p[3] = np.arctan(vz/vx)
    # XZ plane
    elif(probing_direction == 'y'):
        t_bp = (y-ne_extent)/vy
        # Positions on plane
        ray_p[0] = x-vx*t_bp
        ray_p[2] = z-vz*t_bp
        # Angles to plane
        ray_p[1] = np.arctan(vx/vy)
        ray_p[3] = np.arctan(vz/vy)
    # XY plane
    elif(probing_direction == 'z'):
        t_bp = (z-ne_extent)/vz
        # Positions on plane
        ray_p[0] = x-vx*t_bp
        ray_p[2] = y-vy*t_bp
        # Angles to plane
        ray_p[1] = np.arctan(vx/vz)
        ray_p[3] = np.arctan(vy/vz)

    # Resolve Jones vectors
    amp,phase,pol = ode_sol[6], ode_sol[7], ode_sol[8]
    # Assume initially polarised along y
    E_x_init = np.zeros(Np)
    E_y_init = np.ones(Np)
    # Perform rotation for polarisation, multiplication for amplitude, and complex rotation for phase
    ray_J[0] = amp*(np.cos(phase)+1.0j*np.sin(phase))*(np.cos(pol)*E_x_init-np.sin(pol)*E_y_init)
    ray_J[1] = amp*(np.cos(phase)+1.0j*np.sin(phase))*(np.sin(pol)*E_x_init+np.cos(pol)*E_y_init)

    # ray_p [x,phi,y,theta], ray_J [E_x,E_y]

    return ray_p,ray_J


def ray_histogram(rf,Jf,Lx,Ly,pix_x,pix_y):
    x=rf[0,:]
    y=rf[2,:]
    
    nonans = ~np.isnan(x)
    
    x=x[nonans]
    y=y[nonans]
    
    #treat the imaginary and real parts of E_x and E_y all separately.
    E_x_real = np.real(Jf[0,:])
    E_x_imag = np.imag(Jf[0,:])
    E_y_real = np.real(Jf[1,:])
    E_y_imag = np.imag(Jf[1,:])
    
    E_x_real = E_x_real[nonans]
    E_x_imag = E_x_imag[nonans]
    E_y_real = E_y_real[nonans]
    E_y_imag = E_y_imag[nonans]

    ## create four separate histograms for the real and imaginary parts of E_x and E-y
    H_Ex_real, xedges, yedges = np.histogram2d(x, y, bins=[pix_x, pix_y], 
                                                     range=[[-Lx/2, Lx/2],[-Ly/2,Ly/2]],
                                                     normed = False, weights = E_x_real)
    
    H_Ex_imag, xedges, yedges = np.histogram2d(x, y, bins=[pix_x, pix_y], 
                                                     range=[[-Lx/2, Lx/2],[-Ly/2,Ly/2]],
                                                     normed = False, weights = E_x_imag)
        
    H_Ey_real, xedges, yedges = np.histogram2d(x, y, bins=[pix_x, pix_y], 
                                                     range=[[-Lx/2, Lx/2],[-Ly/2,Ly/2]],
                                                     normed = False, weights = E_y_real)
            
    H_Ey_imag, xedges, yedges = np.histogram2d(x, y, bins=[pix_x, pix_y], 
                                                     range=[[-Lx/2, Lx/2],[-Ly/2,Ly/2]],
                                                     normed = False, weights = E_y_imag)
    
    # Recontruct the complex valued E_x and E_y components
    H_Ex = H_Ex_real+1j*H_Ex_imag
    H_Ey = H_Ey_real+1j*H_Ey_imag
    
    # Find the intensity using complex conjugates. Take the real value to remove very small (numerical error) imaginary components
    H = np.real(H_Ex*np.conj(H_Ex) + H_Ey*np.conj(H_Ey))

    return H ,xedges, yedges