from particle_tracker import *
import numpy as np
from scipy.interpolate import interp1d
import scipy.constants as sc
import matplotlib.pyplot as plt

# 30 keV beam
xhv = 30.0e3 

# Opacity data
data_file = "../Data/HenkeCu_100um.dat"
with open(data_file,'r') as f:
	line = f.readline()
	equals_split = line.split('=')
	dens_split = equals_split[1]
	thick_split = equals_split[2]

	rho_solid = float(dens_split.split()[0])*1e3
	thickness = float(thick_split.split()[0])*1e-6

opa_data = np.loadtxt(data_file,skiprows=2)
opa_data[:,1] = -np.log(opa_data[:,1])/thickness
opa_interp = interp1d(opa_data[:,0],opa_data[:,1])
opa_solid = opa_interp(xhv)

# Set up data grid
L = 1.0e-2 # m
Nx = 51
x,y,z = np.linspace(-L/2,L/2,Nx),np.linspace(-L/2,L/2,Nx),np.linspace(-L/2,L/2,Nx)

# Set up densities
rho = np.ones((Nx,Nx,Nx))
rho *= 10.0e0 + 0.1e0*0.5*(1+np.tanh(x[:,None,None]/(0.01*L)))
ne = rho/(63.546*sc.value('atomic mass constant'))

# Initialise solver Object
TestCase = ElectronCube(x, y, z, L, absorption = True, phaseshift = True, probing_direction = 'z')
TestCase.external_ne(ne)
TestCase.external_absorption(opa_solid, rho, rho_solid)
TestCase.calc_dndr(xhv)
TestCase.set_up_interps()

print(np.amax(TestCase.ne_nc))

# X-ray beam properties
Np = int(1e5)
beam_size = [L/2,L/2]
divergence = 0.05e-3 # 0.05 mrad, realistic
s0 = init_beam(Np, beam_size, divergence, TestCase.extent, probing_direction = 'z')

# Solve ray trace
TestCase.solve(s0)

Npix = 30
H, xedge, yedges = ray_histogram(TestCase.rf,TestCase.Jf,L,L,Npix,Npix)
H = np.sqrt(H)

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(x,ne[:,0,0])

ax2.plot(0.5*(xedge[:-1]+xedge[1:]),np.sum(H,axis=1))
ax2.plot(x,Np/Npix*np.exp(-np.sum(TestCase.opa_a[:,0,:],axis=1)*L/Nx))

im3 = ax3.imshow(np.exp(-np.sum(TestCase.opa_a,axis=2)*L/Nx))
fig.colorbar(im3,ax=ax3)

im4 = ax4.imshow(H,cmap='Greys_r')
fig.colorbar(im4,ax=ax4)

fig.tight_layout()

plt.show()