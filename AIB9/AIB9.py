# Application to AIB9
import os, math, sys
import matplotlib
import matplotlib.pyplot as plt
import openmm.app  as omm_app
import openmm as omm
import simtk.unit as unit
import openmmtools as ommt
import torch
import torch.nn as nn
import time
import numpy as np
import copy
import utils
import pickle
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(42)
from constructor import committor_constructor
import utils

# Create committor network
class PredictNet(torch.nn.Module):
    def __init__(self):
        super(PredictNet, self).__init__()
        block = [nn.Linear(8274, 100),
                      nn.LeakyReLU(),
                      nn.Linear(100, 100),
                      nn.LeakyReLU(),
                      nn.Linear(100, 100),
                      nn.LeakyReLU(),
                      nn.Linear(100, 100),
                      nn.LeakyReLU(),
                      nn.Linear(100, 1)]
        self.block = nn.Sequential(*block)

    def compute_aib9_embedding(self,positions):
    
        def _compute_dihedrals(p0, p1, p2, p3):
         
            """Calculates a dihedral angle.
            Args:
                p0, p1, p2, p3: The 3D coordinates of each atom in the dihedral
            Returns:
                The dihedral angle (in degrees).
            """
    
            b0 = -1.0*(p1 - p0)
            b1 = p2 - p1
            b2 = p3 - p2

            # normalize b1 
            b1 /= torch.sqrt(torch.sum(torch.square(b1), axis = 1)).unsqueeze(0).T

            # vector projections
            v = b0 - torch.sum(b0*b1, axis = 1).unsqueeze(1)*b1
            w = b2 - torch.sum(b2*b1, axis = 1).unsqueeze(1)*b1
            
           # angle between v and w in a plane is the torsion angle
            x = torch.sum(v*w, axis = 1)
            y = torch.sum(torch.cross(b1, v)*w, axis = 1)
            return torch.rad2deg(torch.atan2(y, x))

        phi0 = _compute_dihedrals(positions[:,4], positions[:,6], positions[:,8], positions[:,17])
        phi1 = _compute_dihedrals(positions[:,17], positions[:,19], positions[:,21], positions[:,30])
        phi2 = _compute_dihedrals(positions[:,30], positions[:,32], positions[:,34], positions[:,43])
        phi3 = _compute_dihedrals(positions[:,43], positions[:,45], positions[:,47], positions[:,56])
        phi4 = _compute_dihedrals(positions[:,56], positions[:,58], positions[:,60], positions[:,69])
        phi5 = _compute_dihedrals(positions[:,69], positions[:,71], positions[:,73], positions[:,82])
        phi6 = _compute_dihedrals(positions[:,82], positions[:,84], positions[:,86], positions[:,95])
        phi7 = _compute_dihedrals(positions[:,95], positions[:,97], positions[:,99], positions[:,108])
        phi8 = _compute_dihedrals(positions[:,108], positions[:,110], positions[:,112], positions[:,121])
    
        phis = torch.stack((phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8))
    
        psi0 = _compute_dihedrals(positions[:,6], positions[:,8], positions[:,17], positions[:,19])
        psi1 = _compute_dihedrals(positions[:,19], positions[:,21], positions[:,30], positions[:,32])
        psi2 = _compute_dihedrals(positions[:,32], positions[:,34], positions[:,43], positions[:,45])
        psi3 = _compute_dihedrals(positions[:,45], positions[:,47], positions[:,56], positions[:,58])
        psi4 = _compute_dihedrals(positions[:,58], positions[:,60], positions[:,69], positions[:,71])
        psi5 = _compute_dihedrals(positions[:,71], positions[:,73], positions[:,82], positions[:,84])
        psi6 = _compute_dihedrals(positions[:,84], positions[:,86], positions[:,95], positions[:,97])
        psi7 = _compute_dihedrals(positions[:,97], positions[:,99], positions[:,108], positions[:,110])
        psi8 = _compute_dihedrals(positions[:,110], positions[:,112], positions[:,121], positions[:,123])
        psis = torch.stack((psi0, psi1, phi2, psi3, psi4, psi5, psi6, psi7, psi8))
        return torch.cat((phis, psis), axis = 0)
    def positions_to_pairwise_distances(self, positions):
        n = positions.size()[0]
        D = positions.unsqueeze(2)
        D = D.expand(positions.size()[0], positions.size()[1], positions.size()[1], 3)
        D_t = D.transpose(1,2)
        D = D - D_t
        D = (torch.sum(torch.square(D), dim = -1))
        Ds = torch.nonzero(torch.flatten(torch.tril(D, diagonal = -1))).squeeze()
        D = torch.flatten(torch.tril(D, diagonal = -1))[Ds].squeeze()
        D = torch.reshape(D, (n, -1))
        return D


    def forward(self, positions, velos):
        with torch.no_grad():
            D = self.compute_aib9_embedding(positions).T
            D2 = self.positions_to_pairwise_distances(positions[:,:129])
            D3 = torch.cat((D, 1/D2), dim = -1)
        prediction = self.block(D3)
        return prediction

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = PredictNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)

# Create context
inpcrd = omm.app.AmberInpcrdFile('aib_unfolded.crd')
prmtop = omm.app.AmberPrmtopFile('aib_unfolded.prmtop')
system = prmtop.createSystem(nonbondedMethod=omm.app.NoCutoff, nonbondedCutoff=1.2*unit.nanometer,
        constraints=omm.app.HBonds)
integrator = ommt.integrators.BAOABIntegrator(500*unit.kelvin, 1/unit.picosecond, 0.001*unit.picoseconds)
platform = omm.Platform.getPlatformByName('OpenCL')
context = omm.Context(system, integrator, platform)

# Define basin indicator functions
def in_a(positions):
    a_center = np.array([-60, -40])
    if len(np.shape(positions)) != 3:
        positions = [positions] # Unsqueeze to keep dimensions consistent
    return np.array([utils.dihedral_difference(utils.compute_aib9_dihedrals(position), a_center) < 10 for position in positions])


def in_b(positions):
    b_center = np.array([60, 40])
    if len(np.shape(positions)) != 3:
        positions = [positions] # Unsqueeze to keep dimensions consistent
    return np.array([utils.dihedral_difference(utils.compute_aib9_dihedrals(position), b_center) < 10 for position in positions])

#Create constructor and run equilibrium simulations
C = committor_constructor(context, net, optimizer, in_a, in_b, n_reporters = 100)
C.sample_basins(np.load("AIB9_A_init_positions.npy"), np.load("AIB9_B_init_positions.npy"), n_samples = 1000)
C.save_data()

#Collect samples and iterate the committor
C = committor_constructor(context, net, optimizer, in_a, in_b, n_reporters = 100)
for i in range(10000):
    start_time = time.time()
    C.step(n_reporter_steps = 100000000, stride = 1000, fraction = 0)
    C.save_data()
    print(time.time() - start_time)
    
# Generate figure

# First, load and analyze data
C = committor_constructor(context, net, optimizer, in_a, in_b, n_reporters = 100)

a_confs = np.array(np.memmap(C.data_path + "a_sample_confs.dat", dtype = 'float64', mode = 'r', shape = (C.current_iteration, C.n_particles, 3)))
a_velos = np.array(np.memmap(C.data_path + "a_sample_velos.dat", dtype = 'float64', mode = 'r', shape = (C.current_iteration, C.n_particles, 3)))
        
b_confs = np.array(np.memmap(C.data_path + "b_sample_confs.dat", dtype = 'float64', mode = 'r', shape = (C.current_iteration, C.n_particles, 3)))
b_velos = np.array(np.memmap(C.data_path + "b_sample_velos.dat", dtype = 'float64', mode = 'r', shape = (C.current_iteration, C.n_particles, 3)))
        
a_dihedrals = np.array([utils.compute_aib9_dihedrals(position) for position in a_confs])
b_dihedrals = np.array([utils.compute_aib9_dihedrals(position) for position in b_confs])

all_dihedrals = (np.concatenate((a_dihedrals, b_dihedrals)))

all_confs = torch.tensor(np.concatenate((a_confs, b_confs))).to(C.device)
all_velos = torch.tensor(np.concatenate((a_velos, b_velos))).to(C.device)

targets = torch.sigmoid(C.net(all_confs, all_velos)).cpu().detach().numpy()

# Generate committor surfaces
H,D = utils.generate_2D_projection(all_dihedrals, targets)

# Generate PMF for AIB9
x = np.linspace(-180, 180, 50)
y = np.linspace(-180, 180, 50)
X,Y = np.meshgrid(x,y)
Z = np.load("AIB9_PMF.npy")

# Start plotting
matplotlib.rcParams['text.usetex'] = True
plt.tight_layout()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (12.8, 9.6))

# Subplot 1: Committor contour on the PMF
a_center = np.array([-60, -40])
b_center = np.array([60, 40])

times = (np.array(C.a_times) + np.array(C.b_times))*1e-8
ax1.contourf(X, Y, Z, cmap = 'mycmap', extend = 'max')
fig.colorbar(ax1.contour(X, Y, (H+np.flip(1 - H))/2, levels = 9, cmap='mycmap2'), label = r"$q(\phi, \psi)$", ax = ax1)
ax1.set_facecolor('white')
ax1.set_xlabel(r'$\textrm{$\phi$}$', size = 20)
ax1.set_ylabel(r'$\textrm{$\psi$}$', size = 20)
ax1.add_patch(plt.Circle(b_center, 10, linewidth = 2, color = 'black', fill = True))
ax1.add_patch(plt.Circle(a_center, 10, linewidth = 2, color = 'white', fill = True))
ax1.text(a_center[0]-5, a_center[1]-5, r"$\textbf{A}$", zorder = 2, color = 'black')
ax1.text(b_center[0]-4, b_center[1]-5, r"$\textbf{B}$", zorder = 2, color = 'white')

# Subplot 2: On-the-fly Rate Estimates
a_rolling_mean, a_rolling_std = utils.compute_rolling_statistics(np.array(C.a_rate_estimates), 200)
b_rolling_mean, b_rolling_std = utils.compute_rolling_statistics(np.array(C.b_rate_estimates), 200)
a_rolling_time, a_rolling_timestd = utils.compute_rolling_statistics(1/np.array(C.a_rate_estimates), 200)
b_rolling_time, b_rolling_timestd = utils.compute_rolling_statistics(1/np.array(C.b_rate_estimates), 200)

ax2.plot(times, a_rolling_mean, c = '#1B346C')
ax2.plot(times, b_rolling_mean, c = '#F54B1A')
ax2.plot(times, np.ones_like(np.array(a_rolling_mean))*(1/4.7), '--', c = 'black')
ax2.fill_between(times, np.array(a_rolling_mean) - np.array(a_rolling_std), np.array(a_rolling_mean) + np.array(a_rolling_std), color = '#1B346C', alpha = 0.5)
ax2.fill_between(times, np.array(b_rolling_mean) - np.array(b_rolling_std), np.array(b_rolling_mean) + np.array(b_rolling_std), color = '#F54B1A', alpha = 0.5)
ax2.set_yscale('log')
ax2.set_ylim(10**-2, 10**2)
ax2.legend([r'$\textrm{Estimated} \quad A \rightarrow B$', r'$\textrm{Estimated} \quad B \rightarrow A$', r'$\textrm{Empirical} \quad A \longleftrightarrow B$'])
ax2.set_ylabel(r'$\textrm{Rate Estimate} \, (\textrm{ns}^{-1})$')
ax2.set_xlabel(r'$\textrm{Total Sampling Time} \, (\textrm{ns})$')
ax2.set_ylim(1e-2, 1e2)

# Subplot 3: Log reaction probability
fig.colorbar(ax3.contourf(X, Y, -np.log((H+np.flip(1 - H))/2*(1 - (H+np.flip(1 - H))/2)), levels = 9, cmap='mycmap3'), label = r"$-\log(q)(1-q)$", ax = ax3)
ax3.set_xlabel(r'$\textrm{$\phi$}$', size = 20)
ax3.set_ylabel(r'$\textrm{$\psi$}$', size = 20)
ax3.add_patch(plt.Circle(b_center, 10, linewidth = 2, color = 'black', fill = True))
ax3.add_patch(plt.Circle(a_center, 10, linewidth = 2, color = 'white', fill = True))
ax3.text(a_center[0]-5, a_center[1]-5, r"$\textbf{A}$", zorder = 2, color = 'black')
ax3.text(b_center[0]-4, b_center[1]-5, r"$\textbf{B}$", zorder = 2, color = 'white')

# Subplot 4: Bar graph of MFPTs
ax4.hlines([0.1, 1, 10, 100], -0.25, 3, color = 'black', linestyles = 'dashed', alpha = 0.25, zorder = -1, label = '_lines')
true_mean = 4.7
ax4.bar(0.5, a_rolling_time[-1], width = 0.3, yerr = a_rolling_timestd[-1], capsize = 10, edgecolor = 'black', label = r'$\textrm{Estimated} \quad A \rightarrow B$', color = '#01ABE9')
ax4.bar(1, true_mean, width = 0.3, yerr = 0.69, capsize = 10, edgecolor = 'black', label = r'$\textrm{Empirical} \quad A \rightarrow B$', color = 'white')
ax4.bar(1.5, b_rolling_time[-1], width = 0.3, yerr = b_rolling_timestd[-1], capsize = 10, edgecolor = 'black', label = r'$\textrm{Estimated} \quad B \rightarrow A$', color = '#F54B1A')
ax4.set_yscale('log')
ax4.set_ylim(0.01, 1000)
ax4.set_xlim(-0.25, 2)
ax4.set_ylabel(r'$\textrm{Mean First Passage Time \, (ns)}$')
ax4.tick_params(
            axis='x',          
            which='both',      
            bottom=False,     
            top=False,
            labelbottom=False) 
ax4.legend()  
plt.savefig("AIB_fig.pdf")
plt.show()