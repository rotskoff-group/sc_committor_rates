# Application to Underdamped Alanine Dipeptide
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
import committor_constructor
import utils

# Create committor network
class PredictNet(torch.nn.Module):
    def __init__(self):
        super(PredictNet, self).__init__()
        block = [nn.Linear(45, 100),
                      nn.LeakyReLU(),
                      nn.Linear(100, 100),
                      nn.LeakyReLU(),
                      nn.Linear(100, 100),
                      nn.LeakyReLU(),
                      nn.Linear(100, 100),
                      nn.LeakyReLU(),
                      nn.Linear(100, 1)]
        self.block = nn.Sequential(*block)

    def positions_to_pairwise_distances(self, positions):
        positions = torch.cat([positions[:,0].unsqueeze(1),
                             positions[:,4].unsqueeze(1),
                             positions[:,5].unsqueeze(1),
                             positions[:,6].unsqueeze(1),
                             positions[:,8].unsqueeze(1),
                             positions[:,10].unsqueeze(1),
                             positions[:,14].unsqueeze(1),
                             positions[:,15].unsqueeze(1),
                             positions[:,16].unsqueeze(1),
                             positions[:,18].unsqueeze(1)], dim = 1)
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
            D = self.positions_to_pairwise_distances(positions)
            D = torch.flatten(torch.sqrt(D), start_dim = 1)
            #V = self.positions_to_pairwise_distances(velos)
            #V = torch.flatten(torch.sqrt(V), start_dim = 1)
            #DV = torch.cat((D,V), axis = -1)
        prediction = self.block(D)
        return (prediction)
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = PredictNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)

# Create context
top = omm_app.GromacsTopFile('topol_vacuum.top')
gro = omm_app.GromacsGroFile('alanine_dipeptide_vacuum_boxed.gro')
reporter_system = top.createSystem(nonbondedMethod = omm_app.NoCutoff, constraints = omm_app.HBonds)
T = 300 * unit.kelvin
integrator = omm.BrownianIntegrator(T, 1, 0.01*unit.femtosecond)
integrator.setRandomNumberSeed(42)
platform = omm.Platform.getPlatformByName('OpenCL')
context = omm.Context(reporter_system, integrator, platform)


# Define basin indicator functions
def in_a(positions):
    a_center = np.array([-150, 170])
    if len(np.shape(positions)) != 3:
        positions = [positions] # Unsqueeze to keep dimensions consistent
    return np.array([utils.dihedral_difference(utils.compute_alanine_dihedrals(position), a_center) < 10 for position in positions])


def in_b(positions):
    b_center = np.array([90, -50])
    if len(np.shape(positions)) != 3:
        positions = [positions] # Unsqueeze to keep dimensions consistent
    return np.array([utils.dihedral_difference(utils.compute_alanine_dihedrals(position), b_center) < 10 for position in positions])

#Create constructor and run equilibrium simulations
C = committor_constructor(context, net, optimizer, in_a, in_b, n_reporters = 100)
C.sample_basins(gro.positions, np.load("../Final/AD_Underdamped/AD_B_escape_confs.npy")[-1], n_samples = 1000)
C.save_data()

#Collect samples and iterate the committor
C = committor_constructor(context, net, optimizer, in_a, in_b, n_reporters = 100)
for i in range(10000):
    start_time = time.time()
    C.step(n_reporter_steps = 100, stride = 10, fraction = 0)
    C.save_data()
    print(time.time() - start_time)
    
# Generate figure

# First, load and analyze data
C = committor_constructor(context, net, optimizer, in_a, in_b, n_reporters = 100)

a_confs = np.array(np.memmap(C.data_path + "a_sample_confs.dat", dtype = 'float64', mode = 'r', shape = (C.current_iteration, C.n_particles, 3)))
a_velos = np.array(np.memmap(C.data_path + "a_sample_velos.dat", dtype = 'float64', mode = 'r', shape = (C.current_iteration, C.n_particles, 3)))
        
b_confs = np.array(np.memmap(C.data_path + "b_sample_confs.dat", dtype = 'float64', mode = 'r', shape = (C.current_iteration, C.n_particles, 3)))
b_velos = np.array(np.memmap(C.data_path + "b_sample_velos.dat", dtype = 'float64', mode = 'r', shape = (C.current_iteration, C.n_particles, 3)))
        
a_dihedrals = np.array([utils.compute_alanine_dihedrals(position) for position in a_confs])
b_dihedrals = np.array([utils.compute_alanine_dihedrals(position) for position in b_confs])

all_dihedrals = (np.concatenate((a_dihedrals, b_dihedrals)))

all_confs = torch.tensor(np.concatenate((a_confs, b_confs))).to(C.device)
all_velos = torch.tensor(np.concatenate((a_velos, b_velos))).to(C.device)

targets = torch.sigmoid(C.net(all_confs, all_velos)).cpu().detach().numpy()

# Generate committor surfaces
H,D = utils.generate_2D_projection(all_dihedrals, targets)

# Generate PMF for alanine dipeptide
plt.rcParams['text.usetex'] = True
x = np.linspace(-180, 180, 50)
y = np.linspace(-180, 180, 50)
X, Y = np.meshgrid(x,y)
surface = utils.generate_AD_PMF('./PMF_vacuum.npy')
Z = utils.generate_pmf_graph(surface)    

# Start plotting
plt.tight_layout()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (12.8, 9.6))

# Subplot 1: Committor contour on the PMF
a_center = np.array([-150, 170])
other_a_center = np.array([-150, 150])
b_center = np.array([90, -50])

times = (np.array(C.a_times) + np.array(C.b_times))*1e-8
ax1.contourf(X, Y, Z, levels = np.linspace(-5, 30, 30) , cmap = 'mycmap')
ax1.set_xlabel(r'$\textrm{$\phi$}$', size = 20)
ax1.set_ylabel(r'$\textrm{$\psi$}$', size = 20)
fig.colorbar(ax1.contour(X, Y, H, levels = 9, cmap='mycmap2'), label = r"$q(\phi, \psi)$", ax = ax1)
ax1.add_patch(plt.Circle(b_center, 10, linewidth = 2, color = 'black', fill = True))
ax1.add_patch(plt.Circle(a_center, 10, linewidth = 2, color = 'white', fill = True))
ax1.text(-155, 165, r"$\textbf{A}$", zorder = 2, color = 'black')
ax1.text(86, -55, r"$\textbf{B}$", zorder = 2, color = 'white')

# Subplot 2: On-the-fly Rate Estimates

a_rolling_mean, a_rolling_std = utils.compute_rolling_statistics(np.array(C.a_rate_estimates), 200)
b_rolling_mean, b_rolling_std = utils.compute_rolling_statistics(np.array(C.b_rate_estimates), 200)
a_rolling_time, a_rolling_timestd = utils.compute_rolling_statistics(1/np.array(C.a_rate_estimates), 200)
b_rolling_time, b_rolling_timestd = utils.compute_rolling_statistics(1/np.array(C.b_rate_estimates), 200)

ax2.plot(np.array(times), a_rolling_mean, c = '#1B346C', label = r'$\textrm{Estimated} \quad A \rightarrow B$')
ax2.plot(np.array(times), np.ones_like(np.array(a_rolling_mean))*(1/31), '--', c = '#1B346C', label = r'$\textrm{Empirical} \quad A \rightarrow B$')
ax2.fill_between(np.array(times), np.array(a_rolling_mean) - np.array(a_rolling_std), np.array(a_rolling_mean) + np.array(a_rolling_std), color = '#1B346C', alpha = 0.5)
ax2.plot(np.array(times), b_rolling_mean, c = '#F54B1A', label = r'$\textrm{Estimated} \quad B \rightarrow A$')
ax2.plot(np.array(times), np.ones_like(np.array(b_rolling_mean))*(1/.3), '--', c = '#F54B1A', label = r'$\textrm{Empirical} \quad B \rightarrow A$')
ax2.fill_between(np.array(times), np.array(b_rolling_mean) - np.array(b_rolling_std), np.array(b_rolling_mean) + np.array(b_rolling_std), color = '#F54B1A', alpha = 0.5)
ax2.set_yscale('log')
ax2.set_ylabel(r'$\textrm{Rate Estimate} \, (\textrm{ns}^{-1})$')
ax2.set_xlabel(r'$\textrm{Total Sampling Time} \, (\textrm{ns})$')
ax2.legend()
ax2.set_ylim(1e-4, 1e6)

# Subplot 3: Log reaction probability
ax3.set_xlabel(r'$\textrm{$\phi$}$', size = 20)
ax3.set_ylabel(r'$\textrm{$\psi$}$', size = 20)
fig.colorbar(ax3.contourf(X, Y, -D, levels = 15, cmap='mycmap3'), label = r"$- \log(q(1 - q))$", ax = ax3)
ax3.add_patch(plt.Circle(b_center, 10, linewidth = 2, color = 'black', fill = True))
ax3.add_patch(plt.Circle(a_center, 10, linewidth = 2, color = 'white', fill = True))
ax3.text(-155, 165, r"$\textbf{A}$", zorder = 2, color = 'black')
ax3.text(86, -55, r"$\textbf{B}$", zorder = 2, color = 'white')

# Subplot 4: Bar graph of MFPTs
a_true_mean = 38
a_true_std = 3
b_trues = [0.602, 0.666, .181, .079, .319]
b_true_mean = np.mean(b_trues)
b_true_std = np.std(b_trues)/np.sqrt(5)

ax4.hlines([0.1, 1, 10, 100], -0.25, 3, color = 'black', linestyles = 'dashed', alpha = 0.25, zorder = -1, label = '_lines')
ax4.bar(0.5, a_rolling_time[-1], width = 0.3, yerr = a_rolling_timestd[-1], capsize = 10, edgecolor = 'black', label = r'$\textrm{Estimated} \quad A \rightarrow B$', color = '#01ABE9', hatch = '//')
ax4.bar(1, a_true_mean, width = 0.3, yerr = a_true_std, capsize = 10, edgecolor = 'black', label = r'$\textrm{Empirical} \quad A \rightarrow B$', color = '#01ABE9')
ax4.bar(2, b_rolling_time[-1], width = 0.3, yerr = b_rolling_timestd[-1], capsize = 10, edgecolor = 'black', label = r'$\textrm{Estimated} \quad B \rightarrow A$', color = '#F54B1A', hatch = '//')
ax4.bar(2.5, b_true_mean, width = 0.3, yerr = b_true_std, capsize = 10, edgecolor = 'black', label = r'$\textrm{Empirical} \quad B \rightarrow A$', color = '#F54B1A')
ax4.set_yscale('log')
ax4.set_ylim(0.01, 1000)
ax4.set_xlim(-0.25, 3)
ax4.set_ylabel(r'$\textrm{Mean First Passage Time \, (ns)}$')
ax4.tick_params(
    axis='x',          
    which='both',      
    bottom=False,     
    top=False,
    labelbottom=False) 
ax4.legend()

plt.savefig('AD_Figure_Overdamped.pdf')
plt.show()