import sampling
import training
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# For plotting
matplotlib.rcParams['text.usetex'] = True
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("mycmap", ['#1B346C','#01ABE9','#F1F8F1','#F54B1A'])
cmap2 = LinearSegmentedColormap.from_list("mycmap2", ['#ffffff','#000000','#ffffff'])
cmap3 = LinearSegmentedColormap.from_list("mycmap3", ['#1B346C','#ffffff','#F54B1A'])
cmap4 = LinearSegmentedColormap.from_list("mycmap4", ['#ffffff','#F54B1A'])
cmap5 = LinearSegmentedColormap.from_list("mycmap5", ['#000000','#000000'])
cmap6 = LinearSegmentedColormap.from_list("mycmap5", ['#ffffff','#ffffff'])
from matplotlib.cm import register_cmap
register_cmap(name="mycmap",cmap=cmap)
register_cmap(name="mycmap2",cmap=cmap2)
register_cmap(name="mycmap3",cmap=cmap3)
register_cmap(name="mycmap4",cmap=cmap4)
register_cmap(name="mycmap5",cmap=cmap5)
register_cmap(name="mycmap6",cmap=cmap6)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# Set a deterministic RNG seed
torch.manual_seed(42)

# Define our two-channel potential

def dist(x, y):
    return torch.sqrt(torch.sum(torch.square(x-y), axis = -1))
    
def V(x):

    A = torch.tensor([-20, -10, -17, 1.5])
    a = torch.tensor([-1, -1, -6.5, 0.7])
    b = torch.tensor([0, 0, 11, 0.6])
    c = torch.tensor([-10, -10, -6.5, 0.7])
    x0 = torch.tensor([1, 0, -0.5, -1])
    y0 = torch.tensor([0, 0.5, 1.5, 1])
    def _gau(x, idx): # Defining a multidimensional Gaussian
        return A[idx]*torch.exp(a[idx]*torch.square(x[:,0] - x0[idx]) + b[idx]*(x[:,0] - x0[idx])*(x[:,1] - y0[idx]) +
        c[idx]*torch.square(x[:,1] - y0[idx]))
    
    return _gau(x, 0) + _gau(x,1) +\
            _gau(x, 2) + _gau(x,3)

class CommittorNet(torch.nn.Module):
    def __init__(self, dim):
        super(CommittorNet, self).__init__()
        self.dim = dim
        block = [torch.nn.Linear(dim, 50),
                      torch.nn.Tanh(),
                      torch.nn.Linear(50, 1),]
        self.Block = torch.nn.Sequential(*block)
    
    def forward(self, x):
        prediction = self.Block(x)
        return prediction.squeeze()
# Set dynamical parameters for the optimiziation
k = torch.tensor([100.]).to(device) # The harmonic force constant for umbrella sampling around committor values
beta = torch.tensor([1.0]).to(device) # Inverse kT for our system
sampling_beta = torch.tensor([0]).to(device) # We can sample at a higher temperature
gamma = torch.tensor([1]).to(device) # Friction coefficient for Langevin dynamics
step_size = torch.tensor([1e-3]).to(device) # The step size for each step of Langevin dynamics

# Set hyperparameters for the optimization
n_windows = 11
dim = 2
a_center = torch.tensor([-0.5, 1.5]).to(device)
b_center = torch.tensor([0.5, 0.]).to(device)
cutoff = torch.tensor([0.1]).to(device)

# Initialize evenly-spaced configurations that linearly interpolate the basin centers
init_data = torch.zeros((n_windows, dim)) 
for d in range(dim):
    init_data[:,d] = torch.linspace(a_center[d], b_center[d], n_windows)

net = CommittorNet(2).to(device).double()

print("Initial representation of the committor has been trained!")

# Run the optimization
n_trials = 1
n_opt_steps = 3000
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-2)
xs = init_data.to(device)
us = torch.linspace(0, 1, n_windows).to(device)
n_reporter_trajectories = torch.tensor([100]).to(device)
n_reporter_steps = torch.tensor([5]).to(device)
batch_size = 11
# For plotting:
x = torch.linspace(-2,1.5,100)
y = torch.linspace(-1,2.5,100)
X, Y = torch.meshgrid(x, y)
grid_input = torch.cat((X.unsqueeze(-1),Y.unsqueeze(-1)), dim = -1).to(device)
print(V(grid_input.reshape(-1,2)))
print(V(grid_input.reshape(-1,2)).size())
print(X.size())
V_surface = V(grid_input.reshape((-1, 2))).reshape(X.size())
true_committor = np.transpose((np.load('MB_committor.npy')))

# Calculate Flux out of the basins

print("Calculating Flux...")
a_escape_times, a_escape_confs = sampling.flux_sample(V, beta, gamma, step_size, a_center, b_center, cutoff, 1000, stride = 1)
b_escape_times, b_escape_confs = sampling.flux_sample(V, beta, gamma, step_size, b_center, a_center, cutoff, 1000, stride = 1)

a_escape_confs_list = a_escape_confs.clone().reshape([-1,2]).to(device).double().detach()
b_escape_confs_list = b_escape_confs.clone().reshape([-1,2]).to(device).double().detach()

escape_confs = torch.cat([a_escape_confs, b_escape_confs], axis = 0).reshape(-1,2)

#torch.manual_seed(2)
running_a_exit_confs = []
running_b_exit_confs = []
a_exit_indices = torch.randperm(len(a_escape_confs_list))
b_exit_indices = torch.randperm(len(b_escape_confs_list))
a_transit_history = [0]
b_transit_history = [0]
for trial in range(1): # Can run multiple trials, if you'd like
    a_transit = False
    last_a_transit = 0
    b_transit = False
    last_b_transit = 0
    a_means = []
    a_times = []
    b_means = []
    b_times = []
    losses = []
    log_losses = []
    a_index_counter = 0
    b_index_counter = 0
    for step in range(n_opt_steps):
        optimizer.zero_grad()
        net.zero_grad()

        # Sample some configurations
        
        if step == 0 or a_transit == True:
            a_xs = a_escape_confs_list[a_exit_indices[a_index_counter]]
            a_xs = a_xs.unsqueeze(0)
            a_transit = False
            running_a_exit_confs.append(a_xs.numpy())
        
        else:
            a_reporter_energies = V(a_running_short_reporters.reshape([-1,2])[last_a_transit:])
            a_weights = torch.sigmoid(net(a_running_short_reporters.squeeze().reshape([-1, 2])))[last_a_transit:].detach()
            a_weights = torch.where(dist(a_running_short_reporters.squeeze().reshape([-1, 2])[last_a_transit:], a_center) < cutoff, 0, a_weights)
            _, a_indices = torch.sort(a_weights, descending = True)
            a_indices = a_indices[:1]
            a_xs = a_running_short_reporters.squeeze().reshape([-1, 2])[last_a_transit:][a_indices]
        
        if step == 0 or b_transit == True:
            b_xs = b_escape_confs_list[b_exit_indices[b_index_counter]]
            b_xs = b_xs.unsqueeze(0)
            b_transit = False
            running_b_exit_confs.append(b_xs.numpy())
        else:
            b_reporter_energies = V(b_running_short_reporters.reshape([-1,2])[last_b_transit:])
            b_weights = torch.sigmoid(-net(b_running_short_reporters.squeeze().reshape([-1, 2])))[ last_b_transit:].detach()
            b_weights = torch.where(dist(b_running_short_reporters.squeeze().reshape([-1, 2])[ last_b_transit:], b_center) < cutoff, 0, b_weights)
            _, b_indices = torch.sort(b_weights, descending = True)
            b_indices = b_indices[:1]
            b_xs = b_running_short_reporters.squeeze().reshape([-1, 2])[last_b_transit:][b_indices]
    
        a_short_reporters, a_short_times = sampling.take_reporter_steps(a_xs, V, beta, gamma, step_size, n_reporter_trajectories, n_reporter_steps, a_center, b_center, cutoff, adaptive = True)
        b_short_reporters, b_short_times = sampling.take_reporter_steps(b_xs, V, beta, gamma, step_size, n_reporter_trajectories, n_reporter_steps, a_center, b_center, cutoff, adaptive = True)
        a_times.append(np.mean(a_short_times))
        b_times.append(np.mean(b_short_times))

        # Keep a memory of sampled configurations
        if step == 0:
            a_running_xs = a_xs.detach()
            b_running_xs = b_xs.detach()
            running_xs = torch.cat((a_running_xs, b_running_xs), axis = 0)
            a_running_short_reporters = a_short_reporters
            b_running_short_reporters = b_short_reporters
            running_short_reporters = torch.cat((a_running_short_reporters, b_running_short_reporters), axis = 0)
        else:
            a_running_xs = torch.cat((a_running_xs.detach(), a_xs.detach()))
            b_running_xs = torch.cat((b_running_xs.detach(), b_xs.detach()))
            a_running_short_reporters = torch.cat((a_running_short_reporters.detach(), a_short_reporters), axis = 0)
            b_running_short_reporters = torch.cat((b_running_short_reporters.detach(), b_short_reporters), axis = 0)
            running_short_reporters = torch.cat((a_running_short_reporters, b_running_short_reporters), axis = 0)
            running_xs = torch.cat((a_running_xs, b_running_xs), axis = 0)
        
        for j in range(1): # Can go through multiple optimization steps per data collection step, if needed
            with torch.no_grad():
                a_short_targets, b_short_targets, a_short_var, a_short_means = sampling.calculate_committor_estimates(running_short_reporters.reshape([-1, 2]), net, a_center, b_center, cutoff, n_reporter_trajectories)
            batch_size = running_xs.size()[0]
            for m in range(100):
                permutation = torch.randperm(running_xs.size()[0])
                for i in range(0, len(running_xs), batch_size):
                    print(m)
                    optimizer.zero_grad()
                    net.zero_grad()
                    indices = permutation[i:i+batch_size]
                    loss, individual_losses =  1*training.half_loss(net, running_xs.to(device)[indices], a_short_targets.to(device)[indices], b_short_targets.to(device)[indices])
                    log_loss, individual_log_loss = 1*training.half_log_loss(net, running_xs.to(device)[indices], a_short_targets.to(device)[indices], b_short_targets.to(device)[indices], a_short_targets.to(device)[indices], b_short_targets.to(device)[indices])
                    total_loss = 0*loss + 1*log_loss
                    total_loss.backward()
                    optimizer.step()
        
        # Estimate rates
        with torch.no_grad():
            a_exit_tensor = torch.tensor(np.array(a_escape_confs)).squeeze()
            b_exit_tensor = torch.tensor(np.array(b_escape_confs)).squeeze()
            a_rate_estimates = 1/torch.mean(a_escape_times)*torch.mean(torch.sigmoid(net(a_exit_tensor)))
            b_rate_estimates = 1/torch.mean(b_escape_times)*torch.mean(1 - torch.sigmoid(net(b_exit_tensor)))

            a_rate_mean = torch.mean(a_rate_estimates)
            a_means.append(a_rate_mean.cpu().detach().numpy())
            b_rate_mean = torch.mean(b_rate_estimates)
            b_means.append(b_rate_mean.cpu().detach().numpy())

        
        # Report to the command line
        print(f"Step {step}: Rate Estimate = {a_rate_mean.item()}; Loss = {log_loss.item()}")#, Log Loss 2 = {log_loss_2.item()}")
        print(a_index_counter, b_index_counter)
        

        # Check whether or not a sampling chain has reached the basin
        if torch.min(torch.sqrt(torch.sum(torch.square(a_running_short_reporters.reshape([-1, 2])[last_a_transit:] - b_center), axis = -1))) < cutoff:
            print("A Transit!")
            a_transit = True
            last_a_transit = len(a_running_short_reporters.reshape([-1, 2]))
            a_index_counter += 1
            a_transit_history.append(int(len(running_xs.reshape([-1, 2]))/2))
            
        if torch.min(torch.sqrt(torch.sum(torch.square(b_running_short_reporters.reshape([-1, 2])[last_b_transit:] - a_center), axis = -1))) < cutoff:
            print("B Transit!")
            b_transit = True
            last_b_transit = len(b_running_short_reporters.reshape([-1, 2]))
            b_index_counter += 1
            b_transit_history.append(int(len(running_xs.reshape([-1, 2]))/2))

        if step % 10 == 9:
            plt.tight_layout()
            fig, axs  = plt.subplot_mosaic([['a', 'b', 'c'], ['a', 'b', 'd']], width_ratios = [1., 1., 0.75])
            fig.set_size_inches(15.2, 4.8)
            fig.colorbar(axs['c'].contourf(X, Y, torch.sigmoid(net(grid_input)).cpu().detach().numpy()-true_committor, levels = np.linspace(-0.1, .1, 101), cmap = 'mycmap3', extend = 'both'), ax = axs['c'], ticks = np.linspace(-0.1, 0.1, 5))
            axs['c'].add_patch(plt.Circle((-0.5, 1.5), cutoff, linewidth = 2, color = 'black', fill = True))
            axs['c'].add_patch(plt.Circle((0.5, 0), cutoff, linewidth = 2, color = 'black', fill = True))
            axs['c'].text(-0.54, 1.45, "A", weight = 'bold', size = 8, color = 'white')
            axs['c'].text(0.46, -0.05, "B", weight = 'bold', size = 8, color = 'white')
            axs['c'].set_xlabel(r"$x$", size = 16)
            axs['c'].set_ylabel(r"$y$", size = 16)
            axs['c'].contour(X, Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0., 0., 1), cmap = 'mycmap5')
            axs['c'].contourf(X,Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0, 1e20, 2), cmap = 'mycmap6', zorder = 100)
            axs['c'].set_xlim(-1.7, 1.2)
            axs['c'].set_ylim(-0.4, 2.1)
            axs['c'].text(-1.65, -0.25, r"\textrm{\textbf{(c)}}", weight = 'bold', size = 12, zorder = 200)

            fig.colorbar(axs['d'].contourf(X, Y, np.log10((torch.sigmoid(net(grid_input))*(1 - torch.sigmoid(net(grid_input)))).cpu().detach().numpy())-np.log10(true_committor*(1-true_committor)), levels = np.linspace(-1, 1, 101), cmap = 'mycmap3', extend = 'both'), ax = axs['d'], ticks = np.linspace(-1, 1, 5))
            axs['d'].add_patch(plt.Circle((-0.5, 1.5), cutoff, linewidth = 2, color = 'black', fill = True))
            axs['d'].add_patch(plt.Circle((0.5, 0), cutoff, linewidth = 2, color = 'black', fill = True))
            axs['d'].text(-0.54, 1.45, "A", weight = 'bold', size = 8, color = 'white')
            axs['d'].text(0.46, -0.05, "B", weight = 'bold', size = 8, color = 'white')
            axs['d'].set_xlabel(r"$x$", size = 16)
            axs['d'].set_ylabel(r"$y$", size = 16)
            axs['d'].contour(X, Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0., 0., 1), cmap = 'mycmap5')
            axs['d'].contourf(X,Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0, 1e20, 2), cmap = 'mycmap6', zorder = 100)
            axs['d'].set_xlim(-1.7, 1.2)
            axs['d'].set_ylim(-0.4, 2.1)
            axs['d'].text(-1.65, -0.25, r"\textrm{\textbf{(d)}}", weight = 'bold', size = 12, zorder = 200)

            axs['a'].contourf(X,Y, V_surface.cpu().detach().numpy(), levels = np.linspace(-15, 0, 20), cmap = 'mycmap')
            fig.colorbar(axs['a'].contour(X, Y, torch.sigmoid(net(grid_input)).cpu().detach().numpy(), levels = np.linspace(0.1, 0.9, 9), cmap = 'mycmap2'), ax = axs['a'], ticks = np.linspace(0, 1, 11))
            axs['a'].scatter(torch.reshape(running_xs, [-1, 2]).detach().numpy()[:,0], torch.reshape(running_xs, [-1, 2]).detach().numpy()[:,1], color = 'red', alpha = 1)
            axs['a'].scatter(torch.reshape(a_running_xs, [-1, 2]).detach().numpy()[-1,0], torch.reshape(a_running_xs, [-1, 2]).detach().numpy()[-1,1], color = 'blue', alpha = 1)
            axs['a'].scatter(torch.reshape(b_running_xs, [-1, 2]).detach().numpy()[-1,0], torch.reshape(b_running_xs, [-1, 2]).detach().numpy()[-1,1], color = 'green', alpha = 1)
            axs['a'].add_patch(plt.Circle((0.5, 0), cutoff, linewidth = 2, color = 'black', fill = True))
            axs['a'].add_patch(plt.Circle((-0.5, 1.5), cutoff, linewidth = 2, color = 'black', fill = True))
            axs['a'].text(-0.54, 1.45, "A", weight = 'bold', size = 15, color = 'white')
            axs['a'].text(0.46, -0.05, "B", weight = 'bold', size = 15, color = 'white')
            axs['a'].set_xlabel(r"$x$", size = 16)
            axs['a'].set_ylabel(r"$y$", size = 16)
            #axs['a'].contour(X, Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0., 0., 1), cmap = 'mycmap5')
            axs['a'].contourf(X,Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0, 1e20, 2), cmap = 'mycmap6', zorder = 100)
            axs['a'].set_xlim(-1.7, 1.2)
            axs['a'].set_ylim(-0.4, 2.1)
            axs['a'].text(-1.65, -0.30, r"\textrm{\textbf{(a)}}", weight = 'bold', size = 20, zorder = 200)

            axs['b'].plot(np.arange(step+1)*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.array(a_means), c = '#1B346C')
            axs['b'].plot(np.arange(step+1)*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.array(b_means), c = '#F54B1A')
            #axs['b'].plot(np.arange(step+1)*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 500)*step_size.cpu().detach().numpy(), np.array(losses), c = 'black')
            #axs['b'].plot(np.arange(step+1)*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 100)*step_size.cpu().detach().numpy(), np.array(middle_losses), c = 'red')
            #ax2.vlines(1/(9.18e-8*2), 0, 1, color = '#F54B1A', linestyles = 'dashed')
            #ax2.plot(np.linspace(0, 1/(9.18e-8*2), 10), np.ones(10)*9.18e-8*2, '--', c = 'black')
            axs['b'].plot(np.arange(len(a_means))*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.ones_like(a_means)*3e-4, '--', c = '#1B346C')
            axs['b'].plot(np.arange(len(a_means))*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.ones_like(a_means)*2e-2, '--', c = '#F54B1A')
            axs['b'].set_yscale('log')
            axs['b'].set_ylim(7e-7, 20)
            axs['b'].set_xlabel(r"$\textrm{Sampling Time ($\tau$)}$", size = 12)
            axs['b'].set_ylabel(r"$\textrm{Rate ($\tau^{-1}$)}$", size = 12)
            axs['b'].legend([r'$\textrm{Rate Estimate A to B}$', r'$\textrm{Rate Estimate B to A}$', r'$\textrm{Analytical Rate A to B}$', r'$\textrm{Analytical Rate B to A}$'], prop={'size': 12})
            #axs['b'].text(-20, 1.3e-7, r"\textrm{\textbf{(b)}}", weight = 'bold', size = 20, zorder = 200)
            plt.tight_layout()
            fig.savefig("MB.pdf")
            plt.close()
            
            plt.plot(losses)
            plt.plot(log_losses)
            plt.yscale('log')
            plt.savefig("Losses.pdf")
            plt.close()

# Plot the final committor
plt.contourf(X,Y, V_surface, levels = np.linspace(-5, 0, 15), cmap = 'mycmap')
plt.contour(X, Y, net(grid_input).detach().numpy(), levels = np.linspace(0.1, 0.9, n_windows), cmap = 'mycmap2')
plt.savefig("./Committor_MB.pdf")
#plt.close()
print("SAVING")
torch.save(net.state_dict(), "./MB.pt")


