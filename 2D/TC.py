import sampling
import training
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# For plotting
matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
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
torch.manual_seed(2)

# Define our two-channel potential

def dist(x, y):
    return torch.sqrt(torch.sum(torch.square(x-y), axis = -1))
    
def V(x):
    return 1*(3*torch.exp(-x[:,0]**2 - (x[:,1] - (1/3))**2) - 3*torch.exp(-x[:,0]**2 - (x[:,1]-(5/3))**2) - 5*torch.exp(-(x[:,0]-1)**2 - x[:,1]**2) - 5*torch.exp(-(x[:,0]+1)**2 - x[:,1]**2) + 0.2*x[:,0]**4 +0.2*(x[:,1]-(1/3))**4)

class CommittorNet(torch.nn.Module):
    def __init__(self, dim):
        super(CommittorNet, self).__init__()
        self.dim = dim
        block = [torch.nn.Linear(dim, 100),
                      torch.nn.Tanh(),
                      torch.nn.Linear(100,1)]
        self.Block = torch.nn.Sequential(*block)
    
    def forward(self, x):
        prediction = self.Block(x)
        return prediction.squeeze()

# Set dynamical parameters for the optimiziation
k = torch.tensor([100.]).to(device) # The harmonic force constant for umbrella sampling around committor values
beta = torch.tensor([6.67]).to(device) # Inverse kT for our system
sampling_beta = torch.tensor([0.]).to(device) # We can sample at a higher temperature
gamma = torch.tensor([1]).to(device) # Friction coefficient for Langevin dynamics
step_size = torch.tensor([1e-2]).to(device) # The step size for each step of Langevin dynamics

# Set hyperparameters for the optimization
n_windows = 11
dim = 2
a_center = torch.tensor([-1., 0.]).to(device)
b_center = torch.tensor([1., 0.]).to(device)
cutoff = torch.tensor([0.2]).to(device)

# Initialize evenly-spaced configurations that linearly interpolate the basin centers
init_data = torch.zeros((n_windows, dim)) 
for d in range(dim):
    init_data[:,d] = torch.linspace(a_center[d], b_center[d], n_windows)

# Train a naive initial committor representation to linearly interpolate between the basins
net = CommittorNet(2).to(device).double()

print("Initial representation of the committor has been trained!")

# Run the optimization
n_trials = 10
n_opt_steps = 300
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-2)
xs = init_data.to(device)
us = torch.linspace(0, 1, n_windows).to(device)
n_reporter_trajectories = torch.tensor([1000]).to(device)
n_reporter_steps = torch.tensor([5]).to(device)
batch_size = 11
# For plotting:
x = torch.linspace(-2,2,200)
y = torch.linspace(-1.5,2.5,200)
X, Y = torch.meshgrid(x, y)
grid_input = torch.cat((X.unsqueeze(-1),Y.unsqueeze(-1)), dim = -1).to(device)
V_surface = V(grid_input.reshape((-1, 2))).reshape(X.size())
true_committor = np.transpose((np.load('FD_committor.npy')))

# Grab our flux calculation results
a_escape_times = torch.tensor(np.load('Two_channel_escape_times_a.npy')).to(device).double()
a_escape_confs = torch.tensor(np.load('Two_channel_escape_confs_a.npy')).to(device).double()
b_escape_times = torch.tensor(np.load('Two_channel_escape_times_b.npy')).to(device).double()
b_escape_confs = torch.tensor(np.load('Two_channel_escape_confs_b.npy')).to(device).double()

a_escape_confs_list =torch.tensor(np.load('Two_channel_escape_confs_a.npy')).reshape([-1,2]).double()
b_escape_confs_list =torch.tensor(np.load('Two_channel_escape_confs_b.npy')).reshape([-1,2]).double()

escape_confs = torch.cat([a_escape_confs, b_escape_confs], axis = 0).reshape(-1,2)
print(escape_confs.size())

all_xs = None
all_reporters = None
all_short_reporters = None
n_windows = 1
running_a_exit_confs = []
running_b_exit_confs = []
a_exit_indices = torch.randperm(len(a_escape_confs_list))
b_exit_indices = torch.randperm(len(b_escape_confs_list))
passage_times = []
a_transit_history = [0]
b_transit_history = [0]
for trial in range(1):
    reg_scheduler = 0.0
    a_transit = False
    last_a_transit = 0
    b_transit = False
    last_b_transit = 0
    epsilon = 0.0
    a_means = []
    a_stds = []
    a_times = []
    b_means = []
    b_stds = []
    b_times = []
    losses = []
    log_losses = []
    MFPT_losses = []
    MFPT_rate_estimates = []
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
            if torch.rand(1) < epsilon:
                a_indices = torch.randperm(len(a_running_short_reporters.reshape([-1, 2])[last_a_transit:]))[:n_windows]
            else:
                a_reporter_energies = V(a_running_short_reporters.reshape([-1,2])[last_a_transit:])
                a_weights = torch.sigmoid(net(a_running_short_reporters.squeeze().reshape([-1, 2])))[last_a_transit:].detach()
                a_weights = torch.where(dist(a_running_short_reporters.squeeze().reshape([-1, 2])[last_a_transit:], a_center) < cutoff, 0, a_weights)
                a_weights = a_weights*torch.exp(-sampling_beta*a_reporter_energies)
                a_indices = torch.multinomial(a_weights, n_windows)
                _, a_indices = torch.sort(a_weights, descending = True)
                a_indices = a_indices[:n_windows]
            a_xs = a_running_short_reporters.squeeze().reshape([-1, 2])[last_a_transit:][a_indices]
        
        if step == 0 or b_transit == True:
            b_xs = b_escape_confs_list[b_exit_indices[b_index_counter]]
            b_xs = b_xs.unsqueeze(0)
            b_transit = False
            running_b_exit_confs.append(b_xs.numpy())
        else:
            if torch.rand(1) < epsilon:
                b_indices = torch.randperm(len(b_running_short_reporters.reshape([-1, 2])[last_b_transit:]))[:n_windows]
            else:
                b_reporter_energies = V(b_running_short_reporters.reshape([-1,2])[last_b_transit:])
                b_weights = torch.sigmoid(-net(b_running_short_reporters.squeeze().reshape([-1, 2])))[ last_b_transit:].detach()
                b_weights = torch.where(dist(b_running_short_reporters.squeeze().reshape([-1, 2])[ last_b_transit:], b_center) < cutoff, 0, b_weights)
                b_weights = b_weights*torch.exp(-sampling_beta*b_reporter_energies)
                b_indices = torch.multinomial(b_weights, n_windows)
                _, b_indices = torch.sort(b_weights, descending = True)
                b_indices = b_indices[:n_windows]
                
            b_xs = b_running_short_reporters.squeeze().reshape([-1, 2])[last_b_transit:][b_indices]
    
        a_reporters, a_long_times = sampling.take_reporter_steps(a_xs, V, beta, gamma, step_size, 1000, 3, a_center, b_center, cutoff, adaptive = True)
        b_reporters, b_long_times = sampling.take_reporter_steps(b_xs, V, beta, gamma, step_size, 1000, 3, a_center, b_center, cutoff, adaptive = True)
        a_short_reporters, a_short_times = sampling.take_reporter_steps(a_xs, V, beta, gamma, step_size, 1000, 5, a_center, b_center, cutoff, adaptive = True)
        b_short_reporters, b_short_times = sampling.take_reporter_steps(b_xs, V, beta, gamma, step_size, 1000, 5, a_center, b_center, cutoff, adaptive = True)
        a_times.append(np.mean(a_short_times))
        b_times.append(np.mean(b_short_times))

        # Keep a memory of sampled configurations
        if step == 0:
            a_running_xs = a_xs.detach()
            b_running_xs = b_xs.detach()
            running_xs = torch.cat((a_running_xs, b_running_xs), axis = 0)
            a_running_reporters = a_reporters
            b_running_reporters = b_reporters
            running_reporters = torch.cat((a_running_reporters, b_running_reporters), axis = 0)
            a_running_short_reporters = a_short_reporters
            b_running_short_reporters = b_short_reporters
            running_short_reporters = torch.cat((a_running_short_reporters, b_running_short_reporters), axis = 0)
        else:
            a_running_xs = torch.cat((a_running_xs.detach(), a_xs.detach()))
            b_running_xs = torch.cat((b_running_xs.detach(), b_xs.detach()))
            a_running_reporters = torch.cat((a_running_reporters.detach(), a_reporters), axis = 0)
            b_running_reporters = torch.cat((b_running_reporters.detach(), b_reporters), axis = 0)
            running_reporters = torch.cat((a_running_reporters, b_running_reporters), axis = 0)
            a_running_short_reporters = torch.cat((a_running_short_reporters.detach(), a_short_reporters), axis = 0)
            b_running_short_reporters = torch.cat((b_running_short_reporters.detach(), b_short_reporters), axis = 0)
            running_short_reporters = torch.cat((a_running_short_reporters, b_running_short_reporters), axis = 0)
            running_xs = torch.cat((a_running_xs, b_running_xs), axis = 0)
        
        for j in range(1): # Can go through multiple optimization steps per data collection step, if needed
            with torch.no_grad():
                a_targets, b_targets, a_var, b_var = sampling.calculate_committor_estimates(running_short_reporters.reshape([-1, 2]), net, a_center, b_center, cutoff, 1000)
                a_short_targets, b_short_targets, a_short_var, b_short_var = sampling.calculate_committor_estimates(running_short_reporters.reshape([-1, 2]), net, a_center, b_center, cutoff, 1000)
            indicator = torch.tensor(1)
            plt.scatter(a_targets, a_var/np.square(a_targets))
            q_spacings =np.linspace(0.001, 1, 1000)
            plt.plot(q_spacings, 1000*(1-q_spacings)/(q_spacings*999), linestyle = '--', color = 'black')
            plt.xscale('log')
            plt.savefig('biases.pdf')
            plt.close()
            np.save('q_means.npy', a_targets)
            np.save('q_covs.npy', a_var/np.square(a_targets))
            batch_size = running_xs.size()[0]
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
        
        with torch.no_grad():
            a_exit_tensor = torch.tensor(np.array(a_escape_confs)).squeeze()
            b_exit_tensor = torch.tensor(np.array(b_escape_confs)).squeeze()
            a_rate_estimates = 1/torch.mean(a_escape_times, dim = 1)*torch.mean(torch.sigmoid(net(a_exit_tensor)))
            b_rate_estimates = 1/torch.mean(b_escape_times, dim = 1)*torch.mean(1 - torch.sigmoid(net(b_exit_tensor)))

            a_rate_mean = torch.mean(a_rate_estimates)
            a_rate_std = torch.std(a_rate_estimates)
            a_means.append(a_rate_mean.cpu().detach().numpy())
            a_stds.append(a_rate_std.cpu().detach().numpy())
            b_rate_mean = torch.mean(b_rate_estimates)
            b_rate_std = torch.std(b_rate_estimates)
            b_means.append(b_rate_mean.cpu().detach().numpy())
            b_stds.append(b_rate_std.cpu().detach().numpy())

        
        # Report to the command line
        print(f"Step {step}: Rate Estimate = {a_rate_mean.item()}; Loss = {log_loss.item()}")#, Log Loss 2 = {log_loss_2.item()}")
        print(a_index_counter, b_index_counter)
        

        # Generate an updated figure every 10 steps
        
        if torch.min(torch.sqrt(torch.sum(torch.square(a_running_short_reporters.reshape([-1, 2])[last_a_transit:] - b_center), axis = -1))) < cutoff:
            print("A Transit!")
            a_transit = True
            print((len(a_running_short_reporters.reshape([-1, 2])) - last_a_transit)/1000)
            passage_times.append(int((len(a_running_short_reporters.reshape([-1, 2])) - last_a_transit)/1000))
            print(1/1000**np.mean(passage_times))
            last_a_transit = len(a_running_short_reporters.reshape([-1, 2]))
            a_index_counter += 1
            a_transit_history.append(int(len(running_xs.reshape([-1, 2]))/2))
            print(a_transit_history)
            
        if torch.min(torch.sqrt(torch.sum(torch.square(b_running_short_reporters.reshape([-1, 2])[last_b_transit:] - a_center), axis = -1))) < cutoff:
            print("B Transit!")
            b_transit = True
            last_b_transit = len(b_running_short_reporters.reshape([-1, 2]))
            b_index_counter += 1
            b_transit_history.append(int(len(running_xs.reshape([-1, 2]))/2))
        
        #if a_transit or b_transit:
        if step % 10 == 9:
            plt.tight_layout()
            fig, axs  = plt.subplot_mosaic([['a', 'b', 'c'], ['a', 'b', 'd']], width_ratios = [1., 1., 0.75])
            fig.set_size_inches(15.2, 4.8)
            fig.colorbar(axs['c'].contourf(X, Y, torch.sigmoid(net(grid_input)).cpu().detach().numpy()-true_committor, levels = np.linspace(-0.1, .1, 101), cmap = 'mycmap3', extend = 'both'), ax = axs['c'], ticks = np.linspace(-0.1, 0.1, 5))
            axs['c'].add_patch(plt.Circle((-1, 0), cutoff, linewidth = 2, color = 'black', fill = True))
            axs['c'].add_patch(plt.Circle((1, 0), cutoff, linewidth = 2, color = 'black', fill = True))
            axs['c'].text(-1.08, -0.1, "A", weight = 'bold', size = 10, color = 'white')
            axs['c'].text(0.92, -0.1, "B", weight = 'bold', size = 10, color = 'white')
            axs['c'].set_xlabel(r"$\textrm{x}$", size = 16)
            axs['c'].set_ylabel(r"$\textrm{y}$", size = 16)
            axs['c'].contour(X, Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0., 0, 1), cmap = 'mycmap5')
            axs['c'].contourf(X,Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0, 1e20, 2), cmap = 'mycmap6', zorder = 100)
            axs['c'].text(-1.95, -1.3, r"\textrm{\textbf{(g)}}", weight = 'bold', size = 12, zorder = 200)

            fig.colorbar(axs['d'].contourf(X, Y, np.log10((torch.sigmoid(net(grid_input))*(1 - torch.sigmoid(net(grid_input)))).cpu().detach().numpy())-np.log10(true_committor*(1-true_committor)), levels = np.linspace(-1, 1, 101), cmap = 'mycmap3', extend = 'both'), ax = axs['d'], ticks = np.linspace(-1, 1, 5))
            axs['d'].add_patch(plt.Circle((-1, 0), cutoff, linewidth = 2, color = 'black', fill = True))
            axs['d'].add_patch(plt.Circle((1, 0), cutoff, linewidth = 2, color = 'black', fill = True))
            axs['d'].text(-1.08, -0.1, "A", weight = 'bold', size = 10, color = 'white')
            axs['d'].text(0.92, -0.1, "B", weight = 'bold', size = 10, color = 'white')
            axs['d'].set_xlabel(r"$\textrm{x}$", size = 16)
            axs['d'].set_ylabel(r"$\textrm{y}$", size = 16)
            axs['d'].contour(X, Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0., 0, 1), cmap = 'mycmap5')
            axs['d'].contourf(X,Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0, 1e20, 2), cmap = 'mycmap6', zorder = 100)
            axs['d'].text(-1.95, -1.3, r"\textrm{\textbf{(h)}}", weight = 'bold', size = 12, zorder = 200)

            axs['a'].contourf(X,Y, V_surface.cpu().detach().numpy(), levels = np.linspace(-5, 0, 15), cmap = 'mycmap')
            fig.colorbar(axs['a'].contour(X, Y, torch.sigmoid(net(grid_input)).cpu().detach().numpy(), levels = np.linspace(0.1, 0.9, 9), cmap = 'mycmap2'), ax = axs['a'], ticks = np.linspace(0, 1, 11))
            #axs['a'].scatter(torch.reshape(running_xs, [-1, 2]).detach().numpy()[:,0], torch.reshape(running_xs, [-1, 2]).detach().numpy()[:,1], color = 'red', alpha = 1)
            #axs['a'].scatter(torch.reshape(a_running_xs, [-1, 2]).detach().numpy()[-1,0], torch.reshape(a_running_xs, [-1, 2]).detach().numpy()[-1,1], color = 'blue', alpha = 1)
            #axs['a'].scatter(torch.reshape(b_running_xs, [-1, 2]).detach().numpy()[-1,0], torch.reshape(b_running_xs, [-1, 2]).detach().numpy()[-1,1], color = 'green', alpha = 1)
            axs['a'].add_patch(plt.Circle((-1, 0), cutoff, linewidth = 2, color = 'black', fill = True))
            axs['a'].add_patch(plt.Circle((1, 0), cutoff, linewidth = 2, color = 'black', fill = True))
            axs['a'].text(-1.08, -0.1, "A", weight = 'bold', size = 20, color = 'white')
            axs['a'].text(0.92, -0.1, "B", weight = 'bold', size = 20, color = 'white')
            axs['a'].set_xlabel(r"$\textrm{x}$", size = 16)
            axs['a'].set_ylabel(r"$\textrm{y}$", size = 16)
            #axs['a'].contour(X, Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0., 0, 1), cmap = 'mycmap5')
            axs['a'].contourf(X,Y, V_surface.cpu().detach().numpy(), levels = np.linspace(0, 1e20, 2), cmap = 'mycmap6', zorder = 100)
            axs['a'].text(-1.95, -1.35, r"\textrm{\textbf{(e)}}", weight = 'bold', size = 20, zorder = 200)
                
            

            axs['b'].plot(np.arange(step+1)*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.array(a_means), c = '#1B346C')
            axs['b'].plot(np.arange(step+1)*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.array(b_means), c = '#F54B1A')
            #axs['b'].plot(np.arange(step+1)*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 500)*step_size.cpu().detach().numpy(), np.array(losses), c = 'black')
            #axs['b'].plot(np.arange(step+1)*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 100)*step_size.cpu().detach().numpy(), np.array(middle_losses), c = 'red')
            #ax2.vlines(1/(9.18e-8*2), 0, 1, color = '#F54B1A', linestyles = 'dashed')
            #ax2.plot(np.linspace(0, 1/(9.18e-8*2), 10), np.ones(10)*9.18e-8*2, '--', c = 'black')
            axs['b'].plot(np.arange(len(a_means))*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.ones_like(a_means)*9.18e-8*2, '--', c = 'black')
            axs['b'].fill_between(np.arange(len(a_means))*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.array(a_means) - np.array(a_stds), np.array(a_means) + np.array(a_stds), color = '#1B346C', alpha = 0.5)
            axs['b'].fill_between(np.arange(len(b_means))*2*((n_reporter_trajectories*n_reporter_steps).cpu().detach().numpy() + 1)*step_size.cpu().detach().numpy(), np.array(b_means) - np.array(b_stds), np.array(b_means) + np.array(b_stds), color = '#F54B1A', alpha = 0.5)
            axs['b'].set_yscale('log')
            axs['b'].set_xlabel(r"$\textrm{Sampling Time ($\tau$)}$", size = 12)
            axs['b'].set_ylabel(r"$\textrm{Rate ($\tau^{-1}$)}$", size = 12)
            axs['b'].legend([r'$\textrm{Rate Estimate A to B}$', r'$\textrm{Rate Estimate B to A}$', r'$\textrm{Analytical Rate}$'], prop={'size': 12})
            axs['b'].set_ylim(1e-8, 1e1)
            axs['b'].text(-1000, 2e-8, r"\textrm{\textbf{(f)}}", weight = 'bold', size = 20, zorder = 200)
            plt.tight_layout()
            fig.savefig("TC_log.pdf")
            plt.close()
            
            plt.plot(losses)
            plt.plot(log_losses)
            plt.yscale('log')
            plt.savefig("Losses.pdf")
            plt.close()

    print(f"Trial {trial} Done")
    net = CommittorNet(2).to(device).double()
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)
    if trial == 0:
        all_xs = running_xs.detach()
        all_reporters = running_reporters.detach()
        all_short_reporters = running_short_reporters.detach()
    else:
        all_xs = torch.cat((all_xs.detach(), running_xs.detach()))
        all_reporters = torch.cat((all_reporters.detach(), running_reporters.detach()))
        all_short_reporters = torch.cat((all_short_reporters.detach(), running_short_reporters.detach()))
    np.save("xs.npy", all_xs.numpy())
    np.save("reporters.npy", all_reporters.numpy())
    np.save("short_reporters.npy", all_short_reporters.numpy())

# Plot the final committor
plt.contourf(X,Y, V_surface, levels = np.linspace(-5, 0, 15), cmap = 'mycmap')
plt.contour(X, Y, net(grid_input).detach().numpy(), levels = np.linspace(0.1, 0.9, n_windows), cmap = 'mycmap2')
plt.savefig("./Committor_TwoChannel.pdf")
#plt.close()
print("SAVING")
torch.save(net.state_dict(), "./Two_Channel.pt")

