import torch
import numpy as np
import pymbar
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

def V(x):
    return 3*torch.exp(-x[:,0]**2 - (x[:,1] - (1/3))**2) - 3*torch.exp(-x[:,0]**2 - (x[:,1]-(5/3))**2) - 5*torch.exp(-(x[:,0]-1)**2 - x[:,1]**2) - 5*torch.exp(-(x[:,0]+1)**2 - x[:,1]**2) + 0.2*x[:,0]**4 +0.2*(x[:,1]-(1/3))**4
    
def dist(x, y):
    return torch.sqrt(torch.sum(torch.square(x-y), axis = -1))

def compute_energy(x, V, net, k, u):
    with torch.no_grad():
        return V(x) + 0.5*k*torch.square(torch.sigmoid(net(xs)) - u)

def compute_umbrella_energies(xs, V, net, k, us):
    with torch.no_grad():
        expanded_xs = xs.repeat_interleave(xs.size()[0], axis = 0)
        expanded_us = us.tile(us.size()[0])
    return torch.reshape(V(expanded_xs) + 0.5*k*torch.square(torch.sigmoid(net(expanded_xs)) - expanded_us), [us.size()[0], xs.size()[0]]).T

def sample(x, V, net, beta, k, u, step_size):
    with torch.no_grad():
        old_energy = compute_energy(x, V, net, k, u)
        random_step = torch.normal(torch.zeros(x.size()), std = 1.0).to(device)*step_size
        new_energy = compute_energy(x + random_step, V, net, k, u)
        acceptance = torch.heaviside(torch.exp(-beta*(new_energy - old_energy)) - torch.rand(x.size()[0]).to(device),torch.tensor([0.]).to(device))
        acceptance = torch.unsqueeze(acceptance, 1).expand(x.size())
    return acceptance*(x+random_step) + (1-acceptance)*(x)

def umbrella_sample(xs, V, net, beta, k, us, step_size):
    with torch.no_grad():
        new_xs = sample(xs, V, net, beta, k, us, step_size) # Generate a new sample in each window
        energies = compute_umbrella_energies(new_xs, V, net, k, us) # Compute energy of each sample across all windows
    return(new_xs, energies)

def multiple_umbrella_samples(xs, V, net, beta, k, us, step_size, n_batch_steps = 50, n_steps_per_batch = 10):
    with torch.no_grad():
        batch_positions = torch.zeros([n_batch_steps, xs.size()[0], xs.size()[1]]) # Initialize empty windows for positions and energies
        bias_energies = torch.zeros([xs.size()[0], n_batch_steps*xs.size()[0]])
        for n in range(n_batch_steps): # Umbrella sample n times, storing results
            for _ in range(n_steps_per_batch):
                xs, energies = umbrella_sample(xs, V, net, beta, k, us, step_size)
            while torch.any(dist(xs, a_center) < cutoff) or torch.any(dist(xs, b_center) < cutoff): # Naive way to prevent sampling in basins
                xs, energies = umbrella_sample(xs, V, net, beta, k, us, step_size)
            #print(torch.where(dist(xs, a_center) < cutoff))
            #print(torch.where(dist(xs, b_center) < cutoff))
            batch_positions[n] = xs
            bias_energies[:,n:n+n_batch_steps*xs.size()[0]:n_batch_steps] = energies
    return xs, batch_positions, bias_energies
    
def umbrella_sample_Langevin(xs, V, net, beta, gamma, k, us, step_size, single_sample = False, i = None):
    #with torch.no_grad():
    new_xs = Langevin_step_net(xs, V, net, beta, gamma, k, us, step_size, single_sample, i) # Generate a new sample in each window
    if not single_sample:
        energies = compute_umbrella_energies(new_xs, V, net, k, us) # Compute energy of each sample across all windows
        return(new_xs, energies)
    else:
        return(new_xs, None)
        
def umbrella_sample_doob(xs, V, net, beta, gamma, k, us, step_size, B, single_sample = False, i = None):
    #with torch.no_grad():
    new_xs = Langevin_step_doob(xs, V, net, beta, gamma, k, us, step_size, B, single_sample, i) # Generate a new sample in each window
    #print("!")
    #print(new_xs)
    #print("!")
    if not single_sample:
        energies = compute_umbrella_energies(new_xs, V, net, k, us) # Compute energy of each sample across all windows
        return(new_xs, energies)
    else:
        return(new_xs, None)
        
def umbrella_sample_underdamped(xs, vs, V, net, beta, gamma, k, us, step_size, single_sample = False, i = None):
    #with torch.no_grad():
    new_xs, new_vs = Langevin_step_net(xs, vs, V, net, beta, gamma, k, us, step_size, single_sample, i) # Generate a new sample in each window
    if not single_sample:
        energies = compute_umbrella_energies(new_xs, V, net, k, us) # Compute energy of each sample across all windows
        return(new_xs, new_vs, energies)
    else:
        return(new_xs, new_vs, None)

def multiple_umbrella_samples_Langevin(xs, V, net, beta, gamma, k, us, step_size, a_center, b_center, cutoff, n_batch_steps = 50, n_steps_per_batch = 10):
    #with torch.no_grad():
    batch_positions = torch.zeros([n_batch_steps, xs.size()[0], xs.size()[1]]) # Initialize empty windows for positions and energies
    bias_energies = torch.zeros([xs.size()[0], n_batch_steps*xs.size()[0]])
    for n in range(n_batch_steps): # Umbrella sample n times, storing results
        for _ in range(n_steps_per_batch):
            xs, energies = umbrella_sample_Langevin(xs, V, net, beta, gamma, k, us, step_size)
        for i in range(len(xs)):
            while dist(xs[i], a_center) < cutoff or dist(xs[i], b_center) < cutoff:
                xs[i] = umbrella_sample_Langevin(xs[i].unsqueeze(0), V, net, beta, gamma, k, us, step_size, True, i)[0].squeeze()
        batch_positions[n] = xs
        bias_energies[:,n:n+n_batch_steps*xs.size()[0]:n_batch_steps] = energies
    return xs, batch_positions, bias_energies
    
def multiple_umbrella_samples_doob(xs, V, net, beta, gamma, k, us, step_size, a_center, b_center, cutoff, n_batch_steps = 50, n_steps_per_batch = 10, B = False):
    #with torch.no_grad():
    batch_positions = torch.zeros([n_batch_steps, xs.size()[0], xs.size()[1]]) # Initialize empty windows for positions and energies
    bias_energies = torch.zeros([xs.size()[0], n_batch_steps*xs.size()[0]])
    for n in range(n_batch_steps): # Umbrella sample n times, storing results
        #print(n)
        for _ in range(n_steps_per_batch):
            xs, energies = umbrella_sample_doob(xs, V, net, beta, gamma, k, us, step_size, B)
        for i in range(len(xs)):
            while dist(xs[i], a_center) < cutoff or dist(xs[i], b_center) < cutoff:
                xs[i] = umbrella_sample_doob(xs[i].unsqueeze(0), V, net, beta, gamma, k, us, step_size, B, True, i)[0].squeeze()
        batch_positions[n] = xs
        bias_energies[:,n:n+n_batch_steps*xs.size()[0]:n_batch_steps] = energies
    return xs, batch_positions, bias_energies
    
def multiple_umbrella_samples_underdamped(xs, vs, V, net, beta, gamma, k, us, step_size, a_center, b_center, cutoff, n_batch_steps = 50, n_steps_per_batch = 10):
    #with torch.no_grad():
    batch_positions = torch.zeros([n_batch_steps, xs.size()[0], xs.size()[1]]) # Initialize empty windows for positions and energies
    batch_velocities = torch.zeros([n_batch_steps, vs.size()[0], vs.size()[1]]) # Initialize empty windows for positions and energies
    bias_energies = torch.zeros([xs.size()[0], n_batch_steps*xs.size()[0]])
    for n in range(n_batch_steps): # Umbrella sample n times, storing results
        for _ in range(n_steps_per_batch):
            xs, vs, energies = umbrella_sample_underdamped(xs, vs, V, net, beta, gamma, k, us, step_size)
        for i in range(len(xs)):
            while dist(xs[i], a_center) < cutoff or dist(xs[i], b_center) < cutoff:
                xs[i], vs[i] = umbrella_sample_underdamped(xs[i].unsqueeze(0), V, net, beta, gamma, k, us, step_size, True, i)[0].squeeze()
        batch_positions[n] = xs
        batch_velocities[n] = vs
        bias_energies[:,n:n+n_batch_steps*xs.size()[0]:n_batch_steps] = energies
    return xs, vs, batch_positions, batch_velocities, bias_energies
    
def Langevin_step(x, V, beta, gamma, step_size):
    x = torch.clone(x).detach()
    x.requires_grad = True
    gradient = torch.autograd.grad(outputs=V(x),
                                    inputs=x, grad_outputs = torch.ones_like(V(x)),
                                    create_graph = True)[0]
    step = -(1/gamma)*gradient*step_size.unsqueeze(-1) + torch.sqrt(2/(beta*gamma))*torch.normal(torch.zeros(x.size())).to(device)*torch.sqrt(step_size.unsqueeze(-1))
    return x + step
    
def Underdamped_step(x, v, V, beta, gamma, step_size):
    x = torch.clone(x).detach()
    x.requires_grad = True
    gradient = torch.autograd.grad(outputs=V(x),
                                    inputs=x, grad_outputs = torch.ones_like(V(x)),
                                    create_graph = True)[0]
    v += -(1/gamma)*gradient*step_size.unsqueeze(-1) - gamma*v*step_size.unsqueeze(-1) + torch.sqrt(2/(beta*gamma))*torch.normal(torch.zeros(x.size())).to(device)*torch.sqrt(step_size.unsqueeze(-1))
    return x + v*step_size.unsqueeze(-1), v
    
def Langevin_step_net(x, V, net, beta, gamma, k, us, step_size, single_sample = False, i = None):
    x = torch.clone(x).detach()
    x.requires_grad = True
    if single_sample:
        output = V(x) + 0.5*k*torch.square(torch.sigmoid(net(x)) - us[i])
    else:
        output = V(x) + 0.5*k*torch.square(torch.sigmoid(net(x)) - us)
    gradient = torch.autograd.grad(outputs=output,
                                    inputs=x, grad_outputs = torch.ones_like(V(x)),
                                    create_graph = True)[0]
    step = -(1/gamma)*gradient*step_size.unsqueeze(-1) + torch.sqrt(2/(beta*gamma))*torch.normal(torch.zeros(x.size())).to(device)*torch.sqrt(step_size.unsqueeze(-1))
    return x + step
    
def Langevin_step_doob(x, V, net, beta, gamma, k, us, step_size, B, single_sample = False, i = None):
    x = torch.clone(x).detach()
    x.requires_grad = True
    potential_gradient = torch.autograd.grad(outputs=V(x),
                                    inputs=x, grad_outputs = torch.ones_like(V(x)),
                                    create_graph = True)[0]
                                    
    if B:
        doob_gradient = torch.autograd.grad(outputs=torch.log(torch.sigmoid(-net(x))).squeeze(),
                                    inputs=x, grad_outputs = torch.ones_like(net(x)),
                                    create_graph = True)[0]
        #doob_force = 2/(beta)*doob_gradient/(torch.sigmoid(-net(x)).unsqueeze(-1))
        #print('B')
        #print(doob_force)
    else:

        doob_gradient = torch.autograd.grad(outputs=torch.log(torch.sigmoid(net(x))).squeeze(),
                                    inputs=x, grad_outputs = torch.ones_like(net(x)),
                                    create_graph = True)[0]
        #print('A')
        #print(doob_force)
        

    doob_mags = torch.sqrt(torch.sum(torch.square(doob_gradient), axis = -1))
    #print(doob_force)
    #print(doob_mags)
    #print(doob_force / doob_mags.unsqueeze(-1))
    #indices_too_large = torch.where(doob_mags > 10)[0]
    #doob_force[indices_too_large] = 10*doob_force[indices_too_large]/(doob_mags[indices_too_large].unsqueeze(-1))
    step = -(1/gamma)*(potential_gradient - 2*0.1/beta * doob_gradient)*step_size.unsqueeze(-1) + torch.sqrt(2/(beta*gamma))*torch.normal(torch.zeros(x.size())).to(device)*torch.sqrt(step_size.unsqueeze(-1))
    return x + step
    
def Underdamped_step_net(x, v, V, net, beta, gamma, k, us, step_size, single_sample = False, i = None):
    x = torch.clone(x).detach()
    x.requires_grad = True
    if single_sample:
        output = V(x) + 0.5*k*torch.square(torch.sigmoid(net(x,v)) - us[i])
    else:
        output = V(x) + 0.5*k*torch.square(torch.sigmoid(net(x,v)) - us)
    gradient = torch.autograd.grad(outputs=output,
                                    inputs=x, grad_outputs = torch.ones_like(V(x)),
                                    create_graph = True)[0]
    v += -(1/gamma)*gradient*step_size.unsqueeze(-1) - gamma*v*step_size.unsqueeze(-1) + torch.sqrt(2/(beta*gamma))*torch.normal(torch.zeros(x.size())).to(device)*torch.sqrt(step_size.unsqueeze(-1))
    return x+v*step_size.unsqueeze(-1), v
    
def take_reporter_steps_stride(input_xs, V, beta, gamma, step_size, n_trajectories, n_steps, a_center, b_center, cutoff, adaptive = False):
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    #print(xs.size())
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    steps = torch.clone(step_sizes)
    #print(step_sizes.size())
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
    for q in range(n_steps):
        #print(q)
        xs = Langevin_step(xs, V, beta, gamma, step_sizes)
        steps += step_sizes
        #step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        #step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
        # Truncate reporter length once one reporter reaches a basin:
        if adaptive:
            for i in range(len(input_xs)):
            #print(i)
            #print(step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0])
                if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < 0*n_trajectories:
                #print(i)
                #print(step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero())
                #print(step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size())
                #print("ZERO")
                    step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
                #print(step_sizes)
                #print(i)
        #print(step_sizes.nonzero())
        
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            #print('WOOO')
            break
    #print(steps)
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), (steps/step_size).detach().numpy()

def take_reporter_steps(input_xs, V, beta, gamma, step_size, n_trajectories, n_steps, a_center, b_center, cutoff, adaptive = False):
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    #print(xs.size())
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    steps = torch.clone(step_sizes)
    #print(step_sizes.size())
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
    for q in range(n_steps):
        #print(q)
        xs = Langevin_step(xs, V, beta, gamma, step_sizes)
        steps += step_sizes
        step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
        # Truncate reporter length once one reporter reaches a basin:
        if adaptive:
            for i in range(len(input_xs)):
            #print(i)
            #print(step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0])
                if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < n_trajectories:
                #print(i)
                #print(step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero())
                #print(step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size())
                #print("ZERO")
                    step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
                #print(step_sizes)
                #print(i)
        #print(step_sizes.nonzero())
        
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            #print('WOOO')
            break
    #print(steps)
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), (steps/step_size).detach().numpy()
    
def take_reporter_steps_underdamped(input_xs, input_vs, V, beta, gamma, step_size, n_trajectories, n_steps, a_center, b_center, cutoff, lag, adaptive = False):
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    vs = torch.repeat_interleave(input_vs, n_trajectories, axis = 0).to(device)
    print(xs)
    #print(xs.size())
    print(vs)
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    steps = torch.clone(step_sizes)
    #print(step_sizes.size())
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
    for q in range(n_steps):
        print(q)
        for substep in range(lag):
            xs, vs = Underdamped_step(xs, vs, V, beta, gamma, step_sizes)
        steps += step_sizes
        step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
        # Truncate reporter length once one reporter reaches a basin:
        if adaptive:
            for i in range(len(input_xs)):
                #print(i)
                #print(step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0])
                if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < n_trajectories:
                    #print(i)
                    #print(step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero())
                    #print(step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size())
                    #print("ZERO")
                    step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
                    #print(step_sizes)
                    #print(i)
            #print(step_sizes.nonzero())
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            print('WOOO')
            break
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), vs.reshape([-1, n_trajectories, xs.size()[-1]]), (steps/step_size).detach().numpy()
    
def take_reporter_steps_underdamped_2(input_xs, V, beta, gamma, step_size, n_trajectories, n_steps, a_center, b_center, cutoff, lag, adaptive = False):
    #print(input_vs)
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    vs = torch.randn(xs.size())*torch.sqrt(1/beta)
    #print(xs.size())
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    steps = torch.clone(step_sizes)
    #print(step_sizes.size())
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
    for q in range(n_steps):
        print(q)
        for substep in range(lag):
            xs, vs = Underdamped_step(xs, vs, V, beta, gamma, step_sizes)
        steps += step_sizes*lag
        step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
        # Truncate reporter length once one reporter reaches a basin:
        for i in range(len(input_xs)):
            #print(i)
            #print(step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0])
            if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < n_trajectories:
                #print(i)
                #print(step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero())
                #print(step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size())
                #print("ZERO")
                step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
                #print(step_sizes)
                #print(i)
        #print(step_sizes.nonzero())
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            #print('WOOO')
            break
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), (steps/step_size).detach().numpy()
            
def take_reporter_steps_MFPT(input_xs, V, beta, gamma, step_size, n_trajectories, n_steps, a_center, b_center, cutoff):
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    traj_lengths = torch.zeros([len(step_sizes), n_steps])
    zeros = torch.zeros(xs.size()[0]).to(device)
    #step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes)# Set step size to 0 if in basin
    for c in range(n_steps):
        xs = Langevin_step(xs, V, beta, gamma, step_sizes)
        traj_lengths[:,c] = step_sizes
        step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes)
        #step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        # Set step size to 0 if in basin
        # Truncate reporter length once one reporter reaches a basin:
        #for i in range(len(input_xs)):
        #    if 0 in step_sizes[i*n_trajectories:(i+1)*n_trajectories]:
        #        step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
                #print(i)
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            break
    
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), torch.sum(traj_lengths, dim = -1)
    
def take_reporter_steps_2(xs, V, beta, gamma, step_size, n_trajectories, n_max_steps, a_center, b_center, cutoff):
    xs = torch.repeat_interleave(xs, n_trajectories, axis = 0).to(device)
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
    for step in range(n_max_steps):
        xs = Langevin_step(xs, V, beta, gamma, step_sizes)
        step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            break
    
    return xs.reshape([-1, n_trajectories, xs.size()[-1]])
    
def take_reporter_steps_3(xs, V, beta, gamma, step_size, n_trajectories, n_steps, a_center, b_center, cutoff): # Ensures no all-basin reporters
    xs = torch.repeat_interleave(xs, n_trajectories, axis = 0).to(device)
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
    if torch.any(step_sizes == 0):
        doop1
    for _ in range(n_steps):
        old_xs = xs.clone()
        xs = Langevin_step(xs, V, beta, gamma, step_sizes)
        step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
        indices = torch.where(step_sizes == 0.)
        xs[indices] = old_xs[indices]
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            break
        if torch.any(dist(xs, a_center) < cutoff):
            doop2
        if torch.any(dist(xs, b_center) < cutoff):
            doop3
    return xs.reshape([-1, n_trajectories, xs.size()[-1]])

def calculate_committor_estimates(xs, net, a_center, b_center, cutoff, n_trajectories):
    zeros = torch.zeros(xs.size()[0]).to(device)
    ones = torch.ones(xs.size()[0]).to(device)
    a_estimates = torch.sigmoid(net(xs))
    a_estimates_for_var = torch.reshape(a_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_a_var = torch.var(a_estimates_for_var, axis = 1)
    a_estimates = torch.where(dist(xs, a_center) < cutoff, zeros, a_estimates)
    a_estimates = torch.where(dist(xs, b_center) < cutoff, ones, a_estimates)
    a_estimates = torch.reshape(a_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_a_estimates = torch.mean(a_estimates, axis = 1)
    final_a_var = torch.var(a_estimates, axis = 1)
    
    b_estimates = torch.sigmoid(-net(xs))
    b_estimates_for_var = torch.reshape(b_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_b_var = torch.var(b_estimates_for_var, axis = 1)
    b_estimates = torch.where(dist(xs, a_center) < cutoff, ones, b_estimates)
    b_estimates = torch.where(dist(xs, b_center) < cutoff, zeros, b_estimates)
    b_estimates = torch.reshape(b_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_b_estimates = torch.mean(b_estimates, axis = 1)
    final_b_var = torch.var(b_estimates, axis = 1)
    return final_a_estimates, final_b_estimates, final_a_var.detach(), final_b_var.detach()
    
def calculate_committor_estimates_underdamped(xs, vs, net, a_center, b_center, cutoff, n_trajectories):
    zeros = torch.zeros(xs.size()[0]).to(device)
    ones = torch.ones(xs.size()[0]).to(device)
    a_estimates = torch.sigmoid(net(xs, vs))
    a_estimates_for_var = torch.reshape(a_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_a_var = torch.var(a_estimates_for_var, axis = 1)
    a_estimates = torch.where(dist(xs, a_center) < cutoff, zeros, a_estimates)
    a_estimates = torch.where(dist(xs, b_center) < cutoff, ones, a_estimates)
    a_estimates = torch.reshape(a_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_a_estimates = torch.mean(a_estimates, axis = 1)
    final_a_var = torch.var(a_estimates, axis = 1)
    
    b_estimates = torch.sigmoid(-net(xs, vs))
    b_estimates_for_var = torch.reshape(b_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_b_var = torch.var(b_estimates_for_var, axis = 1)
    b_estimates = torch.where(dist(xs, a_center) < cutoff, ones, b_estimates)
    b_estimates = torch.where(dist(xs, b_center) < cutoff, zeros, b_estimates)
    b_estimates = torch.reshape(b_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_b_estimates = torch.mean(b_estimates, axis = 1)
    final_b_var = torch.var(b_estimates, axis = 1)
    return final_a_estimates, final_b_estimates, final_a_var.detach(), final_b_var.detach()
    
def calculate_MFPT_estimates(xs, net, a_center, b_center, cutoff, lengths, n_trajectories, constant):
    #const = (torch.ones(xs.size()[0])*constant).to(device) + lengths
    zeros = torch.zeros(xs.size()[0]).to(device) + lengths
    a_estimates = (net(xs)) + lengths
    #a_estimates = torch.where(dist(xs, a_center) < cutoff, constant, a_estimates)
    a_estimates = torch.where(dist(xs, b_center) < cutoff, zeros, a_estimates)
    a_estimates = torch.reshape(a_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_a_estimates = torch.mean(a_estimates, axis = 1)
    #print(torch.max(final_a_estimates))
    return final_a_estimates
    
def calculate_committor_estimates_2(xs, net, a_center, b_center, cutoff, n_trajectories):
    zeros = torch.zeros(xs.size()[0]).to(device)
    ones = torch.ones(xs.size()[0]).to(device)
    a_estimates = torch.sigmoid(net(xs))
    a_estimates = torch.where(dist(xs, a_center) < cutoff, zeros, a_estimates)
    a_estimates = torch.where(dist(xs, b_center) < cutoff, ones, a_estimates)
    a_estimates = torch.reshape(a_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    a_fractions = torch.where(a_estimates == 0, 1.0, 0.0)
    a_fractions = torch.mean(a_fractions, axis = 1)
    print(a_fractions)
    print(a_fractions.size())
    plt.hist(a_fractions)
    plt.savefig('boundary_fractions.pdf')
    plt.close()
    final_a_estimates = torch.mean(a_estimates, axis = 1)
    #print(len(torch.where(final_estimates == 1.)[0]))
    
    b_estimates = torch.sigmoid(-net(xs))
    b_estimates = torch.where(dist(xs, a_center) < cutoff, ones, b_estimates)
    b_estimates = torch.where(dist(xs, b_center) < cutoff, zeros, b_estimates)
    b_estimates = torch.reshape(b_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_b_estimates = torch.mean(b_estimates, axis = 1)
    return final_a_estimates, final_b_estimates
    
def calculate_committor_estimates_iter(xs, net, iter_net, a_center, b_center, cutoff, n_trajectories):
    zeros = torch.zeros(xs.size()[0]).to(device)
    ones = torch.ones(xs.size()[0]).to(device)
    a_estimates_old = torch.sigmoid(net(xs))
    a_estimates_new = torch.sigmoid(iter_net(xs))
    a_estimates = torch.where(dist(xs, a_center) < cutoff, zeros, a_estimates_new)
    a_estimates = torch.where(a_estimates_old > 0.9, ones, a_estimates_new)
    a_estimates = torch.reshape(a_estimates_new, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_a_estimates = torch.mean(a_estimates, axis = 1)
    return final_a_estimates

def flux_sample(net, x, V, beta, gamma, step_size, a_center, b_center, iso_value, basin_cutoff, n_crossings):
    x = a_center.unsqueeze(0)
    n_steps = 0
    crossings = 0
    in_basin = True
    from_A = True
    escape_confs = []
    escape_times = []
    last_crossing = 0
    just_left_flag = False
    while crossings < n_crossings + 1:
        just_left_flag = False
        x = Langevin_step(x, V, beta, gamma, step_size)
        n_steps += 1
        #print(prediction)
        if torch.sqrt(torch.sum(torch.square(x - a_center))) > basin_cutoff and in_basin:
            just_left_flag = True
            #print("Leaving Basin")
            in_basin = False
            crossings += 1
            print(crossings)
            escape_confs.append(x.squeeze())
            escape_times.append(n_steps)
            from_A = False
        if torch.sqrt(torch.sum(torch.square(x - b_center))) < basin_cutoff:
           print("Actual Transition")
           x = a_center.unsqueeze(0)
           escape_times.append(0)
           escape_confs.append(x.squeeze())
        if torch.sqrt(torch.sum(torch.square(x - a_center))) < basin_cutoff and from_A == False:
            #print("Re-entering Basin")
            from_A = True
            in_basin = True
    return torch.tensor(escape_times)*step_size, torch.stack(escape_confs)
