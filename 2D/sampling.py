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
    
def Langevin_step(x, V, beta, gamma, step_size):
    x = torch.clone(x).detach()
    x.requires_grad = True
    gradient = torch.autograd.grad(outputs=V(x),
                                    inputs=x, grad_outputs = torch.ones_like(V(x)),
                                    create_graph = True)[0]
    step = -(1/gamma)*gradient*step_size.unsqueeze(-1) + torch.sqrt(2/(beta*gamma))*torch.normal(torch.zeros(x.size())).to(device)*torch.sqrt(step_size.unsqueeze(-1))
    return x + step
    
def Langevin_step_constantF(x, V, beta, gamma, F, step_size):
    x = torch.clone(x).detach()
    x.requires_grad = True
    gradient = torch.autograd.grad(outputs=V(x),
                                    inputs=x, grad_outputs = torch.ones_like(V(x)),
                                    create_graph = True)[0]
    step = -(1/gamma)*gradient*step_size.unsqueeze(-1) + torch.sqrt(2/(beta*gamma))*torch.normal(torch.zeros(x.size())).to(device)*torch.sqrt(step_size.unsqueeze(-1)) + F*step_size.unsqueeze(-1)
    # Include Path Weight
    weight = torch.exp(-beta*(2*torch.matmul(step,F)/step_size.unsqueeze(-1) + 2*torch.matmul(gradient, F) - torch.dot(F,F))/4 * step_size.unsqueeze(-1))
    return x + step, weight[0]
    
def Langevin_step_F(x, V, beta, gamma, F, step_size):
    x = torch.clone(x).detach()
    x.requires_grad = True
    gradient = torch.autograd.grad(outputs=V(x),
                                    inputs=x, grad_outputs = torch.ones_like(V(x)),
                                    create_graph = True)[0]
    force = -torch.autograd.grad(outputs=F(x),
                                    inputs=x, grad_outputs = torch.ones_like(F(x)),
                                    create_graph = True)[0]
    step = -(1/gamma)*gradient*step_size.unsqueeze(-1) + torch.sqrt(2/(beta*gamma))*torch.normal(torch.zeros(x.size())).to(device)*torch.sqrt(step_size.unsqueeze(-1)) + force*step_size.unsqueeze(-1)
    
    # Include Path Weight
    weight = torch.exp(-beta*(2*torch.sum(step * force, dim = -1)/step_size.unsqueeze(-1) + 2*torch.sum(gradient*force, dim = -1) - torch.sum(force*force, dim = -1))/4 * step_size.unsqueeze(-1))
    return x + step, weight[0]
    
def Underdamped_step(x, v, V, beta, gamma, step_size):
    x = torch.clone(x).detach()
    x.requires_grad = True
    gradient = torch.autograd.grad(outputs=V(x),
                                    inputs=x, grad_outputs = torch.ones_like(V(x)),
                                    create_graph = True)[0]
    v += -(1/gamma)*gradient*step_size.unsqueeze(-1) - gamma*v*step_size.unsqueeze(-1) + torch.sqrt(2/(beta*gamma))*torch.normal(torch.zeros(x.size())).to(device)*torch.sqrt(step_size.unsqueeze(-1))
    return x + v*step_size.unsqueeze(-1), v
    
def Langevin_step_net(x, V, beta, gamma, net, step_size, b = False):
    x = torch.clone(x).detach()
    #print(x)
    x.requires_grad = True
    gradient = torch.autograd.grad(outputs=V(x),
                                    inputs=x, grad_outputs = torch.ones_like(V(x)),
                                    create_graph = True)[0]
    if not b:
        force = 2/(beta*gamma)*torch.autograd.grad(outputs=torch.log(torch.sigmoid(net(x))),
                                    inputs=x, grad_outputs = torch.ones_like(net(x)),
                                    create_graph = True)[0]
    else:
        force = 2/(beta*gamma)*torch.autograd.grad(outputs=torch.log(torch.sigmoid(-net(x))),
                                    inputs=x, grad_outputs = torch.ones_like(net(x)),
                                    create_graph = True)[0]

    expected_step = (-gradient + force)*step_size.unsqueeze(-1)
    step = -(1/gamma)*gradient*step_size.unsqueeze(-1) + torch.sqrt(2/(beta*gamma))*torch.normal(torch.zeros(x.size())).to(device)*torch.sqrt(step_size.unsqueeze(-1)) + force*step_size.unsqueeze(-1)
    
    # Include Path Weight
    doob_weight = torch.sum(step*expected_step, dim = -1)
    weight = torch.exp(-beta*(2*torch.sum(step * force, dim = -1)/step_size.unsqueeze(-1) + 2*torch.sum(gradient*force, dim = -1) - torch.sum(force*force, dim = -1))/4 * step_size.unsqueeze(-1))
    return x + step, weight[0], doob_weight
    
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
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    steps = torch.clone(step_sizes)
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
    for q in range(n_steps):
        xs = Langevin_step(xs, V, beta, gamma, step_sizes)
        steps += step_sizes
        if adaptive:
            for i in range(len(input_xs)):
                if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < 0*n_trajectories:
                    step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
        
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            break
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), (steps/step_size).detach().numpy()

def take_reporter_steps(input_xs, V, beta, gamma, step_size, n_trajectories, n_steps, a_center, b_center, cutoff, adaptive = False):
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    steps = torch.clone(step_sizes)
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
    for q in range(n_steps):
        xs = Langevin_step(xs, V, beta, gamma, step_sizes)
        steps += step_sizes
        step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
        # Truncate reporter length once one reporter reaches a basin:
        if adaptive:
            for i in range(len(input_xs)):
                if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < n_trajectories:
                    step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
        
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            break
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), (steps/step_size).detach().numpy()
    
def take_reporter_steps_constantF(input_xs, V, beta, gamma, step_size, F, n_trajectories, n_steps, a_center, b_center, cutoff, adaptive = False):
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    running_weights = torch.ones(len(xs))
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    steps = torch.clone(step_sizes)
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
    for q in range(n_steps):
        xs, weights = Langevin_step_constantF(xs, V, beta, gamma, F, step_sizes)
        running_weights = running_weights * weights
        steps += step_sizes
        step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
        # Truncate reporter length once one reporter reaches a basin:
        if adaptive:
            for i in range(len(input_xs)):
                if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < n_trajectories:
                    step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
        
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            break
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), (steps/step_size).detach().numpy(), running_weights/torch.sum(running_weights)#torch.ones(n_trajectories)/n_trajectories

def take_reporter_steps_F(input_xs, V, beta, gamma, step_size, F, n_trajectories, n_steps, a_center, b_center, cutoff, adaptive = False):
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    running_weights = torch.ones(len(xs))
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    steps = torch.clone(step_sizes)
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
    for q in range(n_steps):
        xs, weights = Langevin_step_F(xs, V, beta, gamma, F, step_sizes)
        running_weights = running_weights * weights
        steps += step_sizes
        step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
        if adaptive:
            for i in range(len(input_xs)):
                if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < n_trajectories:
                    step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
        
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            break
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), (steps/step_size).detach().numpy(), running_weights/torch.sum(running_weights)

def take_reporter_steps_net(input_xs, V, beta, gamma, net, step_size, n_trajectories, n_steps, a_center, b_center, cutoff, adaptive = False, b = False):
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    running_weights = torch.ones(len(xs))
    running_doob_weights = torch.ones(len(xs))
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    steps = torch.clone(step_sizes)
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
    for q in range(n_steps):
        xs, weights, doob_weights = Langevin_step_net(xs, V, beta, gamma, net, step_sizes, b)
        running_weights = running_weights * weights
        running_doob_weights = running_doob_weights * doob_weights
        steps += step_sizes
        step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes) # Set step size to 0 if in basin
        # Truncate reporter length once one reporter reaches a basin:
        if adaptive:
            for i in range(len(input_xs)):
                if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < n_trajectories:
                    step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
        
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            break
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), (steps/step_size).detach().numpy(), running_weights/torch.sum(running_weights), torch.mean(torch.log(running_weights))

def take_reporter_steps_underdamped(input_xs, input_vs, V, beta, gamma, step_size, n_trajectories, n_steps, a_center, b_center, cutoff, lag, adaptive = False):
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    vs = torch.repeat_interleave(input_vs, n_trajectories, axis = 0).to(device)
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    steps = torch.clone(step_sizes)
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
                if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < n_trajectories:
                    step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            print('WOOO')
            break
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), vs.reshape([-1, n_trajectories, xs.size()[-1]]), (steps/step_size).detach().numpy()
            
def take_reporter_steps_MFPT(input_xs, V, beta, gamma, step_size, n_trajectories, n_steps, a_center, b_center, cutoff, adaptive = False):
    xs = torch.repeat_interleave(input_xs, n_trajectories, axis = 0).to(device)
    step_sizes = torch.repeat_interleave(step_size, xs.size()[0], axis = 0)
    traj_lengths = torch.zeros([len(step_sizes), n_steps])
    zeros = torch.zeros(xs.size()[0]).to(device)
    step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
    step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes)# Set step size to 0 if in basin
    for c in range(n_steps):
        xs = Langevin_step(xs, V, beta, gamma, step_sizes)
        traj_lengths[:,c] = step_sizes
        step_sizes = torch.where(dist(xs, b_center) < cutoff, zeros, step_sizes)
        step_sizes = torch.where(dist(xs, a_center) < cutoff, zeros, step_sizes)
        for i in range(len(input_xs)):
            if step_sizes[i*n_trajectories:(i+1)*n_trajectories].nonzero().size()[0] < n_trajectories:
                step_sizes[i*n_trajectories:(i+1)*n_trajectories] = zeros[i*n_trajectories:(i+1)*n_trajectories]
        if step_sizes.nonzero().size() == torch.Size([0,1]):
            break
    
    return xs.reshape([-1, n_trajectories, xs.size()[-1]]), torch.sum(traj_lengths, dim = -1)

def calculate_committor_estimates(xs, net, a_center, b_center, cutoff, n_trajectories):
    zeros = torch.zeros(xs.size()[0]).to(device)
    ones = torch.ones(xs.size()[0]).to(device)
    a_estimates = torch.sigmoid(net(xs))
    xs_for_var = torch.reshape(xs, [int(xs.size()[0]/n_trajectories), n_trajectories, 2])
    a_estimates = torch.where(dist(xs, a_center) < cutoff, zeros, a_estimates)
    a_estimates = torch.where(dist(xs, b_center) < cutoff, ones, a_estimates)
    a_estimates = torch.reshape(a_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_a_estimates = torch.mean(a_estimates, axis = 1)
    final_a_var = (torch.sum(torch.var(xs_for_var, axis = 1), axis = -1))
    final_a_means = torch.mean(xs_for_var, axis = 1)
    
    b_estimates = torch.sigmoid(-net(xs))
    b_estimates_for_var = torch.reshape(b_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    b_estimates = torch.where(dist(xs, a_center) < cutoff, ones, b_estimates)
    b_estimates = torch.where(dist(xs, b_center) < cutoff, zeros, b_estimates)
    b_estimates = torch.reshape(b_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_b_estimates = torch.mean(b_estimates, axis = 1)
    final_b_var = torch.sum(torch.var(xs_for_var, axis = 1))
    return final_a_estimates, final_b_estimates, final_a_var.detach(), final_a_means.detach()
    
def calculate_reweighted_committor_estimates(xs, net, a_center, b_center, cutoff, n_trajectories, weights):
    zeros = torch.zeros(xs.size()[0]).to(device)
    ones = torch.ones(xs.size()[0]).to(device)
    a_estimates = torch.sigmoid(net(xs))
    xs_for_var = torch.reshape(xs, [int(xs.size()[0]/n_trajectories), n_trajectories, 2])
    a_estimates = torch.where(dist(xs, a_center) < cutoff, zeros, a_estimates)
    a_estimates = torch.where(dist(xs, b_center) < cutoff, ones, a_estimates)
    a_estimates = torch.reshape(a_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    weights = torch.reshape(weights, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_a_estimates = torch.sum(a_estimates*weights, axis = 1)
    final_a_var = (torch.sum(torch.var(xs_for_var, axis = 1), axis = -1))
    final_a_means = torch.mean(xs_for_var, axis = 1)
    
    b_estimates = torch.sigmoid(-net(xs))
    b_estimates_for_var = torch.reshape(b_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    b_estimates = torch.where(dist(xs, a_center) < cutoff, ones, b_estimates)
    b_estimates = torch.where(dist(xs, b_center) < cutoff, zeros, b_estimates)
    b_estimates = torch.reshape(b_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    weights = torch.reshape(weights, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_b_estimates = torch.sum(b_estimates*weights, axis = 1)
    final_b_var = torch.sum(torch.var(xs_for_var, axis = 1))
    return final_a_estimates, final_b_estimates, final_a_var.detach(), final_a_means.detach()
    
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
    
def calculate_MFPT_estimates(xs, times, net, a_center, b_center, cutoff, n_trajectories):
    ones = (torch.ones(xs.size()[0])).to(device)
    zeros = torch.zeros(xs.size()[0]).to(device)
    a_estimates = (net(xs))
    mask = torch.where(dist(xs, a_center) < cutoff, zeros, ones)
    a_estimates = torch.where(dist(xs, b_center) < cutoff, zeros, a_estimates) + times
    a_estimates = torch.reshape(a_estimates, [int(xs.size()[0]/n_trajectories), n_trajectories])
    mask = torch.reshape(mask, [int(xs.size()[0]/n_trajectories), n_trajectories])
    final_a_estimates = torch.mean(a_estimates, axis = 1)
    print(final_a_estimates)
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

def flux_sample(V, beta, gamma, step_size, a_center, b_center, basin_cutoff, n_crossings, stride = 1):
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
        for i in range(stride):
            x = Langevin_step(x, V, beta, gamma, step_size)
        n_steps += stride
        if torch.sqrt(torch.sum(torch.square(x - a_center))) > basin_cutoff and in_basin:
            just_left_flag = True
            #print("Leaving Basin")
            in_basin = False
            crossings += 1
            print(crossings)
            escape_confs.append(x.squeeze())
            escape_times.append(n_steps)
            n_steps = 0
            from_A = False
        if torch.sqrt(torch.sum(torch.square(x - b_center))) < basin_cutoff:
           print("Actual Transition")
           x = a_center.unsqueeze(0)
           n_steps = 0
        if torch.sqrt(torch.sum(torch.square(x - a_center))) < basin_cutoff and from_A == False:
            #print("Re-entering Basin")
            from_A = True
            in_basin = True
    return torch.tensor(escape_times)*step_size, torch.stack(escape_confs)
