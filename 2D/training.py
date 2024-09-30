import torch
import time
import sampling

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

def V(x):
    return 3*torch.exp(-x[:,0]**2 - (x[:,1] - (1/3))**2) - 3*torch.exp(-x[:,0]**2 - (x[:,1]-(5/3))**2) - 5*torch.exp(-(x[:,0]-1)**2 - x[:,1]**2) - 5*torch.exp(-(x[:,0]+1)**2 - x[:,1]**2) + 0.2*x[:,0]**4 +0.2*(x[:,1]-(1/3))**4

def dist(x, y):
    return torch.sqrt(torch.sum(torch.square(x-y), axis = -1))

def RMSE_loss(net, data, targets):
    predictions = torch.sigmoid(net(data))
    loss = (torch.mean(torch.square(predictions - targets)))
    return loss
    
def exploration_loss(net, data):
    predictions = torch.sigmoid(net(data))
    loss = (torch.mean(torch.square(torch.log(predictions) - torch.log(torch.tensor(0.5)))))
    return loss

def full_loss(net, data, reporters, a_center, b_center, cutoff, n_trajectories):
    targets = sampling.calculate_committor_estimates(reporters, net, a_center, b_center, cutoff, n_trajectories)
    predictions = torch.sigmoid(net(data))
    loss = (torch.mean(torch.square(predictions - targets)))
    return loss
    
def half_loss(net, data, a_targets, b_targets):
    a_predictions = torch.sigmoid(net(data))
    b_predictions = torch.sigmoid(-net(data))
    individual_losses = (torch.square(a_predictions - a_targets) + (torch.square(b_predictions - b_targets)))
    loss = torch.mean((torch.square(a_predictions - a_targets) + (torch.square(b_predictions - b_targets))))
    return loss, individual_losses
    
def inverse_half_loss(net, data, a_targets, b_targets):
    a_predictions = torch.sigmoid(net(data))
    b_predictions = torch.sigmoid(-net(data))
    #a_indices = torch.where(torch.logical_and(torch.logical_not(torch.isinf(torch.log(a_targets))),torch.logical_not(torch.isinf(torch.log(a_predictions)))))

    #b_indices = torch.where(torch.logical_and(torch.logical_not(torch.isinf(torch.log(b_targets))),torch.logical_not(torch.isinf(torch.log(b_predictions)))))
    
    a_indices = torch.where(a_targets != 0)
    b_indices = torch.where(b_targets != 0)
    
    a_predictions, a_targets = a_predictions[a_indices], a_targets[a_indices]
    b_predictions, b_targets = b_predictions[b_indices], b_targets[b_indices]
    
    a_indices = torch.where(a_targets != 1)
    b_indices = torch.where(b_targets != 1)
    #print(len(a_indices[0])+len(b_indices[0]))
    
    a_predictions, a_targets = a_predictions[a_indices], a_targets[a_indices]
    b_predictions, b_targets = b_predictions[b_indices], b_targets[b_indices]
    
    loss = torch.mean(torch.square(1/a_predictions - 1/a_targets)+(torch.square(1/b_predictions - 1/b_targets)))
    return loss
    
def MFPT_half_loss(net, data, a_targets):
    a_predictions = (net(data))
    loss = (torch.mean(torch.square(a_predictions - a_targets)))
    return loss
    
def MFPT_reverse_loss(net, data, reporters, a_center, b_center, cutoff, lengths, n_trajectories, constant):
    a_predictions = net(data)
    zeros = torch.zeros(reporters.size()[0]).to(device)
    a_estimates = net(reporters) + lengths
    #a_estimates = torch.where(dist(xs, a_center) < cutoff, constant, a_estimates)
    a_estimates_1 = torch.where(dist(reporters, b_center) < cutoff, zeros, a_estimates)
    a_estimates_2 = torch.reshape(a_estimates_1, [int(reporters.size()[0]/n_trajectories), n_trajectories])
    a_targets = torch.mean(a_estimates)
    #print(a_predictions)
    #print(a_targets)
    loss = (torch.mean(torch.square(a_predictions - a_targets)))
    print(loss)
    return loss
    
def MFPT_correspondence_loss(MFPT_net, data, center_prediction, committor_targets):
    predictions = MFPT_net(data)
    loss = torch.mean(torch.square(predictions - (1-committor_targets)*center_prediction))
    return loss
    
def Committor_correspondence_loss(net, data, center_prediction, MFPT_targets):
    predictions = torch.sigmoid(net(data))
    loss = torch.mean(torch.square((1-predictions) - (MFPT_targets)/torch.max(MFPT_targets)))
    return loss
    
def BCE_loss(net, data, a_targets, b_targets):
    loss = torch.nn.BCELoss()
    #if torch.any(a_predictions < 0.01) and torch.any(b_predictions< 0.01):
    #    a_indices = torch.where(a_predictions < 0.01)
    #    b_indices = torch.where(b_predictions < 0.01)
    #    return loss(a_predictions[a_indices], a_targets[a_indices]) + loss(b_predictions[b_indices], b_targets[b_indices])
    #else:
    return loss(torch.sigmoid(net(data)), a_targets)

def half_loss_scaled(net, data, a_targets, b_targets):
    a_predictions = torch.sigmoid(net(data))
    b_predictions = torch.sigmoid(-net(data))
    #indices = torch.nonzero(a_predictions*b_predictions, as_tuple = True)
    #loss = (torch.mean(torch.square(a_predictions[indices] - a_targets[indices])/(a_predictions[indices]*b_predictions[indices])))
    individual_losses = (torch.square(a_predictions - a_targets)/a_targets) + (torch.square(b_predictions - b_targets)/b_targets)
    loss = torch.mean(torch.square(a_predictions - a_targets)/a_targets) + torch.mean(torch.square(b_predictions - b_targets)/b_targets)
    return loss, individual_losses

def half_log_loss(net, data, a_targets, b_targets, a_long_targets, b_long_targets):
    a_predictions = torch.sigmoid(net(data))
    b_predictions = torch.sigmoid(-net(data))
    #a_indices = torch.where(torch.logical_and(torch.logical_not(torch.isinf(torch.log(a_targets))),torch.logical_not(torch.isinf(torch.log(a_predictions)))))

    #b_indices = torch.where(torch.logical_and(torch.logical_not(torch.isinf(torch.log(b_targets))),torch.logical_not(torch.isinf(torch.log(b_predictions)))))
    
    a_indices = torch.where(a_targets != 0)
    b_indices = torch.where(b_targets != 0)
    
    a_predictions, a_targets = a_predictions[a_indices], a_targets[a_indices]
    b_predictions, b_targets = b_predictions[b_indices], b_targets[b_indices]
    
    a_indices = torch.where(a_targets != 1)
    b_indices = torch.where(b_targets != 1)
    
    #print(len(a_indices[0]), len(b_indices[0]))
    #if len(a_indices[0]) == 0 or len(b_indices[0]) == 0:
        #print("OOF")
        #return 0*torch.mean(a_predictions), 0*a_predictions
    
    a_predictions, a_targets = a_predictions[a_indices], a_targets[a_indices]
    b_predictions, b_targets = b_predictions[b_indices], b_targets[b_indices]
    #a_indices = torch.where(a_predictions < 0.01)
    #b_indices = torch.where(b_predictions < 0.01)
    #if len(a_indices[0]) == 0 or len(b_indices[0]) == 0:
    #    return torch.tensor(0)
    
    #a_predictions, a_targets = a_predictions[a_indices], a_targets[a_indices]
    #b_predictions, b_targets = b_predictions[b_indices], b_targets[b_indices]
    
    #print(torch.stack((torch.log(a_predictions), torch.log(a_targets))))
    #print(torch.stack((torch.log(b_predictions), torch.log(b_targets))))
    #a_static = a_predictions.clone().detach()
    #b_static = b_predictions.clone().detach()
    #individual_losses = ((torch.square(torch.log(a_predictions) - torch.log(a_targets)))) + ((torch.square(torch.log(b_predictions) - torch.log(b_targets))))
    #print(a_predictions, a_targets)
    return (torch.mean(torch.square(torch.log(a_predictions) - torch.log(a_targets)))) + (torch.mean(torch.square(torch.log(b_predictions) - torch.log(b_targets)))), None #individual_losses
    
def hybrid_loss(net, data, a_long_targets, b_long_targets, a_short_targets, b_short_targets):
    a_predictions = torch.sigmoid(net(data))
    b_predictions = torch.sigmoid(-net(data))

    losses = (torch.square(a_predictions - a_long_targets) + (torch.square(b_predictions - b_long_targets)))
    log_losses = ((torch.square(torch.log(a_predictions) - torch.log(a_short_targets)))) + ((torch.square(torch.log(b_predictions) - torch.log(b_short_targets))))
    return torch.mean(torch.maximum(losses, log_losses))
    
def half_log_loss_2(net, data, a_targets, b_targets):
    a_predictions = torch.sigmoid(net(data))
    b_predictions = torch.sigmoid(-net(data))
    return (torch.mean(torch.square(torch.log(1+a_predictions) - torch.log(1+a_targets)))) + (torch.mean(torch.square(torch.log(2-a_predictions) - torch.log(2-a_targets)))), ((torch.square(torch.log(1+a_predictions) - torch.log(1+a_targets)))) + ((torch.square(torch.log(2-a_predictions) - torch.log(2-a_targets))))
    #return log_loss_a + log_loss_b
    
def individual_log_loss(net, data, a_target, b_target):
    a_prediction = torch.sigmoid(net(data))
    b_prediction = torch.sigmoid(-net(data))
    log_loss_a = (torch.mean(torch.square(torch.log(a_prediction) - torch.log(a_target))))
    log_loss_b = (torch.mean(torch.square(torch.log(b_prediction) - torch.log(b_target))))
    return log_loss_a + log_loss_b, log_loss_a + log_loss_b
    
def laplace_loss(net, data):
    laplacian = torch.zeros(data.shape[0])

    for i, xi in enumerate(data):
        hess = torch.autograd.functional.hessian(net, xi.unsqueeze(0), create_graph=True)
        laplacian[i] = torch.diagonal(hess,offset=0).sum()
    return torch.sum(torch.square(laplacian))

    
def logit_loss(net, data, a_targets, b_targets):
    a_predictions = net(data)
    b_predictions = -net(data)
    #a_indices = torch.where(torch.logical_and(torch.logical_not(torch.isinf(torch.log(a_targets))),torch.logical_not(torch.isinf(torch.log(a_predictions)))))
    #b_indices = torch.where(torch.logical_and(torch.logical_not(torch.isinf(torch.log(b_targets))),torch.logical_not(torch.isinf(torch.log(b_predictions)))))
    #print(a_targets)
    #print(torch.logit(a_targets))
    #print(torch.logit(b_targets))
    return (torch.mean(torch.square(a_predictions - torch.logit(a_targets)))) + (torch.mean(torch.square(b_predictions - torch.logit(b_targets)))), ((torch.square(a_predictions - torch.logit(a_targets)))) + ((torch.square(b_predictions - torch.logit(b_targets))))
    
    
def max_loss(net, data, a_targets, b_targets, a_short_targets, b_short_targets):
    a_predictions = torch.sigmoid(net(data))
    b_predictions = torch.sigmoid(-net(data))
    
    a_indices = torch.where(torch.logical_and((torch.isinf(torch.log(a_short_targets))),(torch.isinf(torch.log(a_predictions)))))

    b_indices = torch.where(torch.logical_and((torch.isinf(torch.log(b_short_targets))),(torch.isinf(torch.log(b_predictions)))))
    
    loss = (torch.square(a_predictions - a_targets)) + (torch.square(b_predictions - b_targets))
    log_loss = torch.square(torch.log(a_predictions) - torch.log(a_short_targets)) + (torch.square(torch.log(b_predictions) - torch.log(b_short_targets)))
    log_loss
    
    return torch.mean(torch.maximum(loss, log_loss))


def eigen_loss(net, data, a_targets, b_targets):
    predictions = torch.sigmoid(net(data))
    indices = torch.where(torch.logical_not(torch.isinf(torch.log(a_targets))))
    int_predictions = predictions[indices]
    int_targets = a_targets[indices]
    indices = torch.where(torch.logical_not(torch.isinf(torch.log(int_predictions))))

    log_loss_a = (torch.mean(torch.square(torch.log(int_predictions[indices]) - torch.log(int_targets[indices]))))
    eigen_a = torch.square(torch.sum(torch.log(int_predictions[indices]) - torch.log(int_targets[indices])))/(len(indices[0])**2)
    
    b_predictions = torch.sigmoid(-net(data))
    #print(torch.min(b_targets), torch.min(b_predictions))
    indices = torch.where(torch.logical_not(torch.isinf(torch.log(b_targets))))
    predictions = b_predictions[indices]
    targets = b_targets[indices]
    indices = torch.where(torch.logical_not(torch.isinf(torch.log(predictions))))
    #print(torch.min(predictions[indices]), torch.min(targets[indices]))

    log_loss_b = (torch.mean(torch.square(torch.log(predictions[indices]) - torch.log(targets[indices]))))
    eigen_b = torch.square(torch.sum(torch.log(predictions[indices]) - torch.log(targets[indices])))/(len(indices[0])**2)
    
    return (log_loss_a - 0.5*eigen_a) + (log_loss_b - 0.5*eigen_b)

def full_log_loss(net, data, reporters, a_center, b_center, cutoff, n_trajectories):
    targets = sampling.calculate_committor_estimates(reporters, net, a_center, b_center, cutoff, n_trajectories)
    predictions = net(data)
    indices = torch.nonzero(targets, as_tuple = True)
    int_predictions = predictions[indices]
    int_targets = targets[indices]
    indices = torch.nonzero(int_predictions, as_tuple = True)

    log_loss_a = (torch.mean(torch.square(torch.log(int_predictions[indices]) - torch.log(int_targets[indices]))))
    
    modified_predictions = torch.abs(1-predictions)
    modified_targets = torch.abs(1-targets)
    indices = torch.nonzero(modified_targets, as_tuple = True)
    predictions = modified_predictions[indices]
    targets = modified_targets[indices]
    indices = torch.nonzero(predictions, as_tuple = True)

    log_loss_b = (torch.mean(torch.square(torch.log(predictions[indices]) - torch.log(targets[indices]))))
    return log_loss_a + log_loss_b

def train_init(net, data, targets, n_init_train_steps=10000, lr=0.1, verbose = True, report_step = 1000):
    start_time = time.time()
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    for i in range(n_init_train_steps): # For each step specified by argument
        net.zero_grad()
        optimizer.zero_grad()
        loss = RMSE_loss(net, data, targets)
        if verbose:
            if i % report_step == 0:
                epoch_time = time.time() - start_time
                print("Initial Training Step {}: Loss = {} ({:0.3f} seconds)".format(i, loss, epoch_time))
                start_time = time.time()
        loss.backward()
        optimizer.step()

    return net
