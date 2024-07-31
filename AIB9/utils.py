import os, math, sys
import matplotlib.pyplot as plt
import openmm.app  as omm_app
import openmm as omm
import simtk.unit as unit
import openmmtools as ommt
import torch
import torch.nn as nn
from openmmtorch import TorchForce
from scipy.interpolate import griddata
from parmed import load_file
import time
import numpy as np
import copy
from torch.autograd import Variable, Function
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
import haiku as hk
import optax
XLA_PYTHON_CLIENT_PREALLOCATE=False

from IPython.display import display
from IPython.display import clear_output

from openmm.app.internal import unitcell

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("mycmap", ['#1B346C','#01ABE9','#F1F8F1','#F54B1A'])
cmap2 = LinearSegmentedColormap.from_list("mycmap2", ['#ffffff','#000000','#ffffff'])
cmap3 = LinearSegmentedColormap.from_list("mycmap3", ['#ffffff','#000000'])
from matplotlib.cm import register_cmap
register_cmap(name="mycmap",cmap=cmap)
register_cmap(name="mycmap2",cmap=cmap2)
register_cmap(name="mycmap3",cmap=cmap3)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
def dist(x, y):
    """Computes the Euclidean distance between points x and y.
    Args:
        x,y: The two points whose distance will be computed
    Returns:
        The distance between the two points
    """
    
    return np.sqrt(np.sum((x - y)**2, axis = -1))

def compute_alanine_dihedrals(positions): 
    """Calculates the [phi,psi] dihedral angles in an alanine dipeptide molecule.
    Args:
        positions: The atomic positions of each atom in the alanine dipeptide molecule, in nm
    Returns:
        An array of the [phi, psi] dihedral angles of the molecule (in degrees).
    """
    
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
        b1 /= np.linalg.norm(b1)

        # vector projections
        v = b0 - np.dot(b0, b1)*b1
        w = b2 - np.dot(b2, b1)*b1

        # angle between v and w in a plane is the torsion angle
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.degrees(np.arctan2(y, x))
    
    #boxvectors = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    #boxsize = np.diagonal(boxvectors)
    #positions = positions - np.floor(positions/boxsize)*boxsize
    a1 = _compute_dihedrals(positions[4], positions[6], positions[8], positions[14])
    a2 = _compute_dihedrals(positions[6], positions[8], positions[14],positions[16])
    return np.array([a1, a2])

def compute_aib9_dihedrals(positions): 
    
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
        b1 /= np.linalg.norm(b1)

        # vector projections
        v = b0 - np.dot(b0, b1)*b1
        w = b2 - np.dot(b2, b1)*b1

        # angle between v and w in a plane is the torsion angle
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.degrees(np.arctan2(y, x))
    
    positions = np.array(positions)
    phi1 = _compute_dihedrals(positions[30], positions[32], positions[34], positions[43])
    phi2 = _compute_dihedrals(positions[43], positions[45], positions[47], positions[56])
    phi3 = _compute_dihedrals(positions[56], positions[58], positions[60], positions[69])
    phi4 = _compute_dihedrals(positions[69], positions[71], positions[73], positions[82])
    phi5 = _compute_dihedrals(positions[82], positions[84], positions[86], positions[95])
    phis = np.stack((phi1, phi2, phi3, phi4, phi5))
    
    psi1 = _compute_dihedrals(positions[32], positions[34], positions[43], positions[45])
    psi2 = _compute_dihedrals(positions[45], positions[47], positions[56], positions[58])
    psi3 = _compute_dihedrals(positions[58], positions[60], positions[69], positions[71])
    psi4 = _compute_dihedrals(positions[71], positions[73], positions[82], positions[84])
    psi5 = _compute_dihedrals(positions[84], positions[86], positions[95], positions[97])
    psis = np.stack((psi1, phi2, psi3, psi4, psi5))
    
    return np.array([phi3, psi3])

def dihedral_difference(angle1, angle2):
    """Computes the distance between two dihedral angles, accounting for the periodic boundary.
    Args:
        angle1, angle2: The two dihedral angles whose distance is to be computed
    Returns:
        The smallest angular distance between the two angles (in degrees)
    """
    return np.sqrt(np.sum(np.square(180 - np.abs(np.abs(angle1 - angle2) - 180))))

def dihedral_differences(angle1, angle2):
    """Computes the distance between two dihedral angles, accounting for the periodic boundary.
    Args:
        angle1, angle2: The two dihedral angles whose distance is to be computed
    Returns:
        The smallest angular distance between the two angles (in degrees)
    """
    return np.sqrt(np.sum(np.square(180 - np.abs(np.abs(angle1 - angle2) - 180)), axis = 1))


def generate_AD_PMF(PMF_file):
    """Unpacks the contents of an array of energies generated from umbrella sampling 
    and generalizes the energy to a finer grid of points.
    Args:
        PMF_file: The file that contains the array of energies obtained by umbrella sampling
    Returns:
        an array of energies on a much finer scale, using cubic interpolation
        """
    
    PMF = np.load(PMF_file)
    nans = jnp.where(PMF != PMF)
    
    x = jnp.linspace(-180, 180, 1000)
    y = jnp.linspace(-180, 180, 1000)
    X, Y = jnp.meshgrid(x,y)
    
    px = jnp.linspace(-180, 180, 25)
    py = jnp.linspace(-180, 180, 25)
    PX, PY = jnp.meshgrid(px, py)
    
    offset = 0
    for i in nans:
        PX = jnp.delete(PX, i-offset)
        PY = jnp.delete(PY, i-offset)
        PMF = jnp.delete(PMF, i-offset)
        offset += 1

    surface = griddata((PX, PY), PMF, (X, Y), method = 'cubic')
    surface = jnp.array(surface)
    return surface

def V_ad(x,y,surface):
    """Uses the surface generated from generate_AD_PMF to calculate the energy of a configuration
    Args:
        x,y: The input configuration
        surface: The surface grid of energies used to calculate the energy
    Returns:
        The energy of the configuration
        """
    phi_bin = ((x + 180) // (360/1000)).astype(int)
    psi_bin = ((y + 180) // (360/1000)).astype(int)
    return surface[psi_bin, phi_bin]

#Vectorized form of V_ad to generate a grid of energies based on grid points
pmf_grid_function = vmap(vmap(V_ad, in_axes = (0, None, None)), in_axes = (None, 0, None))


def generate_pmf_graph(surface):
    """Generates a grid representation of a potential energy surface
    Args:
        surface: The surface to be represented
    Returns:
        A grid of energies that can be used to plot a potential energy surface
        """
    
    x = jnp.linspace(-180,180,50)
    y = jnp.linspace(-180,180,50)

    Z = pmf_grid_function(x,y,surface)
    Z = jnp.transpose(Z)
    return Z

def generate_committor_contour(params, batch_predict_fn):
    """Generates a grid of committor values in dihedral space
    Args:
        params: Parameters of a neural network
        batch_predict_fn: The JAX batch prediction function used to predict committors
    Returns:
        committor_contour: A grid of committor values in dihedral space
        """
    
    x = jnp.linspace(-180,180,50)
    y = jnp.linspace(-180,180,50)
    X, Y = jnp.meshgrid(x,y)
    X, Y = jnp.ravel(X), jnp.ravel(Y)
    positions = jnp.array(list(zip(X, Y)))
    committor_contour = jnp.reshape(batch_predict_fn.apply(params, positions), [50,50])
    return committor_contour

def generate_log_committor_contour(params, batch_predict_fn):
    """Generates a grid of log(q(1-q)) values in dihedral space
    Args:
        params: Parameters of a neural network
        batch_predict_fn: The JAX batch prediction function used to predict committors
    Returns:
        committor_contour: A grid of log(q(1-q)) values in dihedral space
        """
    
    x = jnp.linspace(-180,180,50)
    y = jnp.linspace(-180,180,50)
    X, Y = jnp.meshgrid(x,y)
    X, Y = jnp.ravel(X), jnp.ravel(Y)
    positions = jnp.array(list(zip(X, Y)))
    committor_contour = jnp.reshape(jnp.log(batch_predict_fn.apply(params, positions) * (1 - batch_predict_fn.apply(params, positions))), [50,50])
    return committor_contour

class PeriodicNet(hk.Module):
    """A layer that periodicizes inputs for a neural network (see e.g. Dong and Ni, 2020).
    Attributes:
        dim: The number of dimensions of the input (and output)
        n_cos_fns: The number of cosine functions that will be summed to produce a periodic output
        period: The period of the cosine functions used
        nl: The nonlinearity applied to the output of each cosine function
    """
    def __init__(self, dim, n_cos_fns, output_size,  period = 2*jnp.pi/360, nl = jax.nn.leaky_relu):
        super().__init__()
        self.dim, self.n_cos_fns, self.output_size, self.period, self.nl = dim, n_cos_fns, output_size, period, nl
    
    def __call__(self, x):
        """Runs the periodic net. This net takes an input x, and applies a unique cosine function with 
        parameterized amplitude, offset, and shift to each dimension of x, n_cos_fns times. The nonlinearity 
        nl is applied to each cosine output, which outputs are then summed together to create a new
        periodic representation of x.
            
        Args:
            x: Input to the periodic net
        Returns:
            An output with the same size as x that is now periodic with period given by the instance of the net
        """
        
        w = hk.get_parameter("w", [self.n_cos_fns, self.dim, 3], init = hk.initializers.RandomNormal())
        W = hk.get_parameter("W", [self.dim, self.output_size, self.n_cos_fns], init = hk.initializers.RandomNormal())
        B = hk.get_parameter("B", [self.output_size], init = hk.initializers.RandomNormal())
        vis = self.nl(w[:,:,0]*jnp.cos(self.period*jnp.reshape(jnp.tile(x,self.n_cos_fns),[self.n_cos_fns,self.dim])+w[:,:,1])+w[:,:,2])
        output = jnp.sum(jnp.sum(jnp.matmul(W,vis), axis = 2), axis = 0)
        y = self.nl(output)
        return y
    
def generate_2D_projection(dihedrals, targets, n_steps = 10000, lr = 1e-3):
    """Generates a 2D projection of a high-dimensional committor onto 2D dihedral space
        Args:
            dihedrals: The dihedral angles of points at which the committor has been predicted
            targets: Target committor values to be learned
            n_steps: The number of optimization steps to learn the committor
            lr: The ADAM learning rate at which optimization will be carried out
        Returns:
            contour: A grid of committor (q) values in dihedral space
            log_contour: A grid of log(q(1-q)) values in dihedral space
        """
    
    
    
    def forward(x):
        """The main CommittorNet prediction function.
        Args:
            x: A configuration in state space (or cv space)
        Returns:
            The net's current prediction of the committor value at that point.
        """
        A = PeriodicNet(2,20,10)
        x = jax.nn.relu(A(x))
        B = hk.Linear(100)
        x = jax.nn.relu(B(x))
        C = hk.Linear(1)
        x = jax.nn.sigmoid(C(x))
        return(jnp.reshape(x, ()))
    
    def train(params, data, targets, n_init_train_steps=n_steps, lr=lr, verbose = True, report_step = 1000):
        """Runs the main JAX training loop.
        Args:
            params: The JAX parameters for the 2D committor network
            data: The 2D dihedral angles for which committor values have been predicted
            targes: The target committor values to be learned
            n_init_train_steps: The number of optimization steps
            lr: The learning rate of the ADAM optimizer
            verbose: Whether or not training progress is printed
            report_step: How often a verbose report is generated
        Returns:
            params: The fully trained network parameters
        """
    
        @jit
        def _RMSLE_loss(params, data, targets):
            "Symmetrized root-mean-square-log-error calculation"
            predictions = batch_predict.apply(params, data)
            targets = jnp.reshape(targets, jnp.shape(predictions))
            loss = (jnp.mean(jnp.square(jnp.log(predictions) - jnp.log(targets)))) + (jnp.mean(jnp.square(jnp.log(1 - predictions) - jnp.log(1 - targets)))) + l2_loss(params)
            return loss

        @jit
        def l2_loss(params):
            "Regularization for nicer plots"
            return 0.01 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))

        @jit
        def _update(params, opt_state):
            "Quick jittable parameter update"
            grads = jax.grad(_RMSLE_loss)(params, data, targets)
            updates, opt_state = opt_update(grads, opt_state)
            return optax.apply_updates(params, updates), opt_state
    
        opt_init, opt_update = optax.adam(lr)
        opt_state = opt_init(params)
        print("Training 2D projection...")
        for step in range(n_init_train_steps):
            if verbose:
                if step % report_step == 0:
                    print("Step %r Loss: %r"%(step, _RMSLE_loss(params, data, targets).item()))
            params, opt_state = _update(params, opt_state)

        return(params)

    batch_forward = vmap(forward, in_axes = (0))
    predict = hk.without_apply_rng(hk.transform(forward))
    batch_predict = hk.without_apply_rng(hk.transform(batch_forward))
    batch_grad = jit(vmap(grad(predict.apply, argnums = 1), in_axes = (None, 0)))

    key = jax.random.PRNGKey(42)

    dummy_x = jnp.ones([10,2])
    params = batch_predict.init(key, dummy_x)
    params = train(params, dihedrals, targets)
    contour = generate_committor_contour(params, batch_predict)
    log_contour = generate_log_committor_contour(params, batch_predict)
    return contour, log_contour

def compute_rolling_statistics(array, interval):
    """Calculates a retroactive rolling average
    Args:
        array: Array of time series data to be smoothed
        interval: Rolling averages are calculated from a maximum of {interval} previous steps
    Returns:
        rolling_means: Retroactive rolling averages of the data
        rolling_stds: Retroactive rolling standard deviations of the data
        """
    rolling_means = []
    rolling_stds = []
    for i in range(len(array)):
        if len(array[:i]) < interval:
            rolling_means.append(np.mean(array[:i]))
            rolling_stds.append(np.std(array[:i])/np.sqrt(len(array[:i])))
        else:
            rolling_means.append(np.mean(array[i-interval:i]))
            rolling_stds.append(np.std(array[i-interval:i])/np.sqrt(interval))
    return np.array(rolling_means), np.array(rolling_stds)
    
 

