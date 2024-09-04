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

class committor_constructor:                                                   #
    """ A class for learning the committor function for a given system in (r,v) space.
        This is done by using swarms of trajectores from importance-sampled points
        to train a neural network representation of the committor based on a self-consistent
        Chapman-Kolmogorov criterion.
    """
    def __init__(self, context, net, optimizer, a_indicator, b_indicator, n_reporters, n_iterations = 10000, data_path = "./committor_data/"):
        
        """Initializes the committor constructor.
    
       Args:
           context: An OpenMM context that characterizes the system and its dynamics
           net: A neural network that outputs a committor value given a set of positions and velocities
           optimizer: The PyTorch optimizer that will perform gradient descent on the net
           a_indicator: An indicator function that returns True if the configuration is in the "A" region
           b_indicator: An indicator function that returns True if the configuration is in the "B" region
           n_reporters: The size of swarms (e.g. number of trajectories per sampled point) 
           to use in committor estimation
           n_iterations: The maximum number of interations the simulation will use, needed to initiate storage
           data_path: The folder into which data used to learn the committor will be stored
        """
        
        self.context = context
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.n_particles = len(self.context.getState(getPositions = True).getPositions()/unit.nanometers)
        self.net = net
        self.optimizer = optimizer
        self.a_indicator = a_indicator
        self.b_indicator = b_indicator
        self.data_path = data_path
        self.n_reporters = n_reporters
        self.n_iterations = n_iterations
        
        # Set attributes that describe the current state of estimation
        self.current_iteration = 0
        self.a_flux = None
        self.b_flux = None
        self.a_index_permutation = None
        self.b_index_permutation = None
        self.a_current_index = 0
        self.b_current_index = 0
        self.a_transit = False
        self.b_transit = False
        self.a_transit_history = [0]
        self.b_transit_history = [0]
        self.a_rate_estimates = []
        self.b_rate_estimates = []
        self.a_times = []
        self.b_times = []
        
        
        # Determine whether to continue an existing simulation
        if os.path.isfile(self.data_path + "A_boundary_times.npy"):
            self.has_basin_samples = True
            a_escape_times = np.load(self.data_path + "A_boundary_times.npy")
            b_escape_times = np.load(self.data_path + "B_boundary_times.npy")
            self.n_boundary_samples = len(np.load(self.data_path + "A_boundary_times.npy")) - 1
            self.a_flux = 1/np.mean([a_escape_times[i+1] - a_escape_times[i] for i in range(len(a_escape_times) - 1)])
            self.b_flux = 1/np.mean([b_escape_times[i+1] - b_escape_times[i] for i in range(len(b_escape_times) - 1)])
        else:
            self.has_basin_samples = False
        
        
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
            self.init_data()
            print("Data initialized")
        
        else:
            print("Data path already exists. Reloading data...")
            self.reload_data()
            
        
    @staticmethod
    def dist(x, y):
        """Computes the Euclidean distance between points x and y.
        Args:
            x,y: The two points whose distance will be computed
        Returns:
            The distance between the two points
        """
        return np.sqrt(np.sum((x - y)**2, axis = -1))

    @staticmethod 
    def dihedral_difference(angle1, angle2):
        """Computes the distance between two dihedral angles, accounting for the periodic boundary.
        Args:
            angle1, angle2: The two dihedral angles whose distance is to be computed
        Returns:
            The smallest angular distance between the two angles (in degrees)
        """
        return np.sqrt(np.sum(np.square(180 - np.abs(np.abs(angle1 - angle2) - 180))))
    
    def init_data(self):
        """Creates storage space for trajectory information.

        """

        self.a_sample_confs = np.memmap(self.data_path + 'a_sample_confs.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_particles, 3))
        self.a_sample_velos = np.memmap(self.data_path + 'a_sample_velos.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_particles, 3))
        self.a_reporter_confs = np.memmap(self.data_path + 'a_reporter_confs.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_reporters, self.n_particles, 3))
        self.a_reporter_velos = np.memmap(self.data_path + 'a_reporter_velos.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_reporters, self.n_particles, 3))

        self.b_sample_confs = np.memmap(self.data_path + 'b_sample_confs.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_particles, 3))
        self.b_sample_velos = np.memmap(self.data_path + 'b_sample_velos.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_particles, 3))
        self.b_reporter_confs = np.memmap(self.data_path + 'b_reporter_confs.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_reporters, self.n_particles, 3))
        self.b_reporter_velos = np.memmap(self.data_path + 'b_reporter_velos.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_reporters, self.n_particles, 3))
    
    def reinitialize(self):
        """Resets the simulation state. Boundary simulations remain.

        """
        self.a_sample_confs = np.memmap(self.data_path + 'a_sample_confs.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_particles, 3))
        self.a_sample_velos = np.memmap(self.data_path + 'a_sample_velos.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_particles, 3))
        self.a_reporter_confs = np.memmap(self.data_path + 'a_reporter_confs.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_reporters, self.n_particles, 3))
        self.a_reporter_velos = np.memmap(self.data_path + 'a_reporter_velos.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_reporters, self.n_particles, 3))

        self.b_sample_confs = np.memmap(self.data_path + 'b_sample_confs.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_particles, 3))
        self.b_sample_velos = np.memmap(self.data_path + 'b_sample_velos.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_particles, 3))
        self.b_reporter_confs = np.memmap(self.data_path + 'b_reporter_confs.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_reporters, self.n_particles, 3))
        self.b_reporter_velos = np.memmap(self.data_path + 'b_reporter_velos.dat', dtype='float64', mode='w+', shape = (self.n_iterations, self.n_reporters, self.n_particles, 3))
        
        self.current_iteration = 0
        self.a_flux = None
        self.b_flux = None
        self.a_index_permutation = None
        self.b_index_permutation = None
        self.a_current_index = 0
        self.b_current_index = 0
        self.a_transit = False
        self.b_transit = False
        self.a_transit_history = [0]
        self.b_transit_history = [0]
        self.a_rate_estimates = []
        self.b_rate_estimates = []
        self.a_times = []
        self.b_times = []
    
    def reload_data(self):
        """Reloads data and state information from the data path if such data already exists.

        """

        self.a_sample_confs = np.memmap(self.data_path + 'a_sample_confs.dat', dtype='float64', mode='r+', shape = (self.n_iterations, self.n_particles, 3))
        self.a_sample_velos = np.memmap(self.data_path + 'a_sample_velos.dat', dtype='float64', mode='r+', shape = (self.n_iterations, self.n_particles, 3))
        self.a_reporter_confs = np.memmap(self.data_path + 'a_reporter_confs.dat', dtype='float64', mode='r+', shape = (self.n_iterations, self.n_reporters, self.n_particles, 3))
        self.a_reporter_velos = np.memmap(self.data_path + 'a_reporter_velos.dat', dtype='float64', mode='r+', shape = (self.n_iterations, self.n_reporters, self.n_particles, 3))

        self.b_sample_confs = np.memmap(self.data_path + 'b_sample_confs.dat', dtype='float64', mode='r+', shape = (self.n_iterations, self.n_particles, 3))
        self.b_sample_velos = np.memmap(self.data_path + 'b_sample_velos.dat', dtype='float64', mode='r+', shape = (self.n_iterations, self.n_particles, 3))
        self.b_reporter_confs = np.memmap(self.data_path + 'b_reporter_confs.dat', dtype='float64', mode='r+', shape = (self.n_iterations, self.n_reporters, self.n_particles, 3))
        self.b_reporter_velos = np.memmap(self.data_path + 'b_reporter_velos.dat', dtype='float64', mode='r+', shape = (self.n_iterations, self.n_reporters, self.n_particles, 3))
        
        state_array = np.load(self.data_path + 'state.npy', allow_pickle = True)
        
        self.current_iteration = state_array[0]
        self.a_flux = state_array[1]
        self.b_flux = state_array[2]
        self.a_index_permutation = state_array[3]
        self.b_index_permutation = state_array[4]
        self.a_current_index = state_array[5]
        self.b_current_index = state_array[6]
        self.a_transit = state_array[7]
        self.b_transit = state_array[8]
        self.a_transit_history = state_array[9]
        self.b_transit_history = state_array[10]
        self.a_rate_estimates = state_array[11]
        self.b_rate_estimates = state_array[12]
        self.a_times = state_array[13]
        self.b_times = state_array[14]
        
        self.net.load_state_dict(torch.load(self.data_path + 'net.pt'))
        
        if os.path.isfile(self.data_path + "A_boundary_times.npy"):
            self.has_basin_samples = True
            a_escape_times = np.load(self.data_path + "A_boundary_times.npy")
            b_escape_times = np.load(self.data_path + "A_boundary_times.npy")
            self.n_boundary_samples = len(np.load(self.data_path + "A_boundary_times.npy")) - 1
            self.a_flux = 1/np.mean([a_escape_times[i+1] - a_escape_times[i] for i in range(len(a_escape_times) - 1)])
            self.b_flux = 1/np.mean([b_escape_times[i+1] - b_escape_times[i] for i in range(len(b_escape_times) - 1)])
        else:
            self.has_basin_samples = False
        
        print("Data Reloaded")
        
        
    def save_data(self):
        """Saves simulation state information to disk, which can be reloaded later.

        """
        
        state_array = [self.current_iteration,
                        self.a_flux,
                        self.b_flux,
                        self.a_index_permutation,
                        self.b_index_permutation,
                        self.a_current_index,
                        self.b_current_index,
                        self.a_transit,
                        self.b_transit,
                        self.a_transit_history,
                        self.b_transit_history,
                        self.a_rate_estimates,
                        self.b_rate_estimates,
                        self.a_times,
                        self.b_times
                     ]
        np.save(self.data_path + 'state.npy', np.array(state_array,  dtype = 'object'))
        torch.save(self.net.state_dict(), self.data_path + 'net.pt')
        
        print("Data Saved")
        
    def begin_sample_chain(self, basin):
        """Begins a sample chain by launching a swarm from a randomly-sampled boundary configuration.
        
        Args:
            basin: (A or B) The basin from which to launch the swarm

        """
        if basin == "A":
            if self.a_index_permutation == None:
                self.a_index_permutation = torch.randperm(self.n_boundary_samples)
            a_index = self.a_index_permutation[self.a_current_index]
            self.a_sample_confs[self.current_iteration] = np.load(self.data_path + "A_boundary_confs.npy")[a_index].copy()
            self.a_sample_confs.flush()
            self.a_sample_velos[self.current_iteration] = np.load(self.data_path + "A_boundary_velos.npy")[a_index].copy()
            self.a_sample_velos.flush()
        elif basin == "B":
            if self.b_index_permutation == None:
                self.b_index_permutation = torch.randperm(self.n_boundary_samples)
            b_index = self.b_index_permutation[self.b_current_index]
            self.b_sample_confs[self.current_iteration] = np.load(self.data_path + "B_boundary_confs.npy")[b_index].copy()
            self.b_sample_confs.flush()
            self.b_sample_velos[self.current_iteration] = np.load(self.data_path + "B_boundary_velos.npy")[b_index].copy()
            self.b_sample_velos.flush()
            
    def continue_sample_chain(self, basin):
        """Continues a sample chain by launching a swarm from an importance-sampled point.
        
        Args:
            basin: (A or B) The basin from which the chain originated

        """

        if basin == "A":
            a_weights = np.ravel(self.load_and_predict_committors("A"))
            a_index = np.argsort(a_weights.squeeze())[-1]
            self.a_sample_confs[self.current_iteration] = np.array((np.memmap(self.data_path + 'a_reporter_confs.dat', dtype = 'float64', mode = 'r', offset = int((self.a_transit_history[-1]*self.n_reporters + a_index)*self.n_particles*3*64/8), shape = (self.n_particles, 3))))
            self.a_sample_confs.flush()
            self.a_sample_velos[self.current_iteration] = np.array((np.memmap(self.data_path + 'a_reporter_velos.dat', dtype = 'float64', mode = 'r', offset = int((self.a_transit_history[-1]*self.n_reporters + a_index)*self.n_particles*3*64/8), shape = (self.n_particles, 3))))
            self.a_sample_velos.flush()
        elif basin == "B":
            b_weights = np.ravel(self.load_and_predict_committors("B"))
            b_index = np.argsort(b_weights.squeeze())[-1]
            self.b_sample_confs[self.current_iteration] = np.array((np.memmap(self.data_path + 'b_reporter_confs.dat', dtype = 'float64', mode = 'r', offset = int((self.b_transit_history[-1]*self.n_reporters + b_index)*self.n_particles*3*64/8), shape = (self.n_particles, 3))))
            self.b_sample_confs.flush()
            self.b_sample_velos[self.current_iteration] = np.array((np.memmap(self.data_path + 'b_reporter_velos.dat', dtype = 'float64', mode = 'r', offset = int((self.b_transit_history[-1]*self.n_reporters + b_index)*self.n_particles*3*64/8), shape = (self.n_particles, 3))))
            self.b_sample_velos.flush()
    def sample_basins(self, init_A, init_B, stride = 1, n_samples = 1000):

        """Runs an equilibrium simulation from a basin to generate boundary samples.
           These samples are used to calculate a rate estimate state-state transitions
           via a product of the flux out of each state and the integrated
           committor value along the boundary. All sample information will be stored in the 
           constructor's data directory.

            Args:
               init_A, init_B: Configurations from each basin from which
               sampling will start
               stride: How often to check basin membership in the equilibrium simulation
               small values generate a more accurate representation of the boundary ensemble, but
               require more calls to indicator functions and are therefore slower
               n_samples: How many samples from each state boundary will be collected

            Returns:
               a_flux, b_flux: the flux (in 1/s) out of the state boundary
               (Positions, velocities, and times of boundary configurations are written
               to the data directory).

            """
        
        if not self.has_basin_samples:
            self.n_boundary_samples = n_samples
            # basin A
            escape_confs, escape_times, escape_velocities = [],[],[]
            crossings, n_steps = 0,0
            from_basin = False
            self.context.setTime(0)
            self.context.setPositions(init_A)
            self.context.setVelocitiesToTemperature(self.context.getIntegrator().getTemperature()/unit.kelvin)
            while crossings < n_samples + 1:
                self.context.getIntegrator().step(stride)
                state = self.context.getState(getPositions = True, getVelocities = True)
                positions = state.getPositions()/unit.nanometers
                if self.a_indicator(positions) and not from_basin:
                    from_basin = True
                if not self.a_indicator(positions) and from_basin:
                    escape_confs.append(state.getPositions()/unit.nanometers)
                    escape_velocities.append(state.getVelocities()/(unit.nanometers/unit.picoseconds))
                    escape_times.append(self.context.getState().getTime()/unit.nanoseconds)
                    from_basin = False
                    crossings += 1
            np.save(self.data_path + "A_boundary_times.npy", np.array(escape_times))
            np.save(self.data_path + "A_boundary_confs.npy", np.array(escape_confs))
            np.save(self.data_path + "A_boundary_velos.npy", np.array(escape_velocities))
            a_flux = 1/np.mean([escape_times[i+1] - escape_times[i] for i in range(len(escape_times) - 1)])
            print("Boundary samples from state A have been generated.")
            print(f"Flux out of boundary A: {a_flux}/ns")



            # basin B
            escape_confs, escape_times, escape_velocities = [],[],[]
            crossings, n_steps = 0,0
            from_basin = False
            self.context.setTime(0)
            self.context.setPositions(init_B)
            self.context.setVelocitiesToTemperature(self.context.getIntegrator().getTemperature()/unit.kelvin)
            while crossings < n_samples + 1:
                self.context.getIntegrator().step(stride)
                state = self.context.getState(getPositions = True, getVelocities = True, getEnergy = True)
                positions = state.getPositions()/unit.nanometers
                if self.b_indicator(positions) and not from_basin:
                    from_basin = True
                if not self.b_indicator(positions) and from_basin:
                    escape_confs.append(state.getPositions()/unit.nanometers)
                    escape_velocities.append(state.getVelocities()/(unit.nanometers/unit.picoseconds))
                    escape_times.append(self.context.getState().getTime()/unit.nanoseconds)
                    from_basin = False
                    crossings += 1
            np.save(self.data_path + "B_boundary_times.npy", np.array(escape_times))
            np.save(self.data_path + "B_boundary_confs.npy", np.array(escape_confs))
            np.save(self.data_path + "B_boundary_velos.npy", np.array(escape_velocities))
            b_flux = 1/np.mean([escape_times[i+1] - escape_times[i] for i in range(len(escape_times) - 1)])
            print("Boundary samples from state B have been generated.")
            print(f"Flux out of boundary B: {b_flux}/ns")
            self.a_flux, self.b_flux = a_flux, b_flux
            np.save(self.data_path + "A_flux.npy", np.array([a_flux]))
            np.save(self.data_path + "B_flux.npy", np.array([b_flux]))
            self.n_basin_samples = n_samples
        else:
            print("Equilibrium basin simulations already exist in the data path.")
        
    def load_and_predict_committors(self, basin, max_chunk = 100, post_collection = False):
        """Utility for loading swarm endpoints and predicting their committor values.
        
        Args:
            basin: (A or B) The origin of the sampled points for which committor targets are calculated
            max_chunk: The largest number of swarms that are loaded into memory at once
            post_collection: Whether or not committors are loaded before or after samples in the current
                step have been collected. If True, the total number of swarms loaded is increased by 1.
        Retuns:
            committors: A (current_iteration(+1 if post_collection) x n_reporters) array of committor
                values calculated at the endpoints of each trajectory in each swarm

        """
        if post_collection:
            A = 1
            B = 0
        else:
            A = 0
            B = 1
        committors = []
        confs = []
        velos = []
        if basin == "A":
            
            n_iterations = (self.current_iteration - B*self.a_transit_history[-1] + A)//max_chunk
            residue = (self.current_iteration - B*self.a_transit_history[-1] + A)%max_chunk
            for iteration in range(n_iterations):
                iter_confs = torch.tensor(np.memmap(self.data_path + "a_reporter_confs.dat", dtype = 'float64', mode = 'r', offset = int((B*self.a_transit_history[-1]+(iteration)*max_chunk)*self.n_reporters*self.n_particles*3*64/8), shape = (max_chunk*self.n_reporters, self.n_particles, 3))).to(self.device)
                iter_velos = torch.tensor(np.memmap(self.data_path + "a_reporter_velos.dat", dtype = 'float64', mode = 'r', offset = int((B*self.a_transit_history[-1]+(iteration)*max_chunk)*self.n_reporters*self.n_particles*3*64/8), shape = (max_chunk*self.n_reporters, self.n_particles, 3))).to(self.device)
                iter_committors = torch.sigmoid(self.net(iter_confs, iter_velos))
                committors.append(iter_committors.cpu().detach().numpy())
                confs.append(iter_confs.cpu().detach().numpy())
                velos.append(iter_velos.cpu().detach().numpy())
            # Add residual
            if residue != 0:
                iter_confs = torch.tensor(np.memmap(self.data_path + "a_reporter_confs.dat", dtype = 'float64', mode = 'r', offset = int((B*self.a_transit_history[-1]+n_iterations*max_chunk)*self.n_reporters*self.n_particles*3*64/8), shape = (residue*self.n_reporters, self.n_particles, 3))).to(self.device)
                iter_velos = torch.tensor(np.memmap(self.data_path + "a_reporter_velos.dat", dtype = 'float64', mode = 'r', offset = int((B*self.a_transit_history[-1]+n_iterations*max_chunk)*self.n_reporters*self.n_particles*3*64/8), shape = (residue*self.n_reporters, self.n_particles, 3))).to(self.device)
                iter_committors = torch.sigmoid(self.net(iter_confs, iter_velos))
                committors.append(iter_committors.cpu().detach().numpy())
                confs.append(iter_confs.cpu().detach().numpy())
                velos.append(iter_velos.cpu().detach().numpy())
            confs = np.concatenate(confs)
            velos = np.concatenate(velos)
            committors = np.concatenate(committors).reshape(-1)
            committors = np.where(self.a_indicator(confs), np.zeros_like(committors), committors)
            committors = np.where(self.b_indicator(confs), np.ones_like(committors), committors)
            
        elif basin == "B":
            n_iterations = (self.current_iteration - B*self.b_transit_history[-1] + A)//max_chunk
            residue = (self.current_iteration - B*self.b_transit_history[-1] + A)%max_chunk
            for iteration in range(n_iterations):
                iter_confs = torch.tensor(np.memmap(self.data_path + "b_reporter_confs.dat", dtype = 'float64', mode = 'r', offset = int((B*self.b_transit_history[-1]+(iteration)*max_chunk)*self.n_reporters*self.n_particles*3*64/8), shape = (max_chunk*self.n_reporters, self.n_particles, 3))).to(self.device)
                iter_velos = torch.tensor(np.memmap(self.data_path + "b_reporter_velos.dat", dtype = 'float64', mode = 'r', offset = int((B*self.b_transit_history[-1]+(iteration)*max_chunk)*self.n_reporters*self.n_particles*3*64/8), shape = (max_chunk*self.n_reporters, self.n_particles, 3))).to(self.device)
                iter_committors = torch.sigmoid(-self.net(iter_confs, iter_velos))
                committors.append(iter_committors.cpu().detach().numpy())
                confs.append(iter_confs.cpu().detach().numpy())
                velos.append(iter_velos.cpu().detach().numpy())
            # Add residual
            if residue != 0:
                iter_confs = torch.tensor(np.memmap(self.data_path + "b_reporter_confs.dat", dtype = 'float64', mode = 'r', offset = int((B*self.b_transit_history[-1]+n_iterations*max_chunk)*self.n_reporters*self.n_particles*3*64/8), shape = (residue*self.n_reporters, self.n_particles, 3))).to(self.device)
                iter_velos = torch.tensor(np.memmap(self.data_path + "b_reporter_velos.dat", dtype = 'float64', mode = 'r', offset = int((B*self.b_transit_history[-1]+n_iterations*max_chunk)*self.n_reporters*self.n_particles*3*64/8), shape = (residue*self.n_reporters, self.n_particles, 3))).to(self.device)
                iter_committors = torch.sigmoid(-self.net(iter_confs, iter_velos))
                committors.append(iter_committors.cpu().detach().numpy())
                confs.append(iter_confs.cpu().detach().numpy())
                velos.append(iter_velos.cpu().detach().numpy())
            confs = np.concatenate(confs)
            velos = np.concatenate(velos)
            committors = np.concatenate(committors).reshape(-1)
            committors = np.where(self.b_indicator(confs), np.zeros_like(committors), committors)
            committors = np.where(self.a_indicator(confs), np.ones_like(committors), committors)
        return committors.reshape(-1, self.n_reporters)
    
    def sample_swarm(self, basin, n_steps, stride, fraction):
        """Initiates a swarm of trajectories from the most recently sampled point and saves their endpoints to disk.
        
        Args:
            basin: (A or B) The origin of the sampled point
            n_steps: (AKA tau) The maximum length of each trajectory in the swarm, in units of the stride
            stride: How often (in number of steps) basin membership for a swarm trajectory is checked
            fraction: Fraction of swarms that must re-enter a basin to trigger an early stopping criterion

        """
        time = 0
        if basin == "A":
            reporter_positions = []
            reporter_velocities = []
            positions = np.array(np.memmap(self.data_path + 'a_sample_confs.dat', dtype = 'float64', mode = 'r', offset = int(self.current_iteration*self.n_particles*3*64/8), shape = (self.n_particles, 3)))
            end_flags = np.zeros(self.n_reporters)
            for n in range(self.n_reporters):
                self.context.setPositions(positions)
                self.context.setVelocitiesToTemperature(self.context.getIntegrator().getTemperature()/unit.kelvin)
                state = self.context.getState(getPositions = True, getVelocities = True, enforcePeriodicBox = True)
                reporter_position = state.getPositions()/unit.nanometers
                reporter_velocity = state.getVelocities()/unit.nanometers*unit.picosecond
                if self.a_indicator(reporter_position) or self.b_indicator(reporter_position):
                    self.a_reporter_confs[self.current_iteration] = np.array([reporter_position]*self.n_reporters)
                    self.a_reporter_confs.flush()
                    self.a_reporter_velos[self.current_iteration] = np.array([reporter_velocity]*self.n_reporters)
                    self.a_reporter_velos.flush()
                    return
                reporter_positions.append(reporter_position)
                reporter_velocities.append(reporter_velocity)

            for steps in range(n_steps):
                if np.sum(end_flags)/self.n_reporters > fraction:
                    break
                for n in range(self.n_reporters):
                    if end_flags[n] == 0:
                        self.context.setPositions(reporter_positions[n])
                        self.context.setVelocities(reporter_velocities[n])
                        self.context.getIntegrator().step(stride)
                        time += stride
                        state = self.context.getState(getPositions = True, getVelocities = True, enforcePeriodicBox = True)
                        reporter_position = state.getPositions()/unit.nanometers
                        reporter_velocity = state.getVelocities()/unit.nanometers*unit.picosecond
                        if self.a_indicator(reporter_position):
                            end_flags[n] = 1
                        elif self.b_indicator(reporter_position):
                            end_flags[n] = 1
                            self.a_transit = True
                        reporter_positions[n] = reporter_position
                        reporter_velocities[n] = reporter_velocity
                
            self.a_reporter_confs[self.current_iteration] = reporter_positions.copy()
            self.a_reporter_confs.flush()
            self.a_reporter_velos[self.current_iteration] = reporter_velocities.copy()
            self.a_reporter_velos.flush()
            if len(self.a_times) == 0:
                self.a_times.append(time)
            else:
                self.a_times.append(self.a_times[-1] + time)
                
        elif basin == "B":
            reporter_positions = []
            reporter_velocities = []
            positions = np.array(np.memmap(self.data_path + 'b_sample_confs.dat', dtype = 'float64', mode = 'r', offset = int(self.current_iteration*self.n_particles*3*64/8), shape = (self.n_particles, 3)))           
            end_flags = np.zeros(self.n_reporters)
            for n in range(self.n_reporters):
                self.context.setPositions(positions)
                self.context.setVelocitiesToTemperature(self.context.getIntegrator().getTemperature()/unit.kelvin)
                state = self.context.getState(getPositions = True, getVelocities = True, enforcePeriodicBox = True)
                reporter_position = state.getPositions()/unit.nanometers
                reporter_velocity = state.getVelocities()/unit.nanometers*unit.picosecond
                if self.a_indicator(reporter_position) or self.b_indicator(reporter_position):
                    self.b_reporter_confs[self.current_iteration] = np.array([reporter_position]*self.n_reporters)
                    self.b_reporter_confs.flush()
                    self.b_reporter_velos[self.current_iteration] = np.array([reporter_velocity]*self.n_reporters)
                    self.b_reporter_velos.flush()
                    return
                reporter_positions.append(reporter_position)
                reporter_velocities.append(reporter_velocity)

            for steps in range(n_steps):
                if np.sum(end_flags)/self.n_reporters > fraction:
                    break
                for n in range(self.n_reporters):
                    if end_flags[n] == 0:
                        self.context.setPositions(reporter_positions[n])
                        self.context.setVelocities(reporter_velocities[n])
                        self.context.getIntegrator().step(stride)
                        time += stride
                        state = self.context.getState(getPositions = True, getVelocities = True, enforcePeriodicBox = True)
                        reporter_position = state.getPositions()/unit.nanometers
                        reporter_velocity = state.getVelocities()/unit.nanometers*unit.picosecond
                        if self.b_indicator(reporter_position):
                            end_flags[n] = 1
                        elif self.a_indicator(reporter_position):
                            end_flags[n] = 1
                            self.b_transit = True
                        reporter_positions[n] = reporter_position
                        reporter_velocities[n] = reporter_velocity
                        
            self.b_reporter_confs[self.current_iteration] = reporter_positions.copy()
            self.b_reporter_confs.flush()
            self.b_reporter_velos[self.current_iteration] = reporter_velocities.copy()
            self.b_reporter_velos.flush()
            if len(self.b_times) == 0:
                self.b_times.append(time)
            else:
                self.b_times.append(self.b_times[-1] + time)

    def compute_log_loss(self, a_targets, b_targets):
        """ Computes the mean-squared-log error between committor estimates and target committor values
        
        Args:
            a_targets, b_targets: The values of the committor target values expressed as q and 1-q, respectively
        Returns:
            loss: The symmetrized loss

        """
        a_input_confs = torch.tensor(np.memmap(self.data_path + "a_sample_confs.dat", dtype = 'float64', mode = 'r', shape = (self.current_iteration + 1, self.n_particles, 3))).to(self.device)
        a_input_velos = torch.tensor(np.memmap(self.data_path + "a_sample_velos.dat", dtype = 'float64', mode = 'r', shape = (self.current_iteration + 1, self.n_particles, 3))).to(self.device)
        
        b_input_confs = torch.tensor(np.memmap(self.data_path + "b_sample_confs.dat", dtype = 'float64', mode = 'r', shape = (self.current_iteration + 1, self.n_particles, 3))).to(self.device)
        b_input_velos = torch.tensor(np.memmap(self.data_path + "b_sample_velos.dat", dtype = 'float64', mode = 'r', shape = (self.current_iteration + 1, self.n_particles, 3))).to(self.device)
        
        input_confs = torch.cat((a_input_confs, b_input_confs))
        input_velos = torch.cat((a_input_velos, b_input_velos))
        
        a_predictions = torch.sigmoid(self.net(input_confs, input_velos)).squeeze()
        b_predictions = torch.sigmoid(-self.net(input_confs, input_velos)).squeeze()
        a_indices = torch.where(a_targets != 0)
        b_indices = torch.where(b_targets != 0)
        a_predictions, a_targets = a_predictions[a_indices], a_targets[a_indices]
        b_predictions, b_targets = b_predictions[b_indices], b_targets[b_indices]

        a_indices = torch.where(a_targets != 1)
        b_indices = torch.where(b_targets != 1)
        a_predictions, a_targets = a_predictions[a_indices], a_targets[a_indices]
        b_predictions, b_targets = b_predictions[b_indices], b_targets[b_indices]
    
        log_loss_a = (torch.mean(torch.square(torch.log(a_predictions) - torch.log(a_targets))))
        log_loss_b = (torch.mean(torch.square(torch.log(b_predictions) - torch.log(b_targets))))

        return log_loss_a + log_loss_b
        
        
    def update_committor(self, batch_size, n_updates = 1, n_optim_steps = 100):
        """Trains the neural-network representation of the committor and updates the net accordingly
        
        Args:
            batch_size: data batch size for neural network training
            n_updates: how many times the target commitor value (based on swarm endpoints) is recalculated
            n_optim_steps: How many steps of gradient descent are performed per target update

        """
        for j in range(n_updates):
            self.net.zero_grad()
            self.optimizer.zero_grad()
            a_committors = torch.tensor(np.mean(self.load_and_predict_committors("A", post_collection = True), axis = -1))
            b_committors = torch.tensor(np.mean(self.load_and_predict_committors("B", post_collection = True), axis = -1))
            a_target_committors, b_target_committors = torch.cat((a_committors, 1 - b_committors)), torch.cat((1-a_committors, b_committors))
            for k in range(n_optim_steps):
                permutation = torch.randperm((self.current_iteration+1)*2)
                for i in range(0, (self.current_iteration+1)*2, batch_size):
                    self.net.zero_grad()
                    self.optimizer.zero_grad()
                    indices = permutation[i:i+batch_size]
                    log_loss = self.compute_log_loss(a_target_committors.detach().to(self.device), b_target_committors.detach().to(self.device))
                    log_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 100000, error_if_nonfinite = True)
                    self.optimizer.step()
                    
    def compute_rate_estimate(self):
        """Computes and online rate estimate based on equilibrium flux and estimated committors.
            The online estimates are stored to evaluate convergence

        """
        a_escape_confs = torch.tensor(np.load(self.data_path + "A_boundary_confs.npy")).to(self.device)
        a_escape_velos = torch.tensor(np.load(self.data_path + "A_boundary_velos.npy")).to(self.device)
        b_escape_confs = torch.tensor(np.load(self.data_path + "B_boundary_confs.npy")).to(self.device)
        b_escape_velos = torch.tensor(np.load(self.data_path + "B_boundary_velos.npy")).to(self.device)
        a_committor_values = (torch.sigmoid(self.net(a_escape_confs, a_escape_velos)))[:self.n_boundary_samples].float().to(self.device)
        b_committor_values = (torch.sigmoid(-self.net(b_escape_confs, b_escape_velos)))[-self.n_boundary_samples:].float().to(self.device)
        a_rate_estimate = torch.mean(a_committor_values) * self.a_flux
        b_rate_estimate = torch.mean(b_committor_values) * self.b_flux   
        self.a_rate_estimates.append(a_rate_estimate.cpu().detach().numpy())
        self.b_rate_estimates.append(b_rate_estimate.cpu().detach().numpy())
        self.a_committors = a_committor_values.cpu().detach().numpy()
        self.b_committors = a_committor_values.cpu().detach().numpy()
        
    def step(self, n_reporter_steps, stride, fraction):
        """High-level method that increments the simulation state by drawing a sample,
            generating an associated swarm, updating the committor network,
            and computing an online rate estimate.
            
            Args:
                n_steps: (AKA tau) The maximum length of each trajectory in the swarm, in units of the stride
                stride: How often (in number of steps) basin membership for a swarm trajectory is checked
                fraction: Fraction of swarms that must re-enter a basin to trigger an early stopping criterion
        """
        if self.has_basin_samples:
            if self.a_transit_history[-1] == self.current_iteration or self.current_iteration == 0:
                self.begin_sample_chain("A")
            else:
                self.continue_sample_chain("A")

            if self.b_transit_history[-1] == self.current_iteration or self.current_iteration == 0:
                self.begin_sample_chain("B")
            else:
                self.continue_sample_chain("B")


            self.sample_swarm("A", n_reporter_steps, stride, fraction)
            self.sample_swarm("B", n_reporter_steps, stride, fraction)
            self.update_committor((self.current_iteration+1)*2)
            self.compute_rate_estimate()
            print(f"Step {self.current_iteration}")
            print(f"A to B Mean First Passage Time: {1/self.a_rate_estimates[-1]}")
            print(f"B to A Mean First Passage Time: {1/self.b_rate_estimates[-1]}")
            if self.a_transit:
                self.a_current_index += 1
                self.a_transit_history.append(self.current_iteration + 1)
                self.a_transit = False
            if self.b_transit:
                self.b_current_index += 1
                self.b_transit_history.append(self.current_iteration + 1)
                self.b_transit = False
            self.current_iteration += 1
        else:
            try: 
                np.load(self.data_path + "A_boundary_confs.npy")
            except:
                print("No equilibrium basin simulations have been run.")
                print("Sampling cannot begin without a boundary ensemble.")
                print("Please generate the ensemble with the sample_basins method.")
                raise SystemError
    
