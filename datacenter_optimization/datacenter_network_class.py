import cvxpy as cp
import numpy as np
import networkx as nx

from collections import defaultdict, deque
from mat4py import loadmat
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

class Network():
    def __init__(self,
                 data,
                 rx_scale:float = 1.0,
                 pq_scale:float = 1.0,
                 slack_bus:int = 1):
        self.G = None  # Placeholder for network graph

        if isinstance(data, list):   # Assume given branch and bus csv file names
            bus_data_file = data[0]
            branch_data_file = data[1]
            data_bus = np.loadtxt(bus_data_file, delimiter=',')
            data_branch = np.loadtxt(branch_data_file, delimiter=',')      
            data = {
                'bus': data_bus,
                    # ... continue for all buses ..
                'branch': data_branch,
            }  
        if isinstance(data, dict):  
            self._process_data_dict(data, rx_scale, pq_scale, slack_bus)
        elif isinstance(data, str):
            if data.endswith('.mat'):
                mat_path = os.path.join("TestCase33.mat")
                case_name = "TestCase"
                data = loadmat(mat_path)[case_name]

                # Sensitivity matrices: dV = R*dP + X*dQ
                R = np.asarray(data["R"], dtype=np.float32)
                X = np.asarray(data["X"], dtype=np.float32)

                # Reference operating point used by the linearization
                V_ref = np.asarray(data["V_ref"], dtype=np.float32)
                P0    = np.asarray(data["P_ref"], dtype=np.float32)
                Q0    = np.asarray(data["Q_ref"], dtype=np.float32)
                ones  = np.asarray(data["ones"], dtype=np.float32)

                # Rescale matrices and references for numerical stability
                R *= rx_scale
                X *= rx_scale
                P0 *= 10.0
                Q0 *= 10.0

                # self.r_map = 
                # self.x_map = 
                # self.edges = edges
                # self.n_buses = R.shape[0]
                # self.bus_ids = [int(b[0]) for b in data['bus']]  # bus numbers
                # self.idx_map = {bus: idx for idx, bus in enumerate(self.bus_ids)}
                # self.slack_idx = self.idx_map[15]  # or whichever bus is Slack
        
            else:
                raise ValueError("Given data is an unrecognized file type! %s" % data)
        else:  
            raise ValueError("Given data is an unrecognized format!")
    
    def _process_data_dict(self, data, rx_scale=1., pq_scale=1., slack_bus=1):
        # Capacitors & regulators
        # self.caps = {int(bus): cap for bus, cap in data['caps']}  #TODO: Currently, not implemented
        # self.regs = {int(bus): ratio for bus, ratio in data['regs']}

        # # Generator (slack & DER) data
        # self.gens = {int(row[0]): row[1:] for row in data['gen']}

        # Now Pd, Qd, edges, r_map/x_map, inverter_buses_idx are ready for optimization
        self.n_buses = len(data['bus'])  # e.g. 123 + added buses
        self.bus_ids = [int(b[0]) for b in data['bus']]  # bus numbers
        self.idx_map = {bus: idx for idx, bus in enumerate(self.bus_ids)}
        self.slack_idx = self.idx_map[slack_bus]  # or whichever bus is Slack

        # Build edges and line parameters
        edges = []
        r_map = {}
        x_map = {}
        for br in data['branch']:
            if br[10]:  # Check if line is in
                i = self.idx_map[int(br[0])]
                j = self.idx_map[int(br[1])]
                r_map[(i, j)] = br[2] * rx_scale
                x_map[(i, j)] = br[3] * rx_scale
                edges.append((i, j))
            else: # Line is out. Ignore it
                pass
        self.r_map = r_map
        self.x_map = x_map
        self.edges = edges

        self.Pd = np.zeros(self.n_buses)
        self.Qd = np.zeros(self.n_buses)
        for b in data['bus']:
            idx = self.idx_map[int(b[0])]
            self.Pd[idx] = b[2] * pq_scale
            self.Qd[idx] = b[3] * pq_scale

        # === Build tree structure ===
        self.children = defaultdict(list)
        self.parent = {}
        for i, j in edges:
            self.children[i].append(j)
            self.parent[j] = i

    def calculate_voltage_sqrt(self, v0, Pinj_full, Qinj_full):
        # === Step 1: Compute branch flows from injections ===
        P_flow = {}
        Q_flow = {}

        def compute_branch_flows(i):  #TODO: Assumes lossless powerflows?
            p_total = Pinj_full[i]
            q_total = Qinj_full[i]
            for j in self.children[i]:
                p_child, q_child = compute_branch_flows(j)
                p_total += p_child
                q_total += q_child
                P_flow[(i, j)] = p_child
                Q_flow[(i, j)] = q_child 
            return p_total, q_total

        compute_branch_flows(self.slack_idx)

        # === Step 2: Compute voltages using full DistFlow quadratic ===
        V = np.zeros(self.n_buses)
        V[self.slack_idx] = v0  # initialize slack bus voltage

        queue = deque([self.slack_idx])
        while queue:
            i = queue.popleft()
            for j in self.children[i]:
                r = self.r_map[(i, j)]
                x = self.x_map[(i, j)]
                Pij = P_flow[(i, j)]
                Qij = Q_flow[(i, j)]
                vi = V[i]

                # Coefficients of the quadratic: vj^2 + b*vj + c = 0
                b = 2 * (r * Pij + x * Qij) - vi
                c = (r ** 2 + x ** 2) * (Pij ** 2 + Qij ** 2)
                disc = b ** 2 + 4 * c
                discriminant = np.sqrt(disc)
                if disc < 0:
                    raise ValueError(f"Negative discriminant at bus {j}, cannot compute voltage.")
                vj = (-b + discriminant) / 2  # take positive root
                V[j] = vj
                queue.append(j)
        return V

from generate_rack_load import load_example_traces
from generate_rack_load import resample
import os

class DataCenters():
    def __init__(self, 
                 bus_idx_map: dict,  # Dictionary mapping from bus name to bus idx
                 dc_bus_loc: list,   # List of buses indicating data center bus locations
                 dc_stor_loc: list,  # List of lists indicating storage units associated with each data center bus 
                 dc_power_rating, # Power rating (in p.u.) of the data center or each individual UPS
                 ups_poi_power_rating, # POI inv power rating
                 ups_pdu_connection,  # Connections between UPS and PDU devices
                 dc_backup_time, # Duration of the time until the backup generator can be brought online for each UPS
                 load_pf: float, # Power factor of the data center load
                 stor_power_rating = None,  # Rated power of the storage in p.u./s
                 stor_duration = None, # Rated duration of the storage in seconds
                ):
        self.bus_idxs = [bus_idx_map[bus] for bus in dc_bus_loc]
        self.dc_power_rating = dc_power_rating
        self.dc_load_scale = dc_power_rating  # Scale maximum load value to this value if using generate_load_profile
        self.n_dc = len(self.bus_idxs)
        self.load_pf = load_pf
        self.pf_pq_factor = np.tan(np.acos(load_pf))
        self.dc_backup_time = dc_backup_time
        # self.eff = # TODO: Add efficiencies for storage, data center inverter, PDU to rack?

        self.poi_inv_power_rating = ups_poi_power_rating

        self.stor_loc = [bus_idx_map[bus] for bus in dc_stor_loc]
        n_stor = len(self.stor_loc)
        #TODO: Update this calculation to parse over each bus to determine the total maximum load (all PDUs at the bus) and the number of storage units at that data center.
        # Potentially update data center and storage location to be a list of lists associated with each data center.
        if stor_power_rating is None:
            stor_power_rating = (self.dc_power_rating / 0.9) * 1 / (n_stor - 1)
        if len(stor_power_rating) < n_stor:
            stor_power_rating = stor_power_rating[0] * np.ones(n_stor)
        if stor_duration is None:
            stor_duration = dc_backup_time

        self.stor = Storage(bus_idx_map, dc_stor_loc, np.array(stor_duration) * np.array(stor_power_rating), stor_power_rating)

        self.A = np.zeros((self.n_dc, self.stor.n_stor)) # UPS -> PDU Biadjacency matrix
        for k, conn in enumerate(ups_pdu_connection):  # UPS storage k
            for j in conn:  # PDU server cluster j
                self.A[j,k] = 1
        
    
    def generate_load_profile(self, T, data_path, dt, dt_orig=0.1, start_idx=0, max_shift=1, pdu_n_racks=None, rack_gpus=8, seed=1, noise_frac=0.01):  #TODO: Update this function to take more of the set parameters or add a seperate data pruning function with this
        self.dc_Pinj_MW_t = np.zeros((T, self.n_dc))
        self.dc_Qinj_MVAR_t = np.zeros((T, self.n_dc))

        # Import actual GPU training data
        data_files = os.listdir(data_path)
        data_files = sorted(data_files)[0:1]
        print(data_files)
        data = load_example_traces(data_files, data_path)
        data_arr = np.array([l[:min(len(l) for l in data)] for l in data])
        print("data_arr.shape", data_arr.shape)
        dflt_dc_Pinj_MW_t = data_arr
        dflt_dc_Qinj_MVAR_t = self.pf_pq_factor * dflt_dc_Pinj_MW_t
        

        selected_traces = [0]
        selected_dflt_dc_Pinj_MW_t = dflt_dc_Pinj_MW_t[selected_traces]
        selected_dflt_dc_Qinj_MVAR_t = dflt_dc_Qinj_MVAR_t[selected_traces]

        # Select and resample data
        selected_data = [data_arr[i] for i in selected_traces]
        from generate_rack_load import generate_training_pdu_trace
        rng = np.random.default_rng(seed=seed)
        if pdu_n_racks is None:
            pdu_n_racks = [1]*self.n_dc

        # built gpu_trace_map
        gpu_trace_maps = {}
        for i, pdu_i_n_racks in enumerate(pdu_n_racks):
            gpu_trace_map = []
            for k in range(pdu_i_n_racks):
                gpu_trace_map += [rng.integers(len(selected_traces))]*rack_gpus
            gpu_trace_maps[i] = gpu_trace_map
        #gpu_trace_map = [0]*rack_gpus + [1]*rack_gpus + [2]*rack_gpus + [3]*rack_gpus

        for i_dc in range(self.n_dc):
            pdu_power_trace = generate_training_pdu_trace(selected_data, num_gpus=8*pdu_n_racks[i_dc], duration_s=T, start_idx=start_idx, noise_frac=noise_frac, max_shift=max_shift, gpu_trace_map=gpu_trace_maps[i])
            pdu_power_trace_vals = pdu_power_trace["P_IT_W"].values.astype(float)[:, None]
            self.dc_Pinj_MW_t[:, i_dc:i_dc+1] = pdu_power_trace_vals
        max_pdu_load = np.max(self.dc_Pinj_MW_t)
        for i_dc in range(self.n_dc):  # Normalize to desired load peak magnitude
            self.dc_Pinj_MW_t[:, i_dc:i_dc+1] = self.dc_Pinj_MW_t[:, i_dc:i_dc+1] * self.dc_load_scale[i_dc] / max_pdu_load
        print("self.dc_Pinj_MW_t.shape", self.dc_Pinj_MW_t.shape)

        # # Compute statistical values of the data center load and plot it
        dc_mean_load = np.sum(np.mean(self.dc_Pinj_MW_t, axis=0))  #TODO: Make this depend on which data center each rack is a part of
        dc_max_mean_load = self.n_dc * np.max(np.mean(self.dc_Pinj_MW_t, axis=0))
        dc_max_load = self.n_dc * np.max(self.dc_Pinj_MW_t)
        self.pdu_rated_load = dc_max_load
        print("Average Total Data Center Load:", dc_mean_load)
        print("Max Possible Data Center Load:", dc_max_load)

        self.dc_Qinj_MVAR_t = self.pf_pq_factor * self.dc_Pinj_MW_t  # constant power factor

        return self.dc_Pinj_MW_t, self.dc_Qinj_MVAR_t  #TODO: Make this purely external variables?

class Storage:
    def __init__(self,
                 bus_idx_map,
                 stor_bus_loc,
                 stor_energy_rating,
                 stor_power_rating,
                 stor_min_energy=0):
        # Energy storage specs 
        self.bus_idxs = [bus_idx_map[bus] for bus in stor_bus_loc]

        self.n_stor = len(self.bus_idxs)
        self.energy_rating = stor_energy_rating
        self.e_upper_lim = stor_energy_rating
        self.e_lower_lim = stor_min_energy if not isinstance(stor_min_energy, (int, float)) else stor_min_energy * np.ones(self.n_stor)   # Set a minimum SoC value for the storage units

        self.p_upper_lim = stor_power_rating  # TODO: Assuming box constraints for now in dispatch
        self.p_lower_lim = stor_power_rating

# class DataCenterStorageDispatchProgram:
#     def __init__(self, 
#                  network: Network,
#                  datacenter: DataCenters,
#                  time_step_seconds: float = 1.0,
#                  horizon_T: int = 10,
#                  min_pf: float = 0.95,
#                  distflow: bool = True):
#         """
#         Initializes the Data Center Storage Dispatch Program.
        
#         Args:
#             storage_power_rating: Max power in/out (e.g., MW).
#             storage_duration: Duration at max power (e.g., Hours).
#             min_soc: Minimum State of Charge (0.0 to 1.0).
#             max_soc: Maximum State of Charge (0.0 to 1.0).
#             efficiency: Round-trip efficiency (0 to 1). 
#                         (Applied as sqrt(eff) to both charge and discharge).
#             time_step_hours: The duration of a single time step in the horizon.
#         """
#         # Base Parameters
#         self.dt = time_step_seconds

#         # System Parameters
#         self.network = network
#         self.n_buses = network.n_buses  #TODO: Make all of these calls individually throughout? No need to copy them here in case there are updates?
#         self.n_lines = len(network.edges)
#         self.n_stor = datacenter.stor.n_stor
#         self.n_dc = datacenter.n_dc
#         self.stor = datacenter.stor
#         self.datacenter = datacenter
#         self.min_pf = min_pf

#         if self.n_stor < 1 or self.n_dc < 1:
#             raise ValueError("System must have at least one datacenter and one storage unit.")  #TODO: Make this not a requirement by optionally creating these CVXPY parameters
        
#         # Data Center Storage Limits
#         inv_P_rating = datacenter.poi_inv_power_rating  # datacenter.dc_power_rating  #TODO: Make a seperate inverter POI interface class or rating? Need interface from UPS to INV? 
#         inv_Q_rating = np.array(inv_P_rating) * np.tan(np.acos(self.min_pf)) #np.zeros(self.stor.n_stor)  #TODO: Make this a variable limit based on P
#         self.P_min = [-inv_P_rating[i] for i in range(self.stor.n_stor)]  # TODO: Make this independant of the data centers so we can add storage anywhere?
#         self.P_max = [+inv_P_rating[i] for i in range(self.stor.n_stor)]
#         self.Q_min = [-inv_Q_rating[i] for i in range(self.stor.n_stor)] #[-0.3 for i in inverter_buses]
#         self.Q_max = [+inv_Q_rating[i] for i in range(self.stor.n_stor)] #[+0.3 for i in inverter_buses]

#         # Derived Parameters (calculated via helper)
#         self.energy_capacity = 0.0
#         self.one_way_eff = 0.0


#         # Data Center Parameters
#         self.backup_T = int(np.ceil(datacenter.dc_backup_time / self.dt))
#         self.A = datacenter.A
#         self.dc_bus_idxs = datacenter.bus_idxs
#         self.stor_bus_idxs = datacenter.stor.bus_idxs
#         self.dc_at_bus = {i: np.where(np.array(self.datacenter.bus_idxs) == i) for i in range(self.n_buses)}  # TODO: Move to respective classes
#         self.stor_at_bus = {i: np.where(np.array(self.stor_bus_idxs) == i) for i in range(self.n_buses)}

#         inv_P_rating = np.array(datacenter.dc_power_rating) if not isinstance(datacenter.dc_power_rating, float) else datacenter.dc_power_rating * np.ones(self.n_dc)
#         inv_Q_rating = np.array(datacenter.dc_power_rating) if not isinstance(datacenter.dc_power_rating, float) else datacenter.dc_power_rating * np.ones(self.n_dc)
#         self.Pdc_min = [-np.sum(inv_P_rating[np.array(self.dc_at_bus[i]).flatten()]) for i in range(self.n_buses)]
#         self.Pdc_max = [+np.sum(inv_P_rating[self.dc_at_bus[i]]) for i in range(self.n_buses)]
#         self.Qdc_min = [-np.sum(inv_Q_rating[self.dc_at_bus[i]]) for i in range(self.n_buses)]
#         self.Qdc_max = [+np.sum(inv_Q_rating[self.dc_at_bus[i]]) for i in range(self.n_buses)]

#         # Program Paramters
#         self.horizon_T = horizon_T
#         self.distflow = distflow

#         # Optimization Artifacts (Placeholders)
#         self.prob = None
#         self.vars = {}
#         self.params = {}

#     def build_optimization_problem(self):
#         """
#         Constructs the CVXPY optimization problem for the system.
#         This allows the problem structure to be compiled once and reused.
#         """
#         # 1. Define Variables
#         self.vars = {}

#         if self.distflow:
#             V = cp.Variable((self.horizon_T, self.n_buses))  # squared voltages
#             P = {(i, j): cp.Variable(self.horizon_T) for (i, j) in self.network.edges}   # Edge power flows
#             Q = {(i, j): cp.Variable(self.horizon_T) for (i, j) in self.network.edges}
#             #l = {(i, j): cp.Variable(self.horizon_T) for (i, j) in edges}
#             Pinj = cp.Variable((self.horizon_T, self.n_buses))  # Node power injections
#             Qinj = cp.Variable((self.horizon_T, self.n_buses))
#             Pslack = cp.Variable((self.horizon_T, 1))
#             Qslack = cp.Variable((self.horizon_T, 1))

#             self.vars['V'] = V
#             self.vars['P'] = P
#             self.vars['Q'] = Q
#             self.vars['Pinj'] = Pinj
#             self.vars['Qinj'] = Qinj
#             self.vars['Pslack'] = Pslack
#             self.vars['Qslack'] = Qslack

#         e = cp.Variable((self.horizon_T, self.n_stor)) 
#         uP = cp.Variable((self.horizon_T, self.n_stor))
#         uQ = cp.Variable((self.horizon_T, self.n_stor))
#         self.vars['e'], self.vars['uP'], self.vars['uQ'] = e, uP, uQ

#         # E = cp.Variable((self.horizon_T, n_stor, n_dc, n_stor), pos=True)  # [time, k contingency, j load, k storage]  # Used SCIPY backend for canonicalization which slow compilation time?
#         Ets = [[cp.Variable((self.n_dc, self.n_stor), pos=True) for k in range(self.n_stor)] for t in range(self.horizon_T)]  # Nested list of time indexed E matrices. Ets[t][i_stor] = E
#         self.vars['Ets'] = Ets

#         # Ak = cp.Parameter((n_stor, n_dc, n_stor))
#         Aks = [cp.Parameter((self.n_dc, self.n_stor)) for k in range(self.n_stor)]  # List of storage contingency A connectivity matrices
#         self.vars['Aks'] = Aks

#         # 2. Define Parameters (Values that change every MPC step)
#         self.params = {}
#         P_load_dc_pred = cp.Parameter((self.horizon_T+self.backup_T, self.n_dc), name="P_Load_Forecast")
#         Q_load_dc_pred = cp.Parameter((self.horizon_T+self.backup_T, self.n_dc), name="Q_Load_Forecast")
#         self.params['P_load_dc_pred'], self.params['Q_load_dc_pred'] = P_load_dc_pred, Q_load_dc_pred

#         if self.distflow:
#             P_load_bus_pred = cp.Parameter((self.horizon_T, self.n_buses), name="P_Load_Forecast")
#             Q_load_bus_pred = cp.Parameter((self.horizon_T, self.n_buses), name="Q_Load_Forecast")
#             self.params['P_load_bus_pred'], self.params['Q_load_bus_pred'] = P_load_bus_pred, Q_load_bus_pred

#         # P_load_dc_pred = np.zeros((self.horizon_T+self.backup_T, self.n_dc))  #TESTING WITH SET VALUES
#         # Q_load_dc_pred = np.zeros((self.horizon_T+self.backup_T, self.n_dc))
#         # P_load_bus_pred = np.zeros((self.horizon_T+self.backup_T, self.n_buses))
#         # Q_load_bus_pred = np.zeros((self.horizon_T+self.backup_T, self.n_buses))

#         # The initial energy state of the storage at t=0
#         stor_init_energy = cp.Parameter(self.n_stor, name="Stor_Init_Energy")
#         self.params['stor_init_energy'] = stor_init_energy

#         # 3. Define Constraints
#         constraints = []
    
#         # ---------------------------------------------------------
#         #  DEFINE CONSTRAINTS BELOW
#         # ---------------------------------------------------------

#         ## System Voltages 
#         if self.distflow:
#             # Slack bus voltage fixed at 1.0²
#             constraints += [V[:, self.network.slack_idx] == 1.0]  #TODO: Set this to v0 parameter

#             # (SOCP/Lin)DistFlow constraints
#             for t in range(self.horizon_T):
#                 for (i, j) in self.network.edges:
#                     r, x = self.network.r_map[(i, j)], self.network.x_map[(i, j)]
#                     # constraints += [
#                     #     V[t, j] == V[t, i] - 2*(r*P[(i, j)][t] + x*Q[(i, j)][t]) + (r**2 + x**2)*l[(i, j)][t],
#                     #     cp.SOC(l[(i, j)][t], cp.hstack([P[(i, j)][t], Q[(i, j)][t]]))
#                     # ]
#                     constraints += [  # LinDistFlow??? https://nlaws.github.io/LinDistFlow/dev/math/
#                     V[t, j] == V[t, i] - 2*(r*P[(i, j)][t] + x*Q[(i, j)][t]),
#                     ]

#             # Nodal power balance
#             for t in range(self.horizon_T):
#                 for i in range(self.n_buses):
#                     inflow_Pt = cp.sum([P[(k, j)][t] for (k, j) in self.network.edges if j == i])
#                     outflow_Pt = cp.sum([P[(k, j)][t] for (k, j) in self.network.edges if k == i])
#                     inflow_Qt = cp.sum([Q[(k, j)][t] for (k, j) in self.network.edges if j == i])
#                     outflow_Qt = cp.sum([Q[(k, j)][t] for (k, j) in self.network.edges if k == i])

#                     constraints += [
#                         Pinj[t, i] - P_load_bus_pred[t, i] + inflow_Pt == outflow_Pt,
#                         Qinj[t, i] - Q_load_bus_pred[t, i] + inflow_Qt == outflow_Qt,
#                         #V[i] <= 1.1,
#                         #V[i] >= 0.9
#                     ]

#             for t in range(self.horizon_T):
#                 for i in range(self.n_buses):
#                     dc_at_bus_idx = self.dc_at_bus[i][0] # unwraps the tuple, likely unnecessary
#                     stor_at_bus_idx = self.stor_at_bus[i] # unwraps the tuple

#                     dc_Pload = cp.sum(P_load_dc_pred[t, dc_at_bus_idx]) if len(dc_at_bus_idx) > 0 else 0
#                     dc_Qload = cp.sum(Q_load_dc_pred[t, dc_at_bus_idx]) if len(dc_at_bus_idx) > 0 else 0
                    
#                     stor_Pinj = cp.sum(uP[t, stor_at_bus_idx]) if len(stor_at_bus_idx) > 0 else 0
#                     stor_Qinj = cp.sum(uQ[t, stor_at_bus_idx]) if len(stor_at_bus_idx) > 0 else 0
                        
#                     slack_Pinj = 0 if i != self.network.slack_idx else Pslack[t]
#                     slack_Qinj = 0 if i != self.network.slack_idx else Qslack[t]
#                     constraints += [
#                         Pinj[t, i] == stor_Pinj - dc_Pload + slack_Pinj,
#                         Qinj[t, i] == stor_Qinj - dc_Qload + slack_Qinj
#                     ]
            
#             # Storage power injection box constraints
#             for t in range(self.horizon_T):  
#                 for i in range(self.n_stor):
#                     constraints += [
#                         self.P_min[i] <= uP[t, i], uP[t, i] <= self.P_max[i],  # Storage power injection box constraints
#                         self.Q_min[i] <= uQ[t, i], uQ[t, i] <= self.Q_max[i]
#                     ]

#         ###TODO: Update this to be convex?
#         # # Inverter power injection box constraints and power factor constraints
#         # for t in range(self.horizon_T):
#         #     for i in range(self.n_buses):  # Treat each UPS as an individual data center # Need to link these using A or connectivity for individual level PF regulation
#         #         dc_at_bus_idx = self.dc_at_bus[i]
#         #         stor_at_bus_idx = self.stor_at_bus[i] 
                
#         #         Pdc = P_load_dc_pred[t, dc_at_bus_idx]
#         #         Qdc = Q_load_dc_pred[t, dc_at_bus_idx]

#         #         Pstor = uP[t, stor_at_bus_idx]
#         #         Qstor = uQ[t, stor_at_bus_idx]

#         #         Pinj_dc = cp.sum(Pdc) + cp.sum(Pstor) 
#         #         Qinj_dc = cp.sum(Qdc) + cp.sum(Qstor)

#         #         constraints += [
#         #             self.Pdc_min[i] <= Pinj_dc, Pinj_dc <= self.Pdc_max[i],
#         #             self.Qdc_min[i] <= Qinj_dc, Qinj_dc <= self.Qdc_max[i]
#         #         ]

#         #         constraints += [
#         #             Pinj_dc >= 0,
#         #             cp.abs(Qinj_dc) <=  Pinj_dc * np.tan(np.acos(self.min_pf)) #TODO: Find a way to encode the PQ mag and PF constraint convexly
#         #         #Qinj_dc >= -cp.abs(Pinj_dc * np.tan(np.acos(self.min_pf)))  #TODO: IS THIS CONVEX???? NO right?
#         #         ]

#         #         # constraints += [  # Magnitude constraint
#         #         #     # Euclidean norm of [P, Q] <= S_rated
#         #         #     cp.norm(cp.vstack([uP[t, i], uQ[t, i]]), 2) <= S_rated
#         #         # ]

#         # # Energy storage power & energy constraints
#         constraints += [e[0, :] == stor_init_energy]  # Initialize storage energy values

#         for t in range(1,self.horizon_T):
#             constraints += [e[t, :] <= self.stor.e_upper_lim]
#             constraints += [self.stor.e_lower_lim <= e[t, :]]
#         for t in range(0,self.horizon_T-1):
#             constraints += [e[t+1, :] == e[t, :] - uP[t, :] * self.dt]
#         # Limit the energy dispatched in the last time step
#         constraints += [uP[-1] <= e[-1]]    # TODO: Plus e_lower_lim????
#         constraints += [uP[-1] >= -(self.stor.e_upper_lim - e[-1])]          

#         #"""
#         # W/o SCIPY backend
#         for k in range(self.n_stor):  
#             Ak_val = np.array(self.datacenter.A)  # Copy A connectivity matrix
#             #Ak_val[:, k] = 0  # Modify to add in contingency connection cases
#             Aks[k].value = Ak_val

#         for t in range(self.horizon_T):  # time steps
#             for k in range(self.n_stor):  # contingencies
#                 # Etk = E[t, k, :, :]  # When using SCIPY backend and 4 dimension E variable
#                 Etk = Ets[t][k]  # When using nested list for E variable storage
#                 #Atk = Ak[k] # When using 3 dim Ak variable
#                 Atk = Aks[k]  # When using list for A contingency variables
#                 constraints += [  #TODO: constraint energy transfers to be positive?  A^k \ocirc E^k leaves unconstrained variable...
#                 e[t, :] >= cp.sum(cp.multiply(Atk, Etk), axis=0), # Energy transfer allocations are less than the current stored energy
#                 cp.sum(P_load_dc_pred[t:t+self.backup_T, :], axis=0)*self.dt == cp.sum(cp.multiply(Atk, Etk), axis=1)  # Predicted load energy is satisfied by allocated storage energy
#                 ]
#                 # for i in range(n_stor):
#                 #     # e[t, i] >= cp.sum(cp.multiply(Ak, E[t, k, :, :]), axis=0)  # Energy transfer allocations are less than the current stored energy
#                 #     e[t, i] >= cp.sum(E[t, k, :, :], axis=0)  # Energy transfer allocations are less than the current stored energy

#                 # for j in range(n_dc):  # loads
#                 #     np.sum(dc_Pinj_MW_t[t:t+T_backup, :]) == cp.sum(cp.multiply(Ak, E[t, k, :, :]), axis=1)  # Predicted load energy is satisfied by allocated storage energy
#         #"""

#         self.constraints = constraints
#         # ---------------------------------------------------------
#         #  DEFINE YOUR OBJECTIVE BELOW
#         # ---------------------------------------------------------
#         # Example Objective: Minimize total cost of Grid Import
#         # cost = sum(p_grid[t] * price[t])
        
#         # Simple Objective for Demo: Minimize Peak Grid Usage + Total Import Cost
#         # We use a slight penalty on p_grid^2 to smooth peaks
#         c = 1e-3 * np.array([[1, 1, 1]])  #TEST: Differing costs for storage charging

#         Pdc = cp.sum(P_load_dc_pred[0:self.horizon_T, :], axis=1) - cp.sum(uP, axis=1) 
#         #Pdc = Pinj[:, self.dc_bus_idxs[0]]  # TODO: Automatically set this to be the power at all the data centers? 
#         scale = Pdc.size  # Normalize by lenght of Pdc
        
#         #objective = cp.Minimize(cp.sum_squares(Pdc - cp.mean(Pdc)) / scale)  # Minimize deviation from a mean value

#         objective = cp.Minimize(cp.sum_squares(Pdc - cp.mean(Pdc)) / scale + 1e-3 * cp.sum_squares(e/self.stor.e_upper_lim - 0.9) / self.stor.n_stor)  # Minimize deviation from a mean value and try to maintain storage level

#         #objective = cp.Minimize(cp.sum_squares(V - 1.0))  # Minimize voltage deviation

#         #objective = cp.Minimize(cp.sum_squares(V - 1.0) + 0.1*cp.sum_squares(e/self.stor.e_upper_lim - 0.9))  # Droop against SoC away from 90%

#         # objective = cp.Minimize(cp.sum_squares(V - 1.0) + 0.1*cp.sum_squares(e/self.stor.e_upper_lim - 0.9) + self.dt*uP[-1])  # Weight backup storage discharge

#         #objective = cp.Minimize(cp.sum_squares(Qinj))

#         # 4. Compile Problem
#         self.prob = cp.Problem(objective, constraints)
        
#         print(f"Optimization problem built for horizon T={self.horizon_T}")

#     def solve_dispatch(self, 
#                        current_stor_energy: list,
#                        P_predicted_load_dc: list, 
#                        Q_predicted_load_dc: list, 
#                        P_predicted_load_bus: list = [], 
#                        Q_predicted_load_bus: list = [], 
#                        predicted_prices: list = [],
#                        ):
#         """
#         Updates parameters and solves the MPC optimization problem.

#         Args:
#             predicted_load: List of load values (MW) for the horizon.
#             predicted_prices: List of price values ($/MWh) for the horizon.
#             current_soc_percentage: Current state of charge (0.0 to 1.0).

#         Returns:
#             dict: The optimized schedule (arrays) and the optimal control for the next step.
#         """
#         if self.prob is None:
#             raise Exception("Problem not built. Call build_optimization_problem(T) first.")

#         # Input Validation
#         if (P_predicted_load_dc).shape != self.params['P_load_dc_pred'].shape or (Q_predicted_load_dc).shape != self.params['Q_load_dc_pred'].shape:
#             raise ValueError(f"Input data length must match Horizon and Dimenstion T,N={self.params['P_load_dc_pred'].shape[0]}")
#         #raise ValueError(f"Input price prediction has incorrect dimensions") if (predicted_prices).shape != self.params['predicted_prices'].shape

#         # 1. Update Parameters
#         self.params['P_load_dc_pred'].value = np.array(P_predicted_load_dc)
#         self.params['Q_load_dc_pred'].value = np.array(Q_predicted_load_dc)

#         if self.distflow:
#             if (P_predicted_load_bus).shape != self.params['P_load_bus_pred'].shape or (Q_predicted_load_bus).shape != self.params['Q_load_bus_pred'].shape:
#                 raise ValueError(f"Input bus load data length must match Horizon T,N={P_predicted_load_bus.shape} or Dimension")
#             self.params['P_load_bus_pred'].value = np.array(P_predicted_load_bus)
#             self.params['Q_load_bus_pred'].value = np.array(Q_predicted_load_bus)
#         # self.params['price_forecast'].value = np.array(predicted_prices)
        
#         #current_energy = current_soc_percentage * self.energy_capacity
#         self.params['stor_init_energy'].value = current_stor_energy

#         # 2. Solve
#         # 'ECOS' is usually good for SOCP/Linear, 'OSQP' for QP.
#         try:
#             self.prob.solve(solver=cp.ECOS) 
#         except cp.SolverError:
#             self.prob.solve() 
#             # self.prob.solve(solver=cp.SCS) # Fallback

#         if self.prob.status not in ["optimal", "optimal_inaccurate"]:
#             print(f"Warning: Problem status is {self.prob.status}")
#             return None

#         # 3. Extract Results
#         results = {
#             "status": self.prob.status,
#             "storPinj": self.vars['uP'].value,
#             "storQinj": self.vars['uQ'].value,
#             "storEnergy": self.vars['e'].value,
#         }

#         if self.distflow:
#             control = self.vars['uP'].value[0], self.vars['uQ'].value[0], self.vars['e'].value[1]
#         else:
#             control = self.vars['uP'].value[0], np.zeros(self.vars['uQ'][0].shape), self.vars['e'].value[1]
        
#         return results, *control
    


#### TEST Class with vectorization
import scipy.sparse as sp

class DataCenterStorageDispatchProgram:
    def __init__(self, 
                 network,
                 datacenter,
                 time_step_seconds: float = 1.0,
                 horizon_T: int = 10,
                 min_pf: float = 0.95,
                 distflow: bool = True):
        
        # --- Base Parameters ---
        self.dt = time_step_seconds
        self.horizon_T = horizon_T
        self.backup_T = int(np.ceil(datacenter.dc_backup_time / self.dt))
        self.distflow = distflow
        self.min_pf = min_pf
        
        # --- System Handles ---
        self.network = network
        self.datacenter = datacenter
        self.stor = datacenter.stor
        
        # --- Dimensions ---
        self.n_buses = network.n_buses
        self.n_edges = len(network.edges)
        self.n_stor = self.stor.n_stor
        self.n_dc = datacenter.n_dc
        
        # Calculate Backup Horizon Steps
        self.backup_steps = int(np.ceil(datacenter.dc_backup_time / self.dt))
        
        if self.n_stor < 1 or self.n_dc < 1:
            raise ValueError("System must have at least one datacenter and one storage unit.")

        # --- Pre-compute Vectorized Parameters ---
        self._build_connectivity_matrices()
        self._build_limit_vectors()
        self._build_contingency_masks()

        # --- Optimization Artifacts ---
        self.prob = None
        self.vars = {}
        self.params = {}

    def _build_connectivity_matrices(self):
        """Creates sparse matrices to handle summation and mapping without loops."""
        # 1. Network Incidence Matrix (Edges x Buses)
        row_idx, col_idx, data = [], [], []
        self.edge_r = np.zeros(self.n_edges)
        self.edge_x = np.zeros(self.n_edges)
        self.edge_from_idxs = []
        self.edge_to_idxs = []

        for k, (u, v) in enumerate(self.network.edges):
            row_idx.extend([k, k])
            col_idx.extend([u, v])
            data.extend([-1, 1]) 
            self.edge_r[k] = self.network.r_map[(u, v)]
            self.edge_x[k] = self.network.x_map[(u, v)]
            self.edge_from_idxs.append(u)
            self.edge_to_idxs.append(v)

        self.Incidence = sp.coo_matrix((data, (row_idx, col_idx)), shape=(self.n_edges, self.n_buses))
        
        # 2. Device Mappings
        # Map Storage -> Bus (n_buses x n_stor)
        s_rows, s_cols = [], []
        for k, bus_idx in enumerate(self.stor.bus_idxs):
            s_rows.append(bus_idx)
            s_cols.append(k)
        self.Map_Stor_Bus = sp.coo_matrix((np.ones(self.n_stor), (s_rows, s_cols)), shape=(self.n_buses, self.n_stor))

        # Map DC -> Bus (n_buses x n_dc)
        d_rows, d_cols = [], []
        for k, bus_idx in enumerate(self.datacenter.bus_idxs):
            d_rows.append(bus_idx)
            d_cols.append(k)
        self.Map_DC_Bus = sp.coo_matrix((np.ones(self.n_dc), (d_rows, d_cols)), shape=(self.n_buses, self.n_dc))

        # Bus_With_DC = np.eye(self.n_buses) #np.zeros(self.n_buses, dtype=int)
        self.Bus_With_DC = np.eye(self.n_buses)[list(set(self.datacenter.bus_idxs)), :]

    def _build_limit_vectors(self):
        """Pre-calculates static limit arrays."""
        inv_P_rating = self.datacenter.poi_inv_power_rating
        if np.isscalar(inv_P_rating):
            inv_P_rating = np.full(self.n_stor, inv_P_rating)
        else:
            inv_P_rating = np.array(inv_P_rating)

        inv_Q_rating = inv_P_rating * np.tan(np.acos(self.min_pf))

        self.P_min_vec = -inv_P_rating
        self.P_max_vec = inv_P_rating
        self.Q_min_vec = -inv_Q_rating
        self.Q_max_vec = inv_Q_rating
        
    def _build_contingency_masks(self):
        """
        Creates a 3D mask for N-1 contingencies.
        Dimensions: (Contingency_Case, Load_ID, Storage_ID) -> (C, L, S)
        """

        # Initialize with ones (Everything Available)
        # Shape: (N_stor, N_dc, N_stor)
        self.contingency_mask = np.ones((self.n_stor, self.n_dc, self.n_stor))
        
        # Apply N-1 Logic
        # For Contingency Case 'k' (index 0), Storage Unit 'k' (index 2) is DOWN (0.0).
        # This applies to ALL Loads (index 1).
        for k in range(self.n_stor):
            # Mask[Case k, All Loads, Unit k] = 0
            self.contingency_mask[k, :, :] = np.array(self.datacenter.A)
            self.contingency_mask[k, :, k] = 0.0

    def build_optimization_problem(self):
        """
        Constructs the CVXPY optimization problem using vectorized operations.
        """
        T = self.horizon_T
        
        # --- 1. Define Variables ---
        uP = cp.Variable((T, self.n_stor), name="uP")
        uQ = cp.Variable((T, self.n_stor), name="uQ")
        e = cp.Variable((T, self.n_stor), nonneg=True, name="e_stored")
        
        # Network Variables
        P_flow = cp.Variable((T, self.n_edges), name="P_flow")
        Q_flow = cp.Variable((T, self.n_edges), name="Q_flow")
        V = cp.Variable((T, self.n_buses), nonneg=True, name="V_sq")
        Pinj = cp.Variable((T, self.n_buses), name="Pinj") 
        Qinj = cp.Variable((T, self.n_buses), name="Qinj")
        Pslack = cp.Variable((T, 1), name="Pslack")
        Qslack = cp.Variable((T, 1), name="Qslack")

        # N-1 Allocation Variable: (Time, Contingency Case, Load ID, Storage Provider ID)
        # Dimensions: T, C, L, S
        E_alloc = cp.Variable((T, self.n_stor, self.n_dc, self.n_stor), nonneg=True, name="E_alloc")

        # Objective Data Center Mean Load Floating Variable
        target_mean_load = cp.Variable(1, name="Target_Mean_Load")

        # --- 2. Define Parameters ---
        # Immediate Load (for Power Flow) - Size T
        P_load_dc_step = cp.Parameter((T, self.n_dc), name="P_load_dc_step")
        Q_load_dc_step = cp.Parameter((T, self.n_dc), name="Q_load_dc_step")
        P_load_bus_step = cp.Parameter((T, self.n_buses), name="P_load_bus_step")
        Q_load_bus_step = cp.Parameter((T, self.n_buses), name="Q_load_bus_step")
        
        # Rolling Backup Energy Requirement - Size T (Pre-calculated in Python)
        Backup_Energy_Req = cp.Parameter((T, self.n_dc), name="Backup_Energy_Req")
        
        stor_init_energy = cp.Parameter(self.n_stor, name="init_energy")
        
        # Updated Parameter for the Contingency Mask (C, L, S)
        # Dimensions: (N_stor, N_dc, N_stor)
        Mask_Contingency = cp.Parameter((self.n_stor, self.n_dc, self.n_stor), name="Mask_Contingency", value=self.contingency_mask)

        # --- 3. Vectorized Constraints ---
        constraints = []

        # -- A. Storage Dynamics --
        constraints += [e[0, :] == stor_init_energy - uP[0, :] * self.dt]
        constraints += [e[1:, :] == e[:-1, :] - uP[1:, :] * self.dt]
        
        constraints += [e <= self.stor.e_upper_lim]
        constraints += [e >= self.stor.e_lower_lim]
        constraints += [uP <= self.P_max_vec, uP >= self.P_min_vec]
        constraints += [uQ <= self.Q_max_vec, uQ >= self.Q_min_vec]

        # -- B. Nodal Power Balance --
        if self.distflow:
            P_stor_at_bus = uP @ self.Map_Stor_Bus.T
            Q_stor_at_bus = uQ @ self.Map_Stor_Bus.T
            P_dc_at_bus = P_load_dc_step @ self.Map_DC_Bus.T
            Q_dc_at_bus = Q_load_dc_step @ self.Map_DC_Bus.T
            
            net_flow_out_P = P_flow @ self.Incidence
            net_flow_out_Q = Q_flow @ self.Incidence
            
            slack_mask = np.zeros(self.n_buses)
            slack_mask[self.network.slack_idx] = 1.0
            P_slack_inj = Pslack @ slack_mask.reshape(1, -1)
            Q_slack_inj = Qslack @ slack_mask.reshape(1, -1)
            
            constraints += [
                (P_stor_at_bus - P_dc_at_bus) + P_slack_inj - P_load_bus_step == net_flow_out_P,
                (Q_stor_at_bus - Q_dc_at_bus) + Q_slack_inj - Q_load_bus_step == net_flow_out_Q
            ]

            # -- C. DistFlow Voltage --
            V_from = V[:, self.edge_from_idxs]
            V_to   = V[:, self.edge_to_idxs]
            V_drop = 2 * (cp.multiply(P_flow, self.edge_r) + cp.multiply(Q_flow, self.edge_x))
            
            constraints += [V_to == V_from - V_drop]
            constraints += [V[:, self.network.slack_idx] == 1.0]

        # # -- D.1. Inverter Circular Apparent Power Limit (SOCP)
        # S_rated_vec = cp.Parameter(self.n_stor, name="S_rated", value=self.P_max_vec)  #TODO: Set this value to be the rating for each data center or storage unit. Unclear is we can do this at a UPS level without directly mapping energy flows from UPDs to PDUs to Loads

        # P_stor_at_bus = uP @ self.Map_Stor_Bus.T
        # Q_stor_at_bus = uQ @ self.Map_Stor_Bus.T
        # P_dc_at_bus = P_load_dc_step @ self.Map_DC_Bus.T
        # Q_dc_at_bus = Q_load_dc_step @ self.Map_DC_Bus.T

        # P_inv = (self.Bus_With_DC @ (P_dc_at_bus - P_stor_at_bus).T).T
        # Q_inv = (self.Bus_With_DC @ (Q_dc_at_bus - Q_stor_at_bus).T).T
        # constraints += [
        #     cp.square(P_inv) + cp.square(Q_inv) <= cp.square(cp.sum(S_rated_vec[np.newaxis, :]))  #TODO: For now using the sum of all powers and ratings assuming one data center with one POI...
        # ]

        # # # -- D.2. Bus Power Factor Constraint
        # pf_limit_slope = np.tan(np.arccos(self.min_pf))

        # P_bus_inj = (self.Bus_With_DC @ (P_load_bus_step + P_dc_at_bus - P_stor_at_bus).T).T
        # Q_bus_inj = (self.Bus_With_DC @ (Q_load_bus_step + Q_dc_at_bus - Q_stor_at_bus).T).T

        # P_dc_inj = cp.Variable([self.horizon_T, self.n_dc], name="P_dc_inj", nonneg=True)  # Define varaible that is constrained to be positive and is equal to the above?
        # constraints += [
        #     P_dc_inj == P_bus_inj,
        #     cp.abs(Q_bus_inj) <= P_dc_inj * pf_limit_slope + 1e-3  # Only works for positive signed bus injections (Data center only consumes power)
        # ]


        # -- D. N-1 Contingency & Rolling Energy Logic --
        
        # 1. Enforce Availability Mask (Explicit 3D Broadcast)
        # E_alloc: (T, C, L, S)
        # Mask:    (   C, L, S) -> Broadcast to (1, C, L, S)
        # Result:  (T, C, L, S)
        # This explicitly enforces the N-1 limits for every load, case, and time step.
        E_masked = cp.multiply(E_alloc, Mask_Contingency[np.newaxis, :, :, :])
        
        # 2. Demand Satisfaction (Equality)
        # Sum allocated energy over Providers (Axis 3)
        # Result: (T, C, L)
        E_sum_per_load = cp.sum(E_masked, axis=3) 
        
        # Backup_Energy_Req: (T, L). Broadcast to (T, 1, L) to match (T, C, L)
        constraints += [E_sum_per_load == Backup_Energy_Req[:, np.newaxis, :]]
        
        # 3. Storage Feasibility (Inequality)
        # Sum allocated energy over Loads (Axis 2)
        # Result: (T, C, S)
        E_sum_per_storage = cp.sum(E_masked, axis=2)
        
        # e: (T, S). Broadcast to (T, 1, S) to match (T, C, S)
        constraints += [E_sum_per_storage <= e[:, np.newaxis, :]]

        # --- 4. Objective ---
        P_total_dc = cp.sum(P_load_dc_step, axis=1) - cp.sum(uP, axis=1)

        objective = cp.Minimize(
            cp.sum_squares(P_total_dc - target_mean_load) / self.datacenter.n_dc + 
            5 * 1e-3 * cp.sum_squares(e / self.stor.e_upper_lim - 0.9) / self.stor.n_stor  #TEST: Heavier weight on SoC deviation. Can we penalize the maximum deviation (DoD) instead?
        )
        # objective = cp.Minimize(
        #     cp.sum_squares(P_total_dc - target_mean_load) / self.datacenter.n_dc
        # )

        if self.distflow:
            #objective = cp.Minimize(cp.sum_squares(V - 1.0))  # Minimize voltage deviation
            objective = cp.Minimize(cp.sum_squares(V - 1.0) + 0.1*cp.sum_squares(e/self.stor.e_upper_lim - 0.9))  # Droop against SoC away from 90%

        # --- 5. Compile ---
        self.prob = cp.Problem(objective, constraints)
        
        self.vars = {
            'uP': uP, 'uQ': uQ, 'e': e, 
            'E_alloc': E_alloc
        }
        if self.distflow:
            self.vars.update({'V': V, 'Pslack': Pslack})

        self.params = {
            'P_load_dc_step': P_load_dc_step,
            'Q_load_dc_step': Q_load_dc_step,
            'P_load_bus_step': P_load_bus_step,
            'Q_load_bus_step': Q_load_bus_step,
            'Backup_Energy_Req': Backup_Energy_Req,
            'stor_init_energy': stor_init_energy,
            'Mask_Contingency': Mask_Contingency
        }
        
        print(f"Optimization problem built for horizon T={T} with N-1 checks.")

    def solve_dispatch(self, 
                       current_stor_energy: list,
                       P_predicted_load_dc: list, # T + T_backup
                       Q_predicted_load_dc: list, # T + T_backup
                       P_predicted_load_bus: list = None, 
                       Q_predicted_load_bus: list = None,
                       predicted_prices: list = None):
        
        if self.prob is None:
            raise Exception("Run build_optimization_problem() first.")

        # --- Data Validation & Pre-processing ---
        T = self.horizon_T
        T_total = P_predicted_load_dc.shape[0]
        
        if T_total < T + self.backup_steps:
             raise ValueError(f"Input Load must be length T({T}) + Backup({self.backup_steps}). Got {T_total}.")

        # 1. Slice Immediate Load (for Power Flow constraints)
        P_dc_step_val = P_predicted_load_dc[:T, :]
        Q_dc_step_val = Q_predicted_load_dc[:T, :]
        
        # 2. Calculate Rolling Backup Energy Requirement (MWh)
        Backup_Req_Val = np.zeros((T, self.n_dc))
        
        for t in range(T):
            # Sum load window [t : t + backup_steps]
            load_window = P_predicted_load_dc[t : t + self.backup_steps, :]
            Backup_Req_Val[t, :] = np.sum(load_window, axis=0) * self.dt

        # 3. Handle Bus Loads
        if P_predicted_load_bus is None:
            P_bus_val = np.zeros((T, self.n_buses))
            Q_bus_val = np.zeros((T, self.n_buses))
        else:
            P_bus_val = P_predicted_load_bus[:T, :]
            Q_bus_val = Q_predicted_load_bus[:T, :]

        # --- Assign Parameters ---
        self.params['P_load_dc_step'].value = P_dc_step_val
        self.params['Q_load_dc_step'].value = Q_dc_step_val
        self.params['P_load_bus_step'].value = P_bus_val
        self.params['Q_load_bus_step'].value = Q_bus_val
        self.params['Backup_Energy_Req'].value = Backup_Req_Val
        self.params['stor_init_energy'].value = np.array(current_stor_energy)
        self.params['Mask_Contingency'].value = self.contingency_mask

        # --- Solve ---
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except:
            try:
                self.prob.solve(solver=cp.ECOS)
            except:
                self.prob.solve(solver=cp.SCS)

        if self.prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Problem status is {self.prob.status}")
            return None, None, None, None

        # Extract Results
        results = {
            "status": self.prob.status,
            "storPinj": self.vars['uP'].value,
            "storQinj": self.vars['uQ'].value,
            "storEnergy": self.vars['e'].value,
        }

        if self.distflow:
            control = self.vars['uP'].value[0], self.vars['uQ'].value[0], self.vars['e'].value[1]
        else:
            control = self.vars['uP'].value[0], np.zeros(self.vars['uQ'][0].shape), self.vars['e'].value[1]
        
        return results, *control

