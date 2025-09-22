import gurobipy as gp
from gurobipy import GRB	
import numpy as np
import scipy.sparse as sp	
import control as ct
from types import SimpleNamespace

"""State Space SARX Identification
Hybrid system identification for state-space models with switching autoregressive exogenous inputs.
https://tinyurl.com/yurxkmcp

Decision Variables: omega

"""
class StateSpaceSARXID:
    def __init__(self, U: np.ndarray, Y: np.ndarray, params: dict = {}):
        self.grb_model = gp.Model("StateSpaceSARXID")
        self.U = U
        self.Y = Y
        self.params = params
        
        # Validate if sysid parameters are internally consistent
        assert self.params.min_modes >= 0 and self.params.max_modes >= self.params.min_modes, "Invalid mode range."
        assert self.params.min_modes <= self.Y.size, "Minimum number of modes is larger than number of measurements."
        self.num_modes = self.params.max_modes
        
        
        
        self.ss_models = {}
        self.submodel_parameter_vectors = {}

        self._init_decision_variables()
        self.set_sysid_objective()
        # self.set_constraints()
        
        # self._initialize_submodel_parameter_vectors()
        # self._initialize_identity_ss_models()


    def _init_decision_variables(self):
        self.theta = self.grb_model.addMVar((self.params.num_y_regressors + self.params.num_u_regressors + 1, self.num_modes), name="theta")
        self.Chi = self.grb_model.addMVar((self.params.N, self.num_modes), name="Chi", vtype=GRB.BINARY)
        pass
    
    # def _initialize_submodel_parameter_vectors(self):
    #     for mode in range(self.params.num_modes):
    #         num_params = self.params.num_y_regressors + self.params.num_u_regressors
    #         self.submodel_parameter_vectors[mode] = np.zeros((num_params, 1))
            
    # def _initialize_identity_ss_models(self):
    #     """Initializes identity state-space models for each mode in the hybrid system.
    #     """
    #     for mode in range(self.params.num_modes):
    #         A = np.eye(self.params.num_y_regressors)
    #         B = np.zeros((self.params.num_y_regressors, self.params.num_u_regressors))
    #         C = np.eye(self.params.num_y_regressors)
    #         D = np.zeros((self.params.num_y_regressors, self.params.num_u_regressors))

    #         ss_model = ct.ss(A, B, C, D)
            
    #         self.ss_models[mode] = ss_model

    def set_sysid_objective(self):
        self.grb_model.setObjective(0, GRB.MINIMIZE)
        
        obj = 0
    
        for k in range(self.params.n_bar, self.params.N): # Iterate over timesteps
            phi_k = self.get_extended_regression_vector(k)
            for i in range(self.num_modes): # Iterate over modes
                pred_error = self.Y[k, :] - phi_k.T @ self.theta[:, i]
                # obj += pred_error @ pred_error * self.Chi[k, i] # Squared L2 norm
                obj += pred_error * self.Chi[k, i]

        self.grb_model.setObjective(obj, GRB.MINIMIZE)


    def set_constraints(self):
        self.grb_model.addConstrs((self.Chi.sum(axis=1) == 1), name="one_mode_active")
        
    def get_regression_vector(self, k: int):
        na = self.params.num_y_regressors
        nb = self.params.num_u_regressors

        # print("Y Regressors\n", self.Y[k - na : k, :])
        
        y_slice = self.Y[k - na : k, :]
        u_slice = self.U[k - nb : k, :]
        # print(np.vstack([y_slice, u_slice]).shape)
        return np.vstack([y_slice, u_slice])

        
    # return self.Y[k-na]

    def get_extended_regression_vector(self, k: int):
        return np.vstack([self.get_regression_vector(k), 1])
    
    def solve(self):
        self.grb_model.optimize()
        if self.grb_model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
        else:
            print(f"Optimization ended with status {self.grb_model.status}")
        return self.grb_model.status
    

U = np.zeros((10, 1))
print(f"U: {U}")
print(f"U shape: {U.shape}")
Y = np.zeros((10 ,1))
print(f"Y: {Y}")
print(f"Y shape: {Y.shape}")


params = SimpleNamespace(
    num_y_regressors=2,
    num_u_regressors=2,
    # num_modes=2,
    min_modes = 0,
    max_modes = 5,
    N = Y.shape[0]
)
params.n_bar = params.num_y_regressors + params.num_u_regressors + 1

sys_id_model = StateSpaceSARXID(U, Y, params)
# print(sys_id_model.Y[1:3][:])
print(sys_id_model.get_extended_regression_vector(5))



# sys_id_model.solve()