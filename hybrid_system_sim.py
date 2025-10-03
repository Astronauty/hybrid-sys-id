import numpy as np

import networkx as nx
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jacfwd
from functools import partial

"""
x: state variable
q: generalized coordinates
dq: generalized velocities
ddq: generalized accelerations

u: control inputs

G: Hybrid System Graph (Directed Graph)
- Nodes: Mode (described by tuple of active constraints for the mode, like (0, 1) for contacts 0 and 1 active)
    - dynamics: f(x, u) continuous time dynamics for the mode
    - a: list of constraint functions active in the mode
- Edges: Possible transitions between modes
    - guard:
    - reset:
"""
class HybridSimulator:
    def __init__(self, G=None):
        """
        Args:
            graph (nx.DiGraph): Directed graph representing the hybrid system
        """
        if G is None:
            G = nx.DiGraph()
        self.G = G
 
        # Define constraint functions (maps int to contact function)
        # self.contact_functions = {1: ground_constraint} 
        self.contact_functions = [ground_constraint]

        # Candidate contact modes (tuples of active contacts described by int)
        modes = [(), (0,)]
        for mode in modes:
            G.add_node(mode)
            G.nodes[mode]['a'] = [self.contact_functions[i] for i in mode]


        # Dynamics for each mode
        self.G.nodes[()]['dynamics'] = flight_dyn
        self.G.nodes[(0,)]['dynamics'] = stance_dyn

        # Possible transitions
        self.G.add_edge((), (0,), a=ground_constraint)
        self.G.add_edge((0,), (), a=None)


        # print(G.nodes(data=True))
        nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrows=True)
        plt.show()
        
        ### Based on the constraints a(q), compute the constraint Jacobian A = da/dq and dA = dA/dt for each mode

        self.nq = 1  # Number of position states
        self.nv = 1  # Number of velocity states
        self.nu = 1  # Number of control inputs

        self.M = np.array([[1.0]])
        self.N = np.array([[9.81]])
        self.C = np.array([[0.0]])
        self.Y = np.array([[0.0]])


    # def add_mode(self, id, dynamics)e:
        # """Add a mode to the hybrid system.
        
        # Args:
        #     id (str): Unique identifier for the mode
        #     dynamics (callable): Function representing the mode's dynamics
        # """
        # self.G.add_node(id, dynamics=dynamics)
    
    # def add_transition(self, from_node, to_node, guard, reset):
    #     """Add a transition between modes.
        
    #     Args:
    #         from_node (str): ID of the source mode
    #         to_node (str): ID of the target mode
    #         guard (callable): Function that returns True if transition can occur
    #         reset (callable): Function that resets state upon transition
    #     """
    #     self.G.add_edge(from_node, to_node, guard=guard, reset=reset)
    
    def compute_block_matrix_inverse(self, x, contact_mode):
        q = x[:self.nq]
        v = x[self.nq:self.nq+self.nv]
        u = x[self.nq+self.nv:self.nq+self.nv+self.nu]


        # Compute the block matrix inverse here
        A = self.compute_A(x, contact_mode)
        A = np.array(A[list(contact_mode), :])

        c = A.shape[0]  # Number of constraints
        M = self.M

        block_matrix_inv = np.linalg.inv(np.block([[M, A.T], 
                                     [A, np.zeros((c, c))]]))

        return block_matrix_inv

    def simulate(self, x0, mode0, tf, max_step=1e-2, max_iters=100):
        """Simulate the hybrid system.
        
        Args:
            x0 (np.ndarray): Initial state
            mode0 (str): Initial mode ID
            T (float): Total simulation time
            dt (float): Time step
            
        Returns:
            states (list): List of states over time
            modes (list): List of modes over time
        """
        t, x, mode = 0.0, x0, mode0
        T, X, M  = [t], [x.copy()], [mode]
        
        while t < tf or len(T) < max_iters:
            f = self.G.nodes[mode]['dynamics']
            
            # Define guards from the current mode
            events = []
            edges = list(self.G.out_edges(mode, data=True))
            for _, _, edata in edges:
                event_fn = lambda t, x, mode=mode: self.guard_functions(t, x, mode)
                event_fn.terminal = True
                event_fn.direction = 0  # or set as needed
                events.append(event_fn)
            
            print("here")
            sol = solve_ivp(f, (t, tf), x, method='RK45', events=events, max_step=max_step)
            print("here1")

            
            # Append solution to hybrid trajectory history
            for k in range (1, len(sol.t)):
                T.append(sol.t[k])
                X.append(sol.y[:, k])
                M.append(mode)
                
            t, x = sol.t[-1], sol.y[:, -1] # Get the last timestep of the continuous ode
            
            # print(sol.t_events)
            # Find the contact mode we transition into via IV complementarity
            # if np.any():  # Did an event occur on a contact that is not active?
                # Determine transition contact mode via IV complementarity if we have a new active constraint
            

            # Check FA complementarity to see if there is liftoff


            # guard_triggered = [i for i, e in enumerate(sol.t_events) if len(e) > 0] # i corresponds to the event index with the mode we should switch into
            
            # if guard_triggered:
            #     idx = guard_triggered[0]
            #     next_mode = next_possible_modes[idx]
            #     reset = edges[idx][2]['reset']
            #     x = reset(x, t) # Apply reset map
            #     mode = next_mode
                
            #     T.append(t)
            #     X.append(x)
            #     M.append(mode)
                
        return np.array(T), np.vstack(X), M
    
    def complementary_IV(self, x):
        q = x[:self.nq]
        dq = x[self.nq:self.nq+self.nv]

        # a = np.array([a for _, a in self.G.nodes(data='a') if a is not None]).reshape(-1, 1)
        # a_eval = np.array([a(q) for a in a]).reshape(-1, 1)

        # # Constraint functions that are 0
        # possible_modes = np.array(abs(a_eval) < 1e-6)
        # K = np.where(possible_modes)[0]

        # modes = list(self.G.nodes)

        # for J in K:
        
        # Identify modes where the constraint is active
        possible_new_modes = []

        for mode in self.G.nodes:
            a = self.G.nodes[mode]['a']
            a_eval = a(q)

            active_con = np.where(abs(a_eval) < 1e-6)[0]

    
            if len(active_con) > 0:
                continue

            dq_p, p_hat = self.compute_reset_map(x, mode)

            dq_p_union, p_hat_union = self.compute_reset_map(x, possible_new_modes)

            cond_1 = np.all(p_hat >= 0)  # Non-negative impulses
            cond_2 = np.all(-p_hat)

    def complementary_FA(self, x):
        q = x[:self.nq]
        dq = x[self.nq:self.nq+self.nv]

        modes = list(self.G.nodes) # All modes in hybrid system

        # Find possible modes to transition into (constraints that are active)
        possible_new_modes = [mode for mode in self.G.nodes if np.all(np.abs(self.G.nodes[mode]['a'](q)) < 1e-6)]
        
        for mode in self.G.nodes:
            # a = self.G.nodes[mode]['a']
            # a_eval = a(q)

            not_mode = np.setdiff1d(possible_new_modes, [mode]) # Possible new modes that are not the current one
            
            if mode in possible_new_modes:
                ddq, lam = self.solve_EOM(x, mode)

                ddq_union, lam_union =   self.solve_EOM(x, possible_new_modes)

                cond_1 = np.all(-lam >= 0)
                cond_2 = np.all(-lam_union(not_mode) <= 0)

                if cond_1 and cond_2:
                    return mode # Returns new mode that satisfies FA complementarity
            else:
                continue
            
        return None
        
        # 
        
        
        return

    def compute_reset_map(self, x, contact_mode):
        q = x[:self.nq]
        dq = x[self.nq:self.nq+self.nv]

        A = self.compute_A(q, contact_mode)
        dA = self.compute_dA(x, contact_mode)

        c = A.shape[0]

        N = self.N
        M = self.M
        C = self.C
        Y = self.Y

        block_matrix_inv = self.compute_block_matrix_inverse(x, contact_mode)


        # TODO handle different restitution coefficients for different contacts
        e = 0.5  # Coefficient of restitution
        sol = block_matrix_inv @ np.array([[M @ dq],
                                            [-e * A @ dq]])

        dq_p = sol[:self.nv] # Post-impact velocities
        p_hat = sol[self.nv:self.nv+c] # Lagrange multipliers (impulse)

        return dq_p, p_hat
        
    def guard_functions(self, t, x, contact_mode):

        q = x[:self.nq]
        dq = x[self.nq:self.nq+self.nv]

        a = self.G.nodes[contact_mode]['a']
        if a is None:
            a_eval = np.array([])
        else:
            a_eval = np.array([fn(q) for fn in a])

        ddq, lam = self.solve_EOM(x, contact_mode)


        ## TODO: assumes 1D impulse
        constraint_fcns = np.concatenate([a_eval.flatten(), lam.flatten()])
        print(f"guard_functions: t={t}, x={x}, contact_mode={contact_mode}, constraint_fcns={constraint_fcns}")

        return constraint_fcns

        # is_terminal = np.ones(len(constraint_fcns), 1)
        # direction = [[-np.ones(len(a), 1)], [np.ones(len(lam), 1)]]
        # return constraint_fcns, is_terminal, direction



    def complementary_FA(x):
        q = np.array([x[0], x[1], 0.0])
        return q
    
    
    def solve_EOM(self, x, contact_mode):
        
        if contact_mode not in self.G.nodes:
            raise ValueError(f"Dynamics are not defined for contact mode: {contact_mode}")

        q = np.array(x[:self.nq])
        dq = np.array(x[self.nq:self.nq+self.nv])

        A = self.compute_A(x, contact_mode)
        dA = self.compute_dA(x, contact_mode)

        # Select rows for current contact mode
        A = np.array(A[list(contact_mode), :])
        dA = np.array(dA[list(contact_mode), :])

        c = A.shape[0]

        N = self.N
        M = self.M
        C = self.C
        Y = self.Y

        block_matrix_inv = self.compute_block_matrix_inverse(x, contact_mode)



        sol = block_matrix_inv @ np.block([[Y - N - C @ dq],
                                            [(-dA @ dq).reshape(-1, 1)]])

        ddq = sol[:self.nv] # Accelerations
        lam = sol[self.nv:self.nv+c] # Lagrange multipliers (constraint forces)

        return ddq, lam
    
    def compute_A(self, x, mode):
        """
        Returns the constraint Jacobian A for the given mode.
        """
        # a = self.G.nodes[mode]['a'] #
        # a = jnp.array([fn(q) for fn in self.contact_functions])
        q = jnp.array(x[:self.nq])
        dq = jnp.array(x[self.nq:self.nq+self.nv])

        a = self.contact_functions



        if len(a) == 0:
            A = jnp.empty((0, self.nq))
            return A
        else:
            def a_fn(q):
                return jnp.array([fn(q) for fn in a])
            A = jacfwd(a_fn)(q)

        return np.array(A)
    
    def compute_dA(self, x, mode):
        """
        Returns the time derivative of the constraint Jacobian dA for the given mode.
        """
        q = jnp.array(x[:self.nq])
        dq = jnp.array(x[self.nq:self.nq+self.nv])

        # a = self.G.nodes[mode]['a']
        
        a = self.contact_functions


        if len(a) == 0:
            dA = jnp.empty((0, self.nq))
            # return dA
        else:
            def a_fn(q):
                A = jnp.array([fn(q) for fn in a])
                return A
            
            # A = jacfwd(a_fn)(q)
            dA_dq = jnp.atleast_2d(jacfwd(a_fn)(q))
            dA = jnp.atleast_2d(jnp.dot(dA_dq, dq)) # dA/dt = dA/dq *dq/dt

        return np.array(dA)

# ----------------------
# Hopper parameters
# ----------------------
g = 9.81
xg = 0.0
e = 0.0  # restitution
m = 1.0
l = 0.5


# For the hybrid system, define:
# Modes
# Discrete time dynamics x_kp1 = f(x_k, u_k) for each mode
# Constraints a(x) = 0 for each mode (by extension, the Jacobian A = da/dq and dA = dA/dt)

# Dynamics
# ----------------------
def flight_dyn(t, x):
    q = x[0]
    qd = x[1]
    return np.array([qd, -g])

def stance_dyn(t, x):
    q = x[0]
    qd = x[1]
    k = 1.0
    m = 1.0
    return np.array([qd, -(q - l)*k/m])

# -----------------------
# Constraint Functions
# -----------------------
# def computeA(x)

def ground_constraint(x):
    q = x[0]
    return q - l

# ----------------------
# Guards
# ----------------------
def guard_touchdown(t, q, u=None):
    x, xd = q
    return x - l
guard_touchdown.direction = -1

# def guard_liftoff(t, q, u):
#     x, xd = q
#     # return x - (xg + l_fixed)
#     # return u
#     return x - l_fixed
# guard_liftoff.direction = 1

# ----------------------
# Resets
# ----------------------
# def reset_touchdown(q, t):
#     x, xd = q

#     e = 0.5
#     return np.array([x, -e*xd])
#     # return np.array([x, 0])

# def reset_liftoff(q, t):
#     x, xd = q
#     # x[1] = 10.0
    
#     return x

# ----------------------
# Build graph
# ----------------------
G = nx.DiGraph()
# G.add_node("flight", dynamics=flight_dyn, a=None)
# G.add_node("stance", dynamics=stance_dyn, a=ground_constraint)


# G.add_edge("flight", "stance", guard=guard_touchdown, a=ground_constraint)
# G.add_edge("stance", "flight", guard=guard_liftoff, reset=reset_liftoff)

# ----------------------
# Simulate
# ----------------------
sim = HybridSimulator()
x0 = [1.0, 0.0]   # initial height and velocity
T, X, M = sim.simulate(x0, (), tf=1.0)

print(T)
print(X)
print(M)

# ----------------------
# Plot results
# ----------------------
plt.figure(figsize=(10,5))
plt.plot(T, X[:,0], label="Mass height z(t)")

# Shade stance mode as contiguous intervals
stance_intervals = []
in_stance = False
start_idx = None
for i in range(len(M)):
    if M[i] == "stance" and not in_stance:
        in_stance = True
        start_idx = i
    elif M[i] != "stance" and in_stance:
        in_stance = False
        stance_intervals.append((start_idx, i))

# Handle case where last mode is stance
if in_stance:
    stance_intervals.append((start_idx, len(M)-1))
for start, end in stance_intervals:
    plt.axvspan(T[start], T[end], color='orange', alpha=0.2)

plt.xlabel("Time [s]")
plt.ylabel("Height [m]")
plt.title("1-DOF Hopper Hybrid Simulation (Controlled Stance)")
plt.legend()
plt.grid(True)
plt.show()
