import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class HybridSimulator:
    def __init__(self, G=None):
        """
        Args:
            graph (nx.DiGraph): Directed graph representing the hybrid system
        """
        if G is None:
            G = nx.DiGraph()
        self.G = G

    def add_mode(self, id, dynamics):
        """Add a mode to the hybrid system.
        
        Args:
            id (str): Unique identifier for the mode
            dynamics (callable): Function representing the mode's dynamics
        """
        self.G.add_node(id, dynamics=dynamics)
    
    def add_transition(self, from_node, to_node, guard, reset):
        """Add a transition between modes.
        
        Args:
            from_node (str): ID of the source mode
            to_node (str): ID of the target mode
            guard (callable): Function that returns True if transition can occur
            reset (callable): Function that resets state upon transition
        """
        self.G.add_edge(from_node, to_node, guard=guard, reset=reset)
        

    def simulate(self, x0, mode0, tf, max_step=1e-5):
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
        
        while t < tf:
            f = self.G.nodes[mode]['dynamics']
            
            edges = list(self.G.out_edges(mode, data=True))
            guards = [edata["guard"] for _, _, edata in edges]
            # resets = [edata["reset"] for _, _, edata in edges]
            
            next_possible_modes = [v for _, v, _ in edges]
            
            def make_event(fn):
                def event(t, q, u=None): return fn(t, q, u)
                event.terminal = True
                event.direction = 0
                return event
                
            events = [make_event(fn) for fn in guards]
            
            sol = solve_ivp(f, (t, tf), x, method='RK45', events=events, max_step=max_step)
            
            # Append solution to hybrid trajectory history
            for k in range (1, len(sol.t)):
                T.append(sol.t[k])
                X.append(sol.y[:, k])
                M.append(mode)
                
            t, x = sol.t[-1], sol.y[:, -1] # Get the last timestep of the continuous ode
            
            guard_triggered = [i for i, e in enumerate(sol.t_events) if len(e) > 0] # i corresponds to the event index with the mode we should switch into
            
            if guard_triggered:
                idx = guard_triggered[0]
                next_mode = next_possible_modes[idx]
                reset = edges[idx][2]['reset']
                x = reset(x, t) # Apply reset map
                mode = next_mode
                
                T.append(t)
                X.append(x)
                M.append(mode)
                
        return np.array(T), np.vstack(X), M
            
            
            
            
# ----------------------
# Hopper parameters
# ----------------------
g = 9.81
xg = 0.0
e = 0.0  # restitution
m = 1.0
l_fixed = 0.5

# commanded leg length trajectory
def l(t):     
    return 0.4 + 0.1*np.sin(2*np.pi*1.0*t)

def ldot(t):  
    return 0.1*2*np.pi*1.0*np.cos(2*np.pi*1.0*t)

# ----------------------
# Dynamics
# ----------------------
def flight_dyn(t, q):
    x, xd = q
    return np.array([xd, -g])

def stance_dyn(t, q):
    x, xd = q
    F = 0.0
    return np.array([xd, -g + F/m])

# ----------------------
# Guards
# ----------------------
def guard_touchdown(t, q, u=None):
    x, xd = q
    return x - (xg + l_fixed)
guard_touchdown.direction = -1

def guard_liftoff(t, q, u):
    x, xd = q
    # return x - (xg + l_fixed)
    # return u
    return x - l_fixed
guard_liftoff.direction = 1

# ----------------------
# Resets
# ----------------------
def reset_touchdown(q, t):
    x, xd = q

    e = 0.5
    return np.array([x, -e*xd])
    # return np.array([x, 0])

def reset_liftoff(q, t):
    x, xd = q
    # x[1] = 10.0
    
    return x

# ----------------------
# Build graph
# ----------------------
G = nx.DiGraph()
G.add_node("flight", dynamics=flight_dyn)
G.add_node("stance", dynamics=stance_dyn)


G.add_edge("flight", "stance", guard=guard_touchdown, reset=reset_touchdown)
# G.add_edge("stance", "flight", guard=guard_liftoff, reset=reset_liftoff)

# ----------------------
# Simulate
# ----------------------
sim = HybridSimulator(G)
x0 = [1.0, 0.0]   # initial height and velocity
T, X, M = sim.simulate(x0, "flight", tf=1.0)

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
