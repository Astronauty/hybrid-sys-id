# import numpy as np
# class KLinReg():
#     """Switched linear regression class implementing the k-LinReg algorithm.
    
#     """
#     def __init__(self, X, y, s, theta_init, max_iters=100, tol=1e-6):
#         """Initialize the KLinReg model.

#         Args:
#             X (np.ndarray): Input features of shape (N, D).
#             y (np.ndarray): Target values of shape (N, 1).
#             s (int): Prespecified number of modes.
#             theta (np.ndarray): Initial model parameters for each mode. (s * D, iters)
#         """
#         self.X = X
#         self.y = y
#         self.s = s
#         self.max_iters = max_iters
#         self.tol = tol
        
#         self.N, self.d = X.shape
#         self.i = 0

#         # self.theta[:, 0, 0] = theta_init # 
#         self.theta = np.zeros((self.d, s, max_iters+1))
#         self.theta[:, :, 0] = theta_init

#         self.mode_assignments = np.zeros((self.N, max_iters+1), dtype=int)
#         self.i = 0



#     def classify_data_points(self):
#         """Classify each data point to the mode minimizing residual error."""
#         residuals = np.zeros((self.N, self.s))
#         for j in range(self.s):
#             y_pred = self.X @ self.theta[:, j, self.i]
#             residuals[:, j] = (self.y.flatten() - y_pred) ** 2

#         self.mode_assignments[:, self.i] = np.argmin(residuals, axis=1)
#         return self.mode_assignments[:, self.i]
    

#     def fit(self):
#         """Run k-LinReg algorithm until convergence."""
#         for it in range(self.max_iters):
#             self.classify_data_points()

#             # Update each mode
#             for j in range(self.s):
#                 idx = np.where(self.mode_assignments[:, self.i] == j)[0]
#                 if len(idx) < self.d:
#                     continue  # not enough points to fit regression
#                 X_j = self.X[idx, :]
#                 y_j = self.y[idx]

#                 # Solve least squares
#                 XtX = X_j.T @ X_j
#                 if np.linalg.matrix_rank(XtX) == self.d:
#                     self.theta[:, j, self.i+1] = np.linalg.inv(XtX) @ X_j.T @ y_j
#                 else:
#                     # fallback: pseudo-inverse
#                     self.theta[:, j, self.i+1] = np.linalg.pinv(X_j) @ y_j
#             # Check convergence
#             diff = np.linalg.norm(self.theta[:, :, self.i+1] - self.theta[:, :, self.i])
#             if diff < self.tol:
#                 print(f"Converged at iteration {it}")
#                 break
        
#             if it > 0 and np.array_equal(self.mode_assignments[:, self.i],
#                                          self.mode_assignments[:, self.i-1]):
#                 print(f"Converged (no assignment changes) at iteration {it}")
#                 break

#             self.i += 1

#         return self.theta[:, :, self.i], self.mode_assignments[:, self.i]
        

import numpy as np

class KLinRegMultiOutput:
    """Switched linear regression for multi-output targets."""
    def __init__(self, X, y, s, theta_init, max_iters=100, tol=1e-6):
        """
        Args:
            X (np.ndarray): (N, d)
            y (np.ndarray): (N, M)
            s (int): number of modes
            theta_init (np.ndarray): (d, s, M)
        """
        self.X = X
        self.y = y
        self.s = s
        self.max_iters = max_iters
        self.tol = tol

        self.N, self.d = X.shape
        self.N, self.M = y.shape
        self.i = 0

        # theta: (d, s, M, max_iters+1)
        self.theta = np.zeros((self.d, s, self.M, max_iters+1))
        self.theta[:, :, :, 0] = theta_init

        self.mode_assignments = np.zeros((self.N, max_iters+1), dtype=int)

    def classify_data_points(self):
        """Assign each sample to the mode with minimal squared error."""
        residuals = np.zeros((self.N, self.s))
        for j in range(self.s):
            y_pred = self.X @ self.theta[:, j, :, self.i]  # (N, M)
            residuals[:, j] = np.sum((self.y - y_pred) ** 2, axis=1)  # sum over outputs
        self.mode_assignments[:, self.i] = np.argmin(residuals, axis=1)
        return self.mode_assignments[:, self.i]

    def fit(self):
        """Fit k-LinReg until convergence."""
        for it in range(self.max_iters):
            self.classify_data_points()

            for j in range(self.s):
                idx = np.where(self.mode_assignments[:, self.i] == j)[0]
                if len(idx) < self.d:
                    continue

                X_j = self.X[idx, :]        # (n_j, d)
                Y_j = self.y[idx, :]        # (n_j, M)

                XtX = X_j.T @ X_j           # (d, d)
                if np.linalg.matrix_rank(XtX) == self.d:
                    self.theta[:, j, :, self.i+1] = np.linalg.inv(XtX) @ X_j.T @ Y_j
                else:
                    self.theta[:, j, :, self.i+1] = np.linalg.pinv(X_j) @ Y_j

            # Check convergence
            diff = np.linalg.norm(self.theta[:, :, :, self.i+1] - self.theta[:, :, :, self.i])
            if diff < self.tol:
                print(f"Converged at iteration {it}")
                break

            if it > 0 and np.array_equal(self.mode_assignments[:, self.i],
                                         self.mode_assignments[:, self.i-1]):
                print(f"Converged (no assignment changes) at iteration {it}")
                break

            self.i += 1

        return self.theta[:, :, :, self.i], self.mode_assignments[:, self.i]
