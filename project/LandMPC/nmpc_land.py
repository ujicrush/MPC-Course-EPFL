import numpy as np
import casadi as ca
from typing import Tuple, Optional
import scipy.linalg


class NmpcCtrl:
    """Nonlinear MPC controller (CasADi + IPOPT).

    get_u returns: u0, x_ol, u_ol, t_ol = nmpc.get_u(t0, x0)

    Shapes:
      - x_ol: (12, N+1)
      - u_ol: (4, N)
      - t_ol: (N+1,)
    """

    def __init__(self, rocket, tf: float, x_ref: np.ndarray, u_ref: np.ndarray):
        """Args:
            rocket: Rocket object
            tf: horizon time (seconds)
            x_ref: reference/trim state (12,)
            u_ref: reference/trim input (4,)
        """
        self.rocket = rocket
        self.tf = float(tf)
        self.x_ref = np.array(x_ref, dtype=float).reshape(12,)
        self.u_ref = np.array(u_ref, dtype=float).reshape(4,)

        # Continuous-time symbolic dynamics from rocket
        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]

        # Debug prints (off by default)
        self.debug: bool = True
        self._k: int = 0

        # Warm-start memory
        self.prev_X: Optional[np.ndarray] = None  # (12, N+1)
        self.prev_U: Optional[np.ndarray] = None  # (4, N)

        self._setup_controller()

    def _setup_controller(self) -> None:
        dt = float(self.rocket.Ts)
        N = int(round(self.tf / dt))
        self.N = N
        self.dt = dt

        opti = ca.Opti()
        self.opti = opti

        X = opti.variable(12, N + 1)
        U = opti.variable(4, N)
        self.X = X
        self.U = U

        x0_param = opti.parameter(12)
        self.x0_param = x0_param

        # --------------------
        # better costs (tuned to reduce roll drift + improve robustness)
        # State order: [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z]
        Q_diag = np.array(
            [
                1.0,
                1.0,
                10.0,  
                10.0,
                10.0,
                150.0,  
                10.0,
                10.0,
                10.0,
                100.0,
                100.0,
                100.0,
            ],
            dtype=float,
        )
        Q = np.diag(Q_diag)

        R_diag = np.array(
            [
                5.0,
                5.0,  # delta1, delta2
                0.02,  # Pavg
                0.5,  # Pdiff lower -> more roll authority
            ],
            dtype=float,
        )
        R = np.diag(R_diag)

        # Small input-rate penalty for smoother commands
        Rd_diag = np.array([5.0, 5.0, 0.02, 0.2], dtype=float)
        Rd = np.diag(Rd_diag)

        # Wrap angle error for gamma in the cost to avoid discontinuities
        def wrap_angle(a):
            return ca.atan2(ca.sin(a), ca.cos(a))

        # Terminal cost from continuous-time LQR around the trim
        A_cont, B_cont = self.rocket.linearize(self.x_ref, self.u_ref)
        P_matrix = scipy.linalg.solve_continuous_are(A_cont, B_cont, Q, R)

        obj = 0
        for k in range(N):
            e = X[:, k] - self.x_ref
            e_gamma = wrap_angle(e[5])
            e_x = ca.vertcat(e[0:5], e_gamma, e[6:12])

            e_u = U[:, k] - self.u_ref
            obj += ca.mtimes([e_x.T, Q, e_x]) + ca.mtimes([e_u.T, R, e_u])

            if k > 0:
                du = U[:, k] - U[:, k - 1]
                obj += ca.mtimes([du.T, Rd, du])

        eN = X[:, N] - self.x_ref
        eN_gamma = wrap_angle(eN[5])
        e_xN = ca.vertcat(eN[0:5], eN_gamma, eN[6:12])
        obj += ca.mtimes([e_xN.T, P_matrix, e_xN])

        opti.minimize(obj)

        # --------------------
        # Constraints
        opti.subject_to(X[:, 0] == x0_param)

        # Dynamics (RK4)
        for k in range(N):
            k1 = self.f(X[:, k], U[:, k])
            k2 = self.f(X[:, k] + (dt / 2) * k1, U[:, k])
            k3 = self.f(X[:, k] + (dt / 2) * k2, U[:, k])
            k4 = self.f(X[:, k] + dt * k3, U[:, k])
            x_next = X[:, k] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next)

        # Safety constraints
        beta_lim = float(np.deg2rad(80))
        opti.subject_to(opti.bounded(-beta_lim, X[4, :], beta_lim))
        opti.subject_to(X[11, :] >= 0)  # z >= 0

        # Input bounds
        LBU = np.array(self.rocket.LBU, dtype=float).reshape(4,)
        UBU = np.array(self.rocket.UBU, dtype=float).reshape(4,)
        for k in range(N):
            opti.subject_to(opti.bounded(LBU, U[:, k], UBU))

        # Solver settings
        opts = {
            "expand": True,
            "ipopt.print_level": 0,
            "print_time": 0,
            # Mild robustness settings
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 10,
        }
        opti.solver("ipopt", opts)

    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x0 = np.array(x0, dtype=float).reshape(12,)
        self.opti.set_value(self.x0_param, x0)

        # Warm start: shift previous solution if available
        if self.prev_X is not None and self.prev_U is not None:
            X_init = np.hstack([self.prev_X[:, 1:], self.prev_X[:, -1:]])
            U_init = np.hstack([self.prev_U[:, 1:], self.prev_U[:, -1:]])
            self.opti.set_initial(self.X, X_init)
            self.opti.set_initial(self.U, U_init)
        else:
            self.opti.set_initial(self.X, ca.repmat(ca.DM(self.x_ref), 1, self.N + 1))
            self.opti.set_initial(self.U, ca.repmat(ca.DM(self.u_ref), 1, self.N))

        solved = True
        try:
            sol = self.opti.solve()
            u0 = np.array(sol.value(self.U[:, 0])).reshape(4,)
            x_ol = np.array(sol.value(self.X))
            u_ol = np.array(sol.value(self.U))
        except RuntimeError:
            solved = False
            # Robust fallback: use last plan if available, otherwise trim
            if self.prev_U is not None and self.prev_X is not None:
                u0 = np.array(self.prev_U[:, 0]).reshape(4,)
                x_ol = np.array(self.prev_X)
                u_ol = np.array(self.prev_U)
            else:
                u0 = np.array(self.u_ref).reshape(4,)
                x_ol = np.tile(self.x_ref.reshape(12, 1), (1, self.N + 1))
                u_ol = np.tile(self.u_ref.reshape(4, 1), (1, self.N))

        # Save for next warm start
        self.prev_X = x_ol
        self.prev_U = u_ol

        # Optional debug print
        if self.debug:
            k = self._k
            self._k += 1
            if k % 5 == 0:
                p = x0[9:12]
                v = x0[6:9]
                beta = float(x0[4])
                gamma = float(x0[5])
                p_ref = self.x_ref[9:12]
                gamma_ref = float(self.x_ref[5])
                e_p = p - p_ref
                e_roll = (gamma - gamma_ref + np.pi) % (2 * np.pi) - np.pi
                print(
                    f"[NMPC {'OK' if solved else 'FB'}] t={t0:5.2f} "
                    f"|e_p|={np.linalg.norm(e_p):.3f} |v|={np.linalg.norm(v):.3f} "
                    f"e_roll={e_roll:.3f} beta={beta:.3f} u0={u0}"
                )

        t_ol = np.linspace(t0, t0 + self.tf, self.N + 1)
        return u0, x_ol, u_ol, t_ol
