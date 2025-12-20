import numpy as np
from .MPCControl_base import MPCControl_base


class MPCControl_y(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7, 10])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        #################################################
        super()._setup_controller()
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        u0, x_traj, u_traj = super().get_u(x0, x_target, u_target)
        return u0, x_traj, u_traj
        #################################################
