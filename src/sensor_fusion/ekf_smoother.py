from typing import Optional, Tuple
import numpy as np


class TrajectoryEKFSmoother:
    """6-DOF Constant Velocity Extended Kalman Filter with Rauch-Tung-Striebel (RTS) backward smoother.
    Fuses noisy GNSS positions, optional barometric altitude, and IMU constraints.
    Incorporates Chi-Square gating to reject GPS multipath jumps (§5 of Spec).
    """

    def __init__(
        self,
        gps_std: float = 1.5,
        baro_std: float = 0.5,
        accel_std: float = 0.5,
        chi_square_gate: float = 9.21  # 99% confidence for 2-DOF innovation
    ):
        self.gps_std = gps_std
        self.baro_std = baro_std
        self.accel_std = accel_std
        self.chi_square_gate = chi_square_gate

    def filter_and_smooth(
        self,
        times: np.ndarray,
        measured_positions: np.ndarray,
        baro_altitudes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Runs forward EKF pass followed by backward RTS smoothing pass.
        measured_positions: (N, 3) in local ENU frame (meters).
        Returns: smoothed_positions: (N, 3) in local ENU frame.
        """
        N = len(times)
        if N < 2:
            return measured_positions.copy()

        # State: [x, y, z, vx, vy, vz]^T (6x1)
        dim_x = 6
        dim_z = 3

        # Forward storage for RTS
        x_pred = np.zeros((N, dim_x))
        P_pred = np.zeros((N, dim_x, dim_x))
        x_filt = np.zeros((N, dim_x))
        P_filt = np.zeros((N, dim_x, dim_x))
        F_list = np.zeros((N, dim_x, dim_x))

        # Initial state from first two measurements
        dt0 = max(times[1] - times[0], 1e-4)
        v0 = (measured_positions[1] - measured_positions[0]) / dt0
        x_filt[0] = np.hstack([measured_positions[0], v0])
        P_filt[0] = np.diag([
            self.gps_std**2, self.gps_std**2, self.baro_std**2,
            5.0**2, 5.0**2, 2.0**2
        ])

        H = np.zeros((dim_z, dim_x))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        R = np.diag([self.gps_std**2, self.gps_std**2, self.baro_std**2])

        # Forward EKF pass
        for k in range(1, N):
            dt = max(times[k] - times[k - 1], 1e-4)

            # State transition matrix F
            F = np.eye(dim_x)
            F[0, 3] = dt
            F[1, 4] = dt
            F[2, 5] = dt
            F_list[k - 1] = F

            # Process noise covariance Q (continuous white noise acceleration model)
            q_val = self.accel_std**2
            Q_pos = (dt**3 / 3.0) * q_val
            Q_vel = dt * q_val
            Q_cross = (dt**2 / 2.0) * q_val

            Q = np.zeros((dim_x, dim_x))
            for i in range(3):
                Q[i, i] = Q_pos
                Q[i + 3, i + 3] = Q_vel
                Q[i, i + 3] = Q_cross
                Q[i + 3, i] = Q_cross

            # Predict
            x_pred[k] = F @ x_filt[k - 1]
            P_pred[k] = F @ P_filt[k - 1] @ F.T + Q

            # Measurement
            z = measured_positions[k].copy()
            if baro_altitudes is not None:
                z[2] = baro_altitudes[k]

            # Innovation
            y = z - H @ x_pred[k]
            S = H @ P_pred[k] @ H.T + R
            S_inv = np.linalg.inv(S)

            # Chi-square gating on horizontal innovation to reject multipath
            d2 = y[:2] @ S_inv[:2, :2] @ y[:2]
            if d2 > self.chi_square_gate:
                # Reject outlier measurement: keep predicted state
                x_filt[k] = x_pred[k]
                P_filt[k] = P_pred[k]
            else:
                # Kalman gain
                K_gain = P_pred[k] @ H.T @ S_inv
                x_filt[k] = x_pred[k] + K_gain @ y
                P_filt[k] = (np.eye(dim_x) - K_gain @ H) @ P_pred[k]

        # Backward Rauch-Tung-Striebel (RTS) smoother pass
        x_smooth = np.zeros((N, dim_x))
        P_smooth = np.zeros((N, dim_x, dim_x))

        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]

        for k in range(N - 2, -1, -1):
            F = F_list[k]
            # Smoother gain C_k = P_filt[k] @ F.T @ inv(P_pred[k+1])
            P_pred_inv = np.linalg.inv(P_pred[k + 1])
            C_k = P_filt[k] @ F.T @ P_pred_inv

            x_smooth[k] = x_filt[k] + C_k @ (x_smooth[k + 1] - x_pred[k + 1])
            P_smooth[k] = P_filt[k] + C_k @ (P_smooth[k + 1] - P_pred[k + 1]) @ C_k.T

        return x_smooth[:, :3]
