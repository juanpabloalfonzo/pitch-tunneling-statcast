import math
from typing import Any, Mapping
import numpy as np


HOME_PLATE_Y = 1.417
DEFAULT_DECISION_Y = 23.83
INCHES_PER_FOOT = 12.0


def solve_crossing_time(y0: float, vy0: float, ay: float, target_y: float) -> float:
    """
    Solve for time t > 0 where y(t) = target_y in the quadratic equation:
        0.5 * ay * t^2 + vy0 * t + (y0 - target_y) = 0

    Selects the physically valid root where the ball travels towards home plate (vy < 0).
    """
    c = y0 - target_y
    b = vy0
    a = 0.5 * ay

    if math.isclose(a, 0.0, abs_tol=1e-12):
        if math.isclose(b, 0.0, abs_tol=1e-12):
            raise ValueError("Invalid kinematics: both acceleration and velocity in y are 0.")
        t = -c / b
        if t <= 0:
            raise ValueError(f"No positive time found to reach target_y={target_y}.")
        return float(t)

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        raise ValueError(
            f"Ball does not reach target_y={target_y} (discriminant={discriminant} < 0)."
        )

    sqrt_disc = math.sqrt(discriminant)
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    valid_roots = [t for t in roots if t > 0 and (vy0 + ay * t) < 0]

    if not valid_roots:
        # Fallback to any positive root if velocity condition isn't strictly met
        valid_roots = [t for t in roots if t > 0]

    if not valid_roots:
        raise ValueError(f"No positive root found to reach target_y={target_y}.")

    return float(min(valid_roots))


def _extract_param(pitch_row: Mapping[str, Any] | Any, key: str) -> float:
    if isinstance(pitch_row, Mapping):
        return float(pitch_row[key])
    if hasattr(pitch_row, key):
        return float(getattr(pitch_row, key))
    return float(pitch_row[key])


def get_trajectory_points(
    pitch_row: Mapping[str, Any] | Any,
    num_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (t, x, y, z) coordinate arrays evaluated from t = 0 (release)
    to t_plate (crossing y = 1.417 ft).
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    x0 = _extract_param(pitch_row, "release_pos_x")
    y0 = _extract_param(pitch_row, "release_pos_y")
    z0 = _extract_param(pitch_row, "release_pos_z")

    vx0 = _extract_param(pitch_row, "vx0")
    vy0 = _extract_param(pitch_row, "vy0")
    vz0 = _extract_param(pitch_row, "vz0")

    ax = _extract_param(pitch_row, "ax")
    ay = _extract_param(pitch_row, "ay")
    az = _extract_param(pitch_row, "az")

    t_plate = solve_crossing_time(y0=y0, vy0=vy0, ay=ay, target_y=HOME_PLATE_Y)
    t = np.linspace(0.0, t_plate, num_points)

    x = x0 + vx0 * t + 0.5 * ax * (t**2)
    y = y0 + vy0 * t + 0.5 * ay * (t**2)
    z = z0 + vz0 * t + 0.5 * az * (t**2)

    return t, x, y, z


def calculate_tunnel_divergence(
    pitch_a_row: Mapping[str, Any] | Any,
    pitch_b_row: Mapping[str, Any] | Any,
    decision_y: float = DEFAULT_DECISION_Y,
) -> float:
    """
    Compute Euclidean distance (in inches) in (x, z) between pitch_a and pitch_b
    at the decision point y = decision_y.
    """
    # Pitch A
    y0_a = _extract_param(pitch_a_row, "release_pos_y")
    vy0_a = _extract_param(pitch_a_row, "vy0")
    ay_a = _extract_param(pitch_a_row, "ay")
    t_a = solve_crossing_time(y0=y0_a, vy0=vy0_a, ay=ay_a, target_y=decision_y)

    x0_a = _extract_param(pitch_a_row, "release_pos_x")
    z0_a = _extract_param(pitch_a_row, "release_pos_z")
    vx0_a = _extract_param(pitch_a_row, "vx0")
    vz0_a = _extract_param(pitch_a_row, "vz0")
    ax_a = _extract_param(pitch_a_row, "ax")
    az_a = _extract_param(pitch_a_row, "az")

    x_a = x0_a + vx0_a * t_a + 0.5 * ax_a * (t_a**2)
    z_a = z0_a + vz0_a * t_a + 0.5 * az_a * (t_a**2)

    # Pitch B
    y0_b = _extract_param(pitch_b_row, "release_pos_y")
    vy0_b = _extract_param(pitch_b_row, "vy0")
    ay_b = _extract_param(pitch_b_row, "ay")
    t_b = solve_crossing_time(y0=y0_b, vy0=vy0_b, ay=ay_b, target_y=decision_y)

    x0_b = _extract_param(pitch_b_row, "release_pos_x")
    z0_b = _extract_param(pitch_b_row, "release_pos_z")
    vx0_b = _extract_param(pitch_b_row, "vx0")
    vz0_b = _extract_param(pitch_b_row, "vz0")
    ax_b = _extract_param(pitch_b_row, "ax")
    az_b = _extract_param(pitch_b_row, "az")

    x_b = x0_b + vx0_b * t_b + 0.5 * ax_b * (t_b**2)
    z_b = z0_b + vz0_b * t_b + 0.5 * az_b * (t_b**2)

    distance_ft = math.hypot(x_a - x_b, z_a - z_b)
    return distance_ft * INCHES_PER_FOOT
