import math
import pytest
from pitch_tunneling.src.trajectory import (
    HOME_PLATE_Y,
    calculate_tunnel_divergence,
    get_trajectory_points,
    solve_crossing_time,
)


@pytest.fixture
def fastball_row():
    return {
        "release_pos_x": -1.8,
        "release_pos_y": 54.5,
        "release_pos_z": 5.9,
        "vx0": 6.8,
        "vy0": -138.2,
        "vz0": -5.1,
        "ax": -12.4,
        "ay": 30.1,
        "az": -14.2,
    }


@pytest.fixture
def slider_row():
    return {
        "release_pos_x": -1.75,
        "release_pos_y": 54.3,
        "release_pos_z": 5.85,
        "vx0": 4.1,
        "vy0": -124.5,
        "vz0": -2.3,
        "ax": 5.2,
        "ay": 24.6,
        "az": -28.9,
    }


def test_solve_crossing_time_basic(fastball_row):
    t = solve_crossing_time(
        y0=fastball_row["release_pos_y"],
        vy0=fastball_row["vy0"],
        ay=fastball_row["ay"],
        target_y=HOME_PLATE_Y,
    )
    # Pitch should take roughly 0.4 seconds to reach home plate
    assert 0.35 < t < 0.45

    # Verify quadratic evaluates to target_y
    y_at_t = (
        fastball_row["release_pos_y"]
        + fastball_row["vy0"] * t
        + 0.5 * fastball_row["ay"] * (t**2)
    )
    assert math.isclose(y_at_t, HOME_PLATE_Y, abs_tol=1e-6)


def test_solve_crossing_time_linear_falloff():
    # ay = 0
    t = solve_crossing_time(y0=50.0, vy0=-100.0, ay=0.0, target_y=0.0)
    assert math.isclose(t, 0.5, abs_tol=1e-6)


def test_get_trajectory_points_endpoints(fastball_row):
    t, x, y, z = get_trajectory_points(fastball_row, num_points=100)

    assert len(t) == 100
    assert len(x) == 100
    assert len(y) == 100
    assert len(z) == 100

    # Start at release
    assert math.isclose(t[0], 0.0, abs_tol=1e-6)
    assert math.isclose(x[0], fastball_row["release_pos_x"], abs_tol=1e-5)
    assert math.isclose(y[0], fastball_row["release_pos_y"], abs_tol=1e-5)
    assert math.isclose(z[0], fastball_row["release_pos_z"], abs_tol=1e-5)

    # End at home plate
    assert math.isclose(y[-1], HOME_PLATE_Y, abs_tol=1e-5)


def test_calculate_tunnel_divergence_identical_pitches(fastball_row):
    divergence = calculate_tunnel_divergence(fastball_row, fastball_row)
    assert math.isclose(divergence, 0.0, abs_tol=1e-6)


def test_calculate_tunnel_divergence_positive(fastball_row, slider_row):
    divergence = calculate_tunnel_divergence(fastball_row, slider_row, decision_y=23.83)
    assert divergence > 0.0
    # Pitch tunneling divergence at decision point typically within reasonable range in inches
    assert 1.0 < divergence < 30.0
