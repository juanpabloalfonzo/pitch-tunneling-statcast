#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from src.data import get_pitcher_statcast
from src.trajectory import (
    DEFAULT_DECISION_Y,
    HOME_PLATE_Y,
    INCHES_PER_FOOT,
    calculate_tunnel_divergence,
    get_trajectory_points,
    solve_crossing_time,
)


def compute_plate_divergence(pitch_a_row, pitch_b_row) -> float:
    return calculate_tunnel_divergence(pitch_a_row, pitch_b_row, decision_y=HOME_PLATE_Y)


def compute_release_separation(pitch_a_row, pitch_b_row) -> float:
    dx = float(pitch_a_row["release_pos_x"]) - float(pitch_b_row["release_pos_x"])
    dz = float(pitch_a_row["release_pos_z"]) - float(pitch_b_row["release_pos_z"])
    return math.hypot(dx, dz) * INCHES_PER_FOOT


def get_representative_mean_pitch(df_pitches: pd.DataFrame) -> dict:
    kinematic_cols = [
        "release_pos_x",
        "release_pos_y",
        "release_pos_z",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
    ]
    mean_series = df_pitches[kinematic_cols].mean()
    return mean_series.to_dict()


def main():
    pitcher_name = "Corbin Burnes"
    start_dt = "2024-04-01"
    end_dt = "2024-09-30"

    print(f"Fetching Statcast data for {pitcher_name} ({start_dt} to {end_dt})...")
    df = get_pitcher_statcast(pitcher_name, start_dt, end_dt)
    if df.empty:
        raise RuntimeError("No pitch data retrieved.")

    pitch_counts = df["pitch_type"].value_counts()
    print("\nPitch arsenal counts:")
    print(pitch_counts)

    # Fastball: Cutter (FC) or 4-Seamer (FF) or Sinker (SI)
    fastball_candidates = [p for p in ["FC", "FF", "SI"] if p in pitch_counts.index]
    primary_fb_type = fastball_candidates[0] if fastball_candidates else pitch_counts.index[0]

    # Breaking ball: Slider (SL), Curveball (CU), Sweeper (ST), Knuckle Curve (KC)
    breaking_candidates = [
        p for p in ["SL", "CU", "ST", "KC"] if p in pitch_counts.index and p != primary_fb_type
    ]
    primary_bb_type = breaking_candidates[0] if breaking_candidates else pitch_counts.index[1]

    df_fb = df[df["pitch_type"] == primary_fb_type].copy()
    df_bb = df[df["pitch_type"] == primary_bb_type].copy()

    print(f"\nAnalyzing primary fastball: {primary_fb_type} (N={len(df_fb)})")
    print(f"Analyzing primary breaking ball: {primary_bb_type} (N={len(df_bb)})")

    # Calculate average kinematic profiles to generate representative trajectory curves
    mean_fb = get_representative_mean_pitch(df_fb)
    mean_bb = get_representative_mean_pitch(df_bb)

    # Subsample pair comparisons to evaluate realistic distribution & mean separation
    sample_size = min(300, len(df_fb), len(df_bb))
    sampled_fb = df_fb.sample(n=sample_size, random_state=42)
    sampled_bb = df_bb.sample(n=sample_size, random_state=42)

    release_seps = []
    decision_divs = []
    plate_divs = []

    for (_, row_a), (_, row_b) in zip(sampled_fb.iterrows(), sampled_bb.iterrows()):
        release_seps.append(compute_release_separation(row_a, row_b))
        decision_divs.append(calculate_tunnel_divergence(row_a, row_b, DEFAULT_DECISION_Y))
        plate_divs.append(compute_plate_divergence(row_a, row_b))

    mean_rel_sep = np.mean(release_seps)
    std_rel_sep = np.std(release_seps)

    mean_dec_div = np.mean(decision_divs)
    std_dec_div = np.std(decision_divs)

    mean_plt_div = np.mean(plate_divs)
    std_plt_div = np.std(plate_divs)

    print("\n" + "=" * 58)
    print(f" TUNNELING DIVERGENCE METRICS: {pitcher_name} (2024)")
    print(f" Comparison: {primary_fb_type} vs {primary_bb_type} (Sample N={sample_size})")
    print("=" * 58)
    print(f"{'Measurement Point':<26} | {'Distance (in)':<14} | {'Std Dev (in)':<12}")
    print("-" * 58)
    print(f"{'Release (y ~ 54 ft)':<26} | {mean_rel_sep:<14.2f} | {std_rel_sep:<12.2f}")
    print(f"{'Decision Point (y = 23.83 ft)':<26} | {mean_dec_div:<14.2f} | {std_dec_div:<12.2f}")
    print(f"{'Home Plate (y = 1.42 ft)':<26} | {mean_plt_div:<14.2f} | {std_plt_div:<12.2f}")
    print("=" * 58)

    # Trajectory points for individual sampled pitches (trajectory cone background)
    fig, (ax_overhead, ax_elevation) = plt.subplots(
        2, 1, figsize=(11, 10), sharex=True, dpi=300
    )

    color_fb = "#1f77b4"
    color_bb = "#d62728"

    for _, row in sampled_fb.head(40).iterrows():
        _, x, y, z = get_trajectory_points(row, num_points=60)
        ax_overhead.plot(y, x, color=color_fb, alpha=0.08, lw=1)
        ax_elevation.plot(y, z, color=color_fb, alpha=0.08, lw=1)

    for _, row in sampled_bb.head(40).iterrows():
        _, x, y, z = get_trajectory_points(row, num_points=60)
        ax_overhead.plot(y, x, color=color_bb, alpha=0.08, lw=1)
        ax_elevation.plot(y, z, color=color_bb, alpha=0.08, lw=1)

    # Mean trajectory lines
    _, x_fb_mean, y_fb_mean, z_fb_mean = get_trajectory_points(mean_fb, num_points=120)
    _, x_bb_mean, y_bb_mean, z_bb_mean = get_trajectory_points(mean_bb, num_points=120)

    # Panel 1: Overhead View (X vs Y)
    ax_overhead.plot(
        y_fb_mean,
        x_fb_mean,
        color=color_fb,
        lw=2.5,
        label=f"{primary_fb_type} Mean Trajectory",
    )
    ax_overhead.plot(
        y_bb_mean,
        x_bb_mean,
        color=color_bb,
        lw=2.5,
        label=f"{primary_bb_type} Mean Trajectory",
    )
    ax_overhead.axvline(
        x=DEFAULT_DECISION_Y,
        color="#333333",
        linestyle="--",
        lw=1.5,
        label="Decision Point (~175ms)",
    )
    ax_overhead.axvline(x=HOME_PLATE_Y, color="black", linestyle=":", lw=1.2, label="Home Plate")

    ax_overhead.set_title(
        f"{pitcher_name} 2024: Overhead Trajectory View ({primary_fb_type} vs {primary_bb_type})",
        fontsize=13,
        fontweight="bold",
    )
    ax_overhead.set_ylabel("Horizontal Position X (ft)", fontsize=11)
    ax_overhead.grid(True, linestyle=":", alpha=0.6)
    ax_overhead.legend(loc="upper left", framealpha=0.9)

    # Panel 2: Side Elevation View (Z vs Y)
    ax_elevation.plot(
        y_fb_mean,
        z_fb_mean,
        color=color_fb,
        lw=2.5,
        label=f"{primary_fb_type} Mean Trajectory",
    )
    ax_elevation.plot(
        y_bb_mean,
        z_bb_mean,
        color=color_bb,
        lw=2.5,
        label=f"{primary_bb_type} Mean Trajectory",
    )
    ax_elevation.axvline(
        x=DEFAULT_DECISION_Y,
        color="#333333",
        linestyle="--",
        lw=1.5,
        label="Decision Point (~175ms)",
    )
    ax_elevation.axvline(x=HOME_PLATE_Y, color="black", linestyle=":", lw=1.2, label="Home Plate")

    ax_elevation.set_title(
        f"Side Elevation Trajectory View ({primary_fb_type} vs {primary_bb_type})",
        fontsize=13,
        fontweight="bold",
    )
    ax_elevation.set_xlabel("Distance to Home Plate Y (ft)", fontsize=11)
    ax_elevation.set_ylabel("Vertical Position Z (ft)", fontsize=11)
    ax_elevation.grid(True, linestyle=":", alpha=0.6)
    ax_elevation.legend(loc="upper left", framealpha=0.9)

    # Invert x-axis so pitch travels from pitcher (right) to batter (left)
    ax_overhead.set_xlim(57, 0)
    ax_elevation.set_xlim(57, 0)

    # Annotate metrics on figure
    summary_text = (
        f"Mean Separation:\n"
        f" • Release: {mean_rel_sep:.1f}\"\n"
        f" • Decision Point: {mean_dec_div:.1f}\"\n"
        f" • Home Plate: {mean_plt_div:.1f}\""
    )
    ax_overhead.text(
        0.02,
        0.05,
        summary_text,
        transform=ax_overhead.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    plt.tight_layout()
    output_filename = "tunneling_burnes_2024.png"
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()

    print(f"\nVisualization saved successfully to: {output_filename}")


if __name__ == "__main__":
    main()
