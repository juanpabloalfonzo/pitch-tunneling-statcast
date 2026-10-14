# Pitch Tunneling & Arsenal Deception Analysis

At 90+ mph, major league hitters do not have the luxury of tracking a baseball all the way into the catcher's mitt before deciding to swing. By the time a pitch crosses the **Commitment Decision Point** (~23.8 ft / ~175–200 ms before home plate), the bat is already launched or held back.

**Pitch Tunneling** is the art and physics of deception: throwing two distinct pitch types along trajectories that are virtually indistinguishable out of the hand and through the decision threshold, only to break violently into different quadrants at the plate.

This repository provides an end-to-end framework to model continuous 3D Statcast flight trajectories, quantify arsenal tunneling metrics, and analyze real-world swinging-strike (whiff) outcomes on consecutive pitch sequences.

---

## Key Metrics & Framework

* **Release Separation ($y \approx 54\text{ ft}$):** Euclidean distance between release origins $(x, z)$, evaluating consistency in arm slots and release points.
* **Decision-Point Divergence ($y = 23.83\text{ ft}$):** Spatial separation between two trajectories when the hitter commits to a swing. Tight separation creates perceptual ambiguity.
* **Plate Divergence ($y = 1.42\text{ ft}$):** Final spatial distance between pitches as they cross home plate.
* **Tunnel Ratio ($\text{Plate Divergence} / \text{Decision Divergence}$):** Quantifies late break efficiency. A high ratio highlights pitch pairings that stay clustered through the tunnel before diverging late.
* **Continuous Divergence Profile $D(y)$:** Traces full-flight separation from release to the mitt to identify whether break occurs early (hitter recognizes pitch) or late (hitter is fooled).

---

## Project Structure

```text
├── analyze_pitcher_tunnels.py       # Pairwise pitch math, representative profiles & divergence metrics
├── pitcher_tunneling_analysis.ipynb # Interactive analysis notebook (case study: Corbin Burnes)
├── src/
│   ├── data.py                      # Statcast data extraction pipeline
│   └── trajectory.py                # 3D equations of motion, crossing times & divergence solvers
└── README.md