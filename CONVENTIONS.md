# Statcast & Pybaseball Guidelines

- Kinematics:
  - Origin (0,0,0) is home plate apex.
  - Y-axis runs from home plate (y = 0) to pitcher mound (y = 60.5 ft).
  - Front edge of home plate is at y = 1.417 ft.
  - Batter decision commitment plane is at y = 23.83 ft (~175 ms before plate).
  - Trajectory equations:
      x(t) = x0 + vx0*t + 0.5*ax*t^2
      y(t) = y0 + vy0*t + 0.5*ay*t^2
      z(t) = z0 + vz0*t + 0.5*az*t^2
- Data Caching:
  - Cache all pybaseball pulls to `data/*.parquet`.
  - Check for existing local cache before making web requests.
- Architecture: Modular functional code with type hints in `src/` and tests in `tests/`.
