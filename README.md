# Playground Streamlit

Interactive web application built with Streamlit, featuring Model Predictive Control (MPC) simulations for autonomous vehicle scenarios.

## Projects

- **MPC Highway** - Simulate a Model Predictive Controller driving a car on a highway toward a goal position, with configurable speed limits and prediction horizons.
- **MPC Parking** - Simulate a Model Predictive Controller navigating a car through a 2D environment with obstacle avoidance and multi-goal waypoints.

## Getting Started

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install and Run

```bash
# Install dependencies (uv will automatically download Python 3.12 if needed)
uv sync

# Run the app
uv run streamlit run "01_🧩_Playgorund.py" --server.port=8002
```

The app will be available at `http://localhost:8002`.

### Docker

A Dockerfile is also provided for containerized deployment:

```bash
docker build -t playground .
docker run -p 8002:8002 playground
```

## Tech Stack

- [Streamlit](https://streamlit.io/) - Web application framework
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Plotting and animation
- [SciPy](https://scipy.org/) - Optimization (SLSQP for MPC)
