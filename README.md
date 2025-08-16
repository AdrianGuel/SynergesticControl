# SynergesticControl

A Python project for system identification and control algorithms, focusing on least squares estimation and data fitting techniques.

## Project Overview

This project contains system identification tools and utilities for control systems analysis. Currently, the main module provides:

- **Least Squares Estimation**: Core utilities for solving least squares problems using different methods
- **Basis Functions**: Support for linear and polynomial basis functions
- **Interactive Visualization**: Plotly-based visualization tools for data fitting and residual analysis

## Features

### Current Implementation (`sysIdent/leastsquares.py`)

- **Multiple LSQ Solvers**: 
  - `lstsq`: NumPy's least squares solver
  - `normal`: Normal equation method with optional ridge regularization
  - `pinv`: Pseudo-inverse method

- **Flexible Basis Functions**:
  - Linear basis: `y = mx + b`
  - Polynomial basis: Support for any degree polynomial fitting

- **Interactive Visualization**:
  - Two-panel plots showing data vs. fitted model and residuals
  - Smooth model curves with customizable titles
  - Built with Plotly for interactive exploration

## Installation and Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management. Make sure you have Poetry installed on your system.

### Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AdrianGuel/SynergesticControl.git
   cd SynergesticControl
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Usage

### Running the Example

To see the least squares fitting in action:

```bash
poetry run python sysIdent/leastsquares.py
```

This will:
1. Generate synthetic noisy linear data
2. Fit both linear and cubic polynomial models
3. Display interactive Plotly visualizations showing:
   - Original data points vs. fitted curves
   - Residual plots for model validation

### Using as a Module

You can import and use the functions in your own scripts:

```python
from sysIdent.leastsquares import solve_least_squares, linear_basis, polynomial_basis, visualize_fit_plotly
import numpy as np

# Your data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2.1, 4.2, 5.8, 8.1, 10.2])

# Fit a linear model
theta, Y_hat, residuals = solve_least_squares(X, Y, linear_basis())
print("Linear fit parameters:", theta)

# Visualize the results
fig = visualize_fit_plotly(X, Y, linear_basis(), theta, title="My Linear Fit")
fig.show()
```

## Development

### Adding Dependencies

To add new dependencies:

```bash
poetry add package_name
```

For development dependencies:

```bash
poetry add --group dev package_name
```

### Running Tests

Once tests are added to the project:

```bash
poetry run pytest
```

### Code Formatting and Linting

You can add development tools like black, flake8, or mypy:

```bash
poetry add --group dev black flake8 mypy
poetry run black .
poetry run flake8 .
poetry run mypy .
```

## Project Structure

```
SynergesticControl/
├── pyproject.toml          # Poetry configuration and dependencies
├── poetry.lock             # Locked dependency versions
├── README.md              # This file
└── sysIdent/              # Main package
    └── leastsquares.py    # Least squares estimation utilities
```

## Dependencies

- **NumPy** (>=2.3.2, <3.0.0): Numerical computing and linear algebra
- **Plotly** (>=6.3.0, <7.0.0): Interactive visualization


## Author

**AdrianGuel** - adrianjguelc@gmail.com
