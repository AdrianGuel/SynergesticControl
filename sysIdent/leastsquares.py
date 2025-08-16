import numpy as np
from typing import Callable, Iterable, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

Array = np.ndarray

# ---------- Core LSQ utilities (from previous message) ----------

def build_design_matrix(X: Iterable[float], phi: Callable[[float], Iterable[float]]) -> Array:
    X = np.asarray(X)
    rows = [np.asarray(list(phi(x)), dtype=float) for x in X]
    Phi = np.vstack(rows)
    return Phi

def solve_least_squares(
    X: Iterable[float],
    Y: Iterable[float],
    phi: Callable[[float], Iterable[float]],
    method: str = "lstsq",
    ridge_lambda: float = 0.0
) -> Tuple[Array, Array, Array]:
    Phi = build_design_matrix(X, phi)
    Y = np.asarray(Y, dtype=float).reshape(-1)

    if method == "lstsq":
        theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    elif method == "normal":
        PtP = Phi.T @ Phi
        if ridge_lambda > 0:
            PtP = PtP + ridge_lambda * np.eye(PtP.shape[0])
        theta = np.linalg.solve(PtP, Phi.T @ Y)
    elif method == "pinv":
        theta = np.linalg.pinv(Phi) @ Y
    else:
        raise ValueError("method must be 'lstsq', 'normal', or 'pinv'.")

    Y_hat = Phi @ theta
    residuals = Y - Y_hat
    return theta, Y_hat, residuals

def linear_basis() -> Callable[[float], Iterable[float]]:
    return lambda x: (x, 1.0)

def polynomial_basis(degree: int) -> Callable[[float], Iterable[float]]:
    if degree < 0:
        raise ValueError("degree must be >= 0")
    return lambda x: (x ** np.arange(degree, -1, -1))

# ---------- Plotly visualizer ----------

def visualize_fit_plotly(
    X: Iterable[float],
    Y: Iterable[float],
    phi: Callable[[float], Iterable[float]],
    theta: Array,
    title: str = "Least Squares Fit"
):
    """
    Creates a two-row Plotly figure:
      Row 1: scatter of data (X, Y) and fitted model curve
      Row 2: residuals vs X with zero reference line
    """
    X = np.asarray(X, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)

    # Predicted on original X (for residuals)
    Phi = build_design_matrix(X, phi)
    Y_hat = Phi @ theta
    residuals = Y - Y_hat

    # For a smooth model curve, sort X and (optionally) densify it
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    # Densify within the observed range for a smooth line (works for any phi)
    x_min, x_max = X_sorted[0], X_sorted[-1]
    X_line = np.linspace(x_min, x_max, 400)

    def predict_on(xs):
        xs = np.asarray(xs)
        Phi_line = np.vstack([np.asarray(list(phi(x)), dtype=float) for x in xs])
        return Phi_line @ theta

    Y_line = predict_on(X_line)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Data vs. Fitted Model", "Residuals")
    )

    # Row 1: data & model
    fig.add_trace(
        go.Scatter(x=X, y=Y, mode="markers", name="Data"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=X_line, y=Y_line, mode="lines", name="Model"),
        row=1, col=1
    )

    # Row 2: residuals + zero line
    fig.add_trace(
        go.Scatter(x=X, y=residuals, mode="markers", name="Residuals"),
        row=2, col=1
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", row=2, col=1)

    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="residual", row=2, col=1)

    fig.update_layout(title=title, height=700, legend=dict(orientation="h"))
    return fig

# ---------- Minimal example ----------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.linspace(-3, 3, 50)
    Y = 2.5 * X - 1.2 + 0.5 * rng.normal(size=X.size)

    # Fit a line
    theta_lin, Yhat_lin, res_lin = solve_least_squares(X, Y, linear_basis(), method="lstsq")
    print("[Linear] theta = [m, b] =", theta_lin)

    # Show figure
    fig = visualize_fit_plotly(X, Y, linear_basis(), theta_lin, title="Linear LSQ Fit")
    fig.show()

    # Try a cubic polynomial
    phi_poly3 = polynomial_basis(3)
    theta_p3, _, _ = solve_least_squares(X, Y, phi_poly3, method="lstsq")
    print("[Poly deg 3] theta =", theta_p3)
    fig2 = visualize_fit_plotly(X, Y, phi_poly3, theta_p3, title="Cubic Polynomial LSQ Fit")
    fig2.show()
