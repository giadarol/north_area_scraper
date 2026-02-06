import numpy as np

# Written by OpenAI CODEX with human guidance

class FieldInterpolator:
    """Bilinear interpolation over a gridded (x, y) -> (Bx, By) field map.

    If the provided grid only covers the first quadrant, the interpolator
    extends it to all four quadrants using mirror symmetries:
      Quadrant (+,+):  Bx,  By
      Quadrant (-,+):  Bx, -By
      Quadrant (-,-): -Bx, -By
      Quadrant (+,-): -Bx,  By

    Parameters
    ----------
    x_grid : 1D array-like
        Monotonically increasing x coordinates of the grid.
    y_grid : 1D array-like
        Monotonically increasing y coordinates of the grid.
    bx_grid : 2D array-like
        Bx values at each (x, y) grid point (Tesla), shape (len(x_grid), len(y_grid)).
    by_grid : 2D array-like
        By values at each (x, y) grid point (Tesla), shape (len(x_grid), len(y_grid)).
    bounds_error : bool, optional
        If True, raise an error when interpolation is requested outside the grid bounds.
        If False, return `fill_value` for out-of-bounds points. Default is False.
    fill_value : float, optional
        Value to return for out-of-bounds points if `bounds_error` is False. Default is 0.0.
    """

    def __init__(self, x_grid, y_grid, bx_grid, by_grid, *, bounds_error=False, fill_value=0.0):
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.y_grid = np.asarray(y_grid, dtype=float)
        self.bx_grid = np.asarray(bx_grid, dtype=float)
        self.by_grid = np.asarray(by_grid, dtype=float)
        self.bounds_error = bounds_error
        self.fill_value = float(fill_value)

        if self.bx_grid.shape != (self.x_grid.size, self.y_grid.size):
            raise ValueError("Bx grid shape must match (len(x_grid), len(y_grid)).")
        if self.by_grid.shape != self.bx_grid.shape:
            raise ValueError("Bx and By grids must have the same shape.")

    def __call__(self, x, y):
        """Return (Bx, By) at points (x, y); inputs can be scalars or arrays."""
        X_raw, Y_raw = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))

        # Use symmetry to reflect into first quadrant, then apply sign flips to components.
        X = np.abs(X_raw)
        Y = np.abs(Y_raw)

        if self.bounds_error:
            outside = (X < self.x_grid[0]) | (X > self.x_grid[-1]) | (Y < self.y_grid[0]) | (Y > self.y_grid[-1])
            if np.any(outside):
                raise ValueError("Requested points fall outside the field map bounds.")
            X_clipped, Y_clipped = X, Y
        else:
            X_clipped = np.clip(X, self.x_grid[0], self.x_grid[-1])
            Y_clipped = np.clip(Y, self.y_grid[0], self.y_grid[-1])

        xi = np.searchsorted(self.x_grid, X_clipped, side="right") - 1
        yi = np.searchsorted(self.y_grid, Y_clipped, side="right") - 1

        xi = np.clip(xi, 0, self.x_grid.size - 2)
        yi = np.clip(yi, 0, self.y_grid.size - 2)

        x0 = self.x_grid[xi]
        x1 = self.x_grid[xi + 1]
        y0 = self.y_grid[yi]
        y1 = self.y_grid[yi + 1]

        tx = np.where(x1 == x0, 0.0, (X_clipped - x0) / (x1 - x0))
        ty = np.where(y1 == y0, 0.0, (Y_clipped - y0) / (y1 - y0))

        bx00 = self.bx_grid[xi, yi]
        bx10 = self.bx_grid[xi + 1, yi]
        bx01 = self.bx_grid[xi, yi + 1]
        bx11 = self.bx_grid[xi + 1, yi + 1]

        by00 = self.by_grid[xi, yi]
        by10 = self.by_grid[xi + 1, yi]
        by01 = self.by_grid[xi, yi + 1]
        by11 = self.by_grid[xi + 1, yi + 1]

        wx0 = 1.0 - tx
        wy0 = 1.0 - ty

        bx_val = wy0 * (wx0 * bx00 + tx * bx10) + ty * (wx0 * bx01 + tx * bx11)
        by_val = wy0 * (wx0 * by00 + tx * by10) + ty * (wx0 * by01 + tx * by11)

        # Apply symmetry-derived sign flips to extend to four quadrants.
        bx_val = np.where(Y_raw < 0, -bx_val, bx_val)
        by_val = np.where(X_raw < 0, -by_val, by_val)

        if not self.bounds_error:
            outside = (X > self.x_grid[-1]) | (Y > self.y_grid[-1])
            if np.any(outside):
                bx_val = np.where(outside, self.fill_value, bx_val)
                by_val = np.where(outside, self.fill_value, by_val)

        return bx_val, by_val


def load_field_map(path, *, bounds_error=False, fill_value=0.0, scale_xy=1.0):
    """Load the (x, y, Bx, By) map from disk and return a FieldInterpolator."""
    x, y, bx, by = np.loadtxt(path, unpack=True, delimiter=",")

    x *= scale_xy
    y *= scale_xy

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx, ny = x_unique.size, y_unique.size

    if x.size != nx * ny:
        raise ValueError("Field map does not contain a complete rectangular grid.")

    order = np.lexsort((y, x))  # sort by x then y
    bx_sorted = bx[order].reshape(nx, ny)
    by_sorted = by[order].reshape(nx, ny)

    return FieldInterpolator(x_unique, y_unique, bx_sorted, by_sorted, bounds_error=bounds_error, fill_value=fill_value)


def plot_field_map(field, *, component="Bmag", cmap="viridis", shading="auto", save=None, nx=200, ny=200):
    """Plot the field map in all four quadrants using pcolormesh; component can be Bmag, Bx, or By."""
    import matplotlib.pyplot as plt

    if component.lower() in ("bx",):
        data = field.bx_grid
        title = "Bx (Tesla)"
    elif component.lower() in ("by",):
        data = field.by_grid
        title = "By (Tesla)"
    elif component.lower() in ("b", "bmag", "mag"):
        data = np.hypot(field.bx_grid, field.by_grid)
        title = "|B| (Tesla)"
    else:
        raise ValueError("component must be one of: Bmag, Bx, By.")

    # Resample via interpolator over symmetric domain.
    x_max = field.x_grid.max()
    y_max = field.y_grid.max()
    xs = np.linspace(-x_max, x_max, nx)
    ys = np.linspace(-y_max, y_max, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    bx_vals, by_vals = field(X, Y)

    if component.lower() in ("bx",):
        data = bx_vals
    elif component.lower() in ("by",):
        data = by_vals
    else:
        data = np.hypot(bx_vals, by_vals)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(X, Y, data, shading=shading, cmap=cmap)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax, label="Tesla")

    if save:
        fig.savefig(save, bbox_inches="tight")

    return fig, ax


def plot_cut_y(field, y_value_cm, *, n_points=400, save=None):
    """Plot Bx and By along a horizontal cut at the given y (m)."""
    import matplotlib.pyplot as plt

    xs = np.linspace(field.x_grid.min(), field.x_grid.max(), n_points)
    ys = np.full_like(xs, float(y_value_cm))
    bx, by = field(xs, ys)

    fig, ax = plt.subplots()
    ax.plot(xs, bx, label="Bx")
    ax.plot(xs, by, label="By")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Field [T]")
    ax.set_title(f"Field cut at y = {y_value_cm} m")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        fig.savefig(save, bbox_inches="tight")

    return fig, ax


def plot_field_lines(field, *, density=1.0, cmap="plasma", save=None, nx=200, ny=200):
    """Plot field lines (streamplot) colored by |B| with arrow orientation."""
    import matplotlib.pyplot as plt
    # streamplot needs uniformly spaced x/y; resample from the interpolator onto a regular grid
    x_max = field.x_grid.max()
    y_max = field.y_grid.max()
    x = np.linspace(-x_max, x_max, nx)
    y = np.linspace(-y_max, y_max, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    bx, by = field(X, Y)
    mag = np.hypot(bx, by)

    fig, ax = plt.subplots()
    strm = ax.streamplot(X, Y, bx, by, color=mag, cmap=cmap, density=density, arrowsize=1.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Field lines (colored by |B|)")
    fig.colorbar(strm.lines, ax=ax, label="|B| [T]")

    if save:
        fig.savefig(save, bbox_inches="tight")

    return fig, ax