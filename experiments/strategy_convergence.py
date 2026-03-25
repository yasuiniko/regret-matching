"""Illustrates convergence and non-convergence properties in a Shapley game."""

from game_theory.full_information_regret_matching import FullInfoRegretMatchingActions
import game_theory.game as game
from utils.blit_manager import BlitManager

import matplotlib.pyplot as plt
import numpy as np


# --- SETUP AND FORMATTING ---


def setup_figure_and_axes():
    """Creates the figure, gridspec, and all subplots."""
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 2)

    axes = {
        "ternA": fig.add_subplot(gs[0, 0]),
        "ternB": fig.add_subplot(gs[0, 1]),
        "freq1": fig.add_subplot(gs[1, :]),
        "ce": fig.add_subplot(gs[2, :]),
        "cce": fig.add_subplot(gs[3, :]),
    }
    return fig, axes


def format_ternary_axis(ax, title):
    """Draws the background triangle for a ternary plot."""
    h = np.sqrt(3) / 2
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], "k-", lw=1.5, alpha=0.5)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title(title, pad=15)

    # Label corners
    ax.text(0, -0.05, "Action 0", ha="center", va="top", fontsize=9)
    ax.text(1, -0.05, "Action 1", ha="center", va="top", fontsize=9)
    ax.text(0.5, h + 0.05, "Action 2", ha="center", va="bottom", fontsize=9)


def format_timeseries_axes(axes):
    """Sets titles, labels, and limits for the time series plots."""
    axes["freq1"].set_title("Player A Strategy (average over time)")
    axes["freq1"].set_ylabel("Probability")
    axes["freq1"].set_ylim(0, 1)

    axes["ce"].set_title("Player A Internal Regret (CE Convergence)")
    axes["ce"].set_ylabel("Regret")

    axes["cce"].set_title("Player A External Regret (CCE Convergence)")
    axes["cce"].set_ylabel("Regret")
    axes["cce"].set_xlabel("Round")


# --- DATA MANAGEMENT ---


def init_tracking_arrays(data_a, data_b, plotfreq):
    """Initializes the data tracking dictionaries with the first timestep."""
    p_A, rext_A, rint_A, s_A = data_a
    p_B, rext_B, rint_B, s_B = data_b
    h = np.sqrt(3) / 2
    y = {}

    # Set up Tracking Arrays for Time Series (Player A)
    y["cce"] = np.full(plotfreq, np.nan)
    y["ce"] = np.full(plotfreq, np.nan)
    y["cce"][0] = rext_A
    y["ce"][0] = rint_A

    for j in range(p_A.size):
        y[f"f{j}1"] = np.full(plotfreq, np.nan)
        y[f"f{j}1"][0] = p_A[j]

    # Set up Tracking Arrays for Ternary Plots
    y["ternA_x"] = np.full(plotfreq, np.nan)
    y["ternA_y"] = np.full(plotfreq, np.nan)
    y["ternB_x"] = np.full(plotfreq, np.nan)
    y["ternB_y"] = np.full(plotfreq, np.nan)

    y["ternA_x"][0] = p_A[1] + 0.5 * p_A[2]
    y["ternA_y"][0] = h * p_A[2]
    y["ternB_x"][0] = p_B[1] + 0.5 * p_B[2]
    y["ternB_y"][0] = h * p_B[2]

    return y


def init_artists(axes, y, num_actions):
    """Creates and returns the matplotlib line objects for all plots."""
    lns = {}

    # Initialize Time Series lines
    ts_keys = ["cce", "ce"] + [f"f{j}1" for j in range(num_actions)]
    for k in ts_keys:
        ax = axes["freq1"] if k.startswith("f") else axes[k]
        lns[k] = ax.plot(np.arange(len(y[k])) + 1, y[k], "-", alpha=0.7)[0]
        ax.axhline(alpha=0)

    # Initialize Ternary lines
    lns["ternA"] = axes["ternA"].plot(
        y["ternA_x"], y["ternA_y"], "-", color="blue", alpha=0.6, lw=1.5
    )[0]
    lns["ternB"] = axes["ternB"].plot(
        y["ternB_x"], y["ternB_y"], "-", color="red", alpha=0.6, lw=1.5
    )[0]

    return lns


def init_artists(axes, y, num_actions):
    """Creates and returns the matplotlib line objects for all plots."""
    lns = {}

    # ce / cce
    for j in range(num_actions):
        k = f"f{j}1"
        lns[k] = axes["freq1"].plot(
            np.arange(len(y[k])) + 1, y[k], "-", alpha=0.7, label=f"Action {j}"
        )[0]

    lns["cce"] = axes["cce"].plot(
        np.arange(len(y["cce"])) + 1, y["cce"], "-", alpha=0.7, label="External Regret"
    )[0]
    lns["ce"] = axes["ce"].plot(
        np.arange(len(y["ce"])) + 1, y["ce"], "-", alpha=0.7, label="Internal Regret"
    )[0]

    # legends
    for ax_key in ["freq1", "ce", "cce"]:
        axes[ax_key].axhline(alpha=0)
        axes[ax_key].legend(loc="upper right", fontsize=8)

    # scatter trails
    lns["ternA"] = axes["ternA"].scatter(
        [], [], c=[], cmap="Blues", vmin=0, vmax=1, s=15, label="Trail"
    )
    lns["ternB"] = axes["ternB"].scatter(
        [], [], c=[], cmap="Reds", vmin=0, vmax=1, s=15, label="Trail"
    )

    # dots for current strategy
    lns["ternA_dot"] = axes["ternA"].plot(
        [y["ternA_x"][0]], [y["ternA_y"][0]], "bo", markersize=8, label="Current"
    )[0]
    lns["ternB_dot"] = axes["ternB"].plot(
        [y["ternB_x"][0]], [y["ternB_y"][0]], "ro", markersize=8, label="Current"
    )[0]

    return lns


def expand_data_capacity(i, plotfreq, redrawtime, y, lns, axes, num_actions):
    """Doubles array capacities and updates axis limits when arrays fill up."""
    plotfreq *= 2
    redrawtime = i + plotfreq
    xs = np.arange(redrawtime)

    ts_keys = ["cce", "ce"] + [f"f{j}1" for j in range(num_actions)]

    for k in y.keys():
        # Pad with NaNs
        y[k] = np.hstack((y[k], np.full(plotfreq, np.nan)))

        # Only extend X limits for time-series graphs
        if k in ts_keys:
            ax = axes["freq1"] if k.startswith("f") else axes[k]
            lns[k].set_xdata(xs)
            ax.set_xlim(right=xs[-1])

            # Recalculate Y limits for the Regret graphs
            if not k.startswith("f"):
                mx = max(np.nanmax(y[k][i // 5 : i]), 0)
                mn = min(np.nanmin(y[k][i // 5 : i]), 0)
                mx += 0.1 * (mx - mn) if mx != mn else 0.1
                mn -= 0.1 * (mx - mn) if mx != mn else 0.1
                ax.set_ylim(mn, mx)

                # Redraw the horizontal zero-line
                ax.lines[-1].remove()
                ax.axhline(color="black", alpha=0.4)

    plt.draw()
    return plotfreq, redrawtime, y


def append_new_data(i, data_a, data_b, y):
    """Inserts the latest simulation step into the tracking arrays."""
    p_A, rext_A, rint_A, s_A = data_a
    p_B, rext_B, rint_B, s_B = data_b
    h = np.sqrt(3) / 2

    y["cce"][i] = rext_A
    y["ce"][i] = rint_A
    for j in range(p_A.size):
        y[f"f{j}1"][i] = s_A[j]

    y["ternA_x"][i] = p_A[1] + 0.5 * p_A[2]
    y["ternA_y"][i] = h * p_A[2]
    y["ternB_x"][i] = p_B[1] + 0.5 * p_B[2]
    y["ternB_y"][i] = h * p_B[2]


def update_artists(i, plotfreq, redrawtime, y, lns, bm):
    """Updates the line artists with the new data and triggers a blit update."""
    fpd = 100
    should_redraw = (
        (i % redrawtime == 0)
        or (plotfreq > fpd and (i - redrawtime) % (plotfreq // fpd) == 0)
        or (plotfreq > 10000 and (i - redrawtime) % 100 == 0)
    )

    if should_redraw:
        for k in y.keys():
            if "tern" not in k:
                lns[k].set_ydata(y[k])

        tail_length = 150  # number of historical points to keep visible
        start_idx = max(0, i - tail_length)
        current_len = i - start_idx

        # Create an intensity array from 0 (light/transparent) to 1 (dark/solid)
        fades = np.linspace(0, 1, current_len)

        # Update X, Y coordinate pairs for the scatter plot
        lns["ternA"].set_offsets(
            np.c_[y["ternA_x"][start_idx:i], y["ternA_y"][start_idx:i]]
        )
        lns["ternB"].set_offsets(
            np.c_[y["ternB_x"][start_idx:i], y["ternB_y"][start_idx:i]]
        )

        # Map the intensities to the Blues/Reds colormaps
        lns["ternA"].set_array(fades)
        lns["ternB"].set_array(fades)

        # Update current strategy dots
        lns["ternA_dot"].set_data([y["ternA_x"][i]], [y["ternA_y"][i]])
        lns["ternB_dot"].set_data([y["ternB_x"][i]], [y["ternB_y"][i]])

        bm.update()


def on_close(event):
    print("Window closed. Exiting program.")
    import sys

    sys.exit()


def plot(g):
    gp = g.selfplay()

    # 1. Setup the dashboard
    fig, axes = setup_figure_and_axes()
    fig.suptitle(f"Sim for game: {g.name}", fontsize=16, fontweight='bold', y=0.98)
    format_ternary_axis(axes["ternA"], "Player A Strategy (instantaneous)")
    format_ternary_axis(axes["ternB"], "Player B Strategy (instantaneous)")
    format_timeseries_axes(axes)
    fig.tight_layout()
    fig.canvas.mpl_connect("close_event", on_close)

    # 2. Extract first points and initialize arrays
    data_a, data_b = next(gp)
    num_actions = data_a[0].size
    plotfreq = 1

    y = init_tracking_arrays(data_a, data_b, plotfreq)
    lns = init_artists(axes, y, num_actions)

    # 3. Start drawing
    bm = BlitManager(fig.canvas, lns.values())
    plt.show(block=False)
    plt.pause(0.1)

    # 4. Main Animation Loop
    i = 1
    redrawtime = plotfreq
    while True:
        data_a, data_b = next(gp)

        # Expand data if we hit the limit
        if i % redrawtime == 0:
            plotfreq, redrawtime, y = expand_data_capacity(
                i, plotfreq, redrawtime, y, lns, axes, num_actions
            )

        # Update data and artists
        append_new_data(i, data_a, data_b, y)
        update_artists(i, plotfreq, redrawtime, y, lns, bm)

        i += 1


def main():
    # change the game as needed
    g = game.shapley(FullInfoRegretMatchingActions)
    plot(g)


if __name__ == "__main__":
    main()
