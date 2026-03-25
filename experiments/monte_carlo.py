"""Sample a bunch of strategies from repeated games and plot the average and final strategies.
THIS ONLY WORKS WITH 3 ACTIONS FOR NOW."""

from game_theory.full_information_regret_matching import FullInfoRegretMatchingActions
import game_theory.game as game

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# --- 1. EXPERIMENT RUNNER ---

def run_experiments(game_fn, num_runs, num_steps, randomize=True):
    """
    Runs N independent Monte Carlo simulations of the game for T timesteps.
    Returns the average and final strategies for both players across all runs, 
    as well as the empirical joint action frequencies.
    """
    print(f"Running {num_runs} experiments for {num_steps} steps each...")

    # Pre-allocate arrays (Shape: N runs x 3 actions)
    avg_strats_A = np.zeros((num_runs, 3))
    avg_strats_B = np.zeros((num_runs, 3))
    fin_strats_A = np.zeros((num_runs, 3))
    fin_strats_B = np.zeros((num_runs, 3))

    # Track joint action frequencies (3x3 grid)
    joint_counts_avg = np.zeros((3, 3))
    joint_counts_fin = np.zeros((3, 3))

    for i in range(num_runs):
        g = game_fn(FullInfoRegretMatchingActions)
        g.reset("randomize", "randomize")

        for step in range(num_steps):
            g.update()
            
            # Use np.outer on the one-hot vectors to create a 3x3 matrix with a single '1', 
            # and add it directly to our cumulative counts.
            joint_counts_avg += np.outer(g.a.action, g.b.action)
            
            if step == num_steps - 1:
                joint_counts_fin += np.outer(g.a.action, g.b.action)

        # Store the historical average strategy
        avg_strats_A[i] = g.a.average_strategy
        avg_strats_B[i] = g.b.average_strategy

        # Store the exact strategy at the final timestep
        fin_strats_A[i] = g.a.p
        fin_strats_B[i] = g.b.p

        if (i + 1) % max(1, (num_runs // 10)) == 0:
            print(f"Progress: {i + 1}/{num_runs} runs completed.")

    # Normalize joint counts into probabilities
    joint_dist_avg = joint_counts_avg / (num_runs * num_steps)
    joint_dist_fin = joint_counts_fin / num_runs

    return avg_strats_A, avg_strats_B, fin_strats_A, fin_strats_B, joint_dist_avg, joint_dist_fin


# --- 2. MATH & PROJECTION ---

def project_to_ternary(strategies):
    """
    Projects a 3D probability distribution (p0, p1, p2)
    onto 2D Cartesian coordinates (x, y) for an equilateral triangle.
    """
    p1 = strategies[:, 1]
    p2 = strategies[:, 2]

    x = p1 + 0.5 * p2
    y = (np.sqrt(3) / 2) * p2
    return x, y


# --- 3. VISUALIZATION FUNCTIONS ---

def setup_dashboard_figure():
    """Creates a 2x3 layout for Average vs Final strategies + Heatmaps."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    return fig, axes

def format_ternary_axis(ax, title):
    """Draws the boundaries and labels of the ternary simplex."""
    h = np.sqrt(3) / 2
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], "k-", lw=1.5, alpha=0.8)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(0.5, 1.3, title, transform=ax.transAxes, 
        ha="center", va="top", fontsize=11, fontweight='semibold')
    
    ax.text(0, -0.05, "Action 0", ha="center", va="top")
    ax.text(1, -0.05, "Action 1", ha="center", va="top")
    ax.text(0.5, h + 0.05, "Action 2", ha="center", va="bottom")

def plot_density(ax, x, y, colormap):
    """
    Calculates the 2D Kernel Density Estimate (KDE) and plots it as a contour map,
    overlaying the actual sampled points.
    """
    ax.scatter(x, y, color="black", alpha=0.2, s=15, edgecolors="none", zorder=2)

    xy = np.vstack([x, y])
    try:
        kde = gaussian_kde(xy)
    except np.linalg.LinAlgError:
        print("Warning: Data is entirely singular (no variance). Skipping KDE contours.")
        return

    h = np.sqrt(3) / 2
    grid_x, grid_y = np.mgrid[0:1:200j, 0:h:200j]

    mask = (grid_y <= np.sqrt(3) * grid_x) & (grid_y <= np.sqrt(3) * (1 - grid_x))

    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
    z = np.zeros(grid_x.size)
    z[mask.ravel()] = kde(grid_coords[:, mask.ravel()])
    z = z.reshape(grid_x.shape)

    ax.contourf(
        grid_x, grid_y, z,
        levels=15, cmap=colormap, alpha=0.6, antialiased=True, zorder=1
    )

def plot_heatmap(ax, data, title, colormap="Purples"):
    """Plots a 3x3 heatmap of the joint action profiles."""
    cax = ax.imshow(data, cmap=colormap, vmin=0, vmax=np.max(data))
    ax.text(0.5, 1.3, title, transform=ax.transAxes, 
            ha="center", va="top", fontsize=11, fontweight='semibold')
    
    # Configure axes
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel("Player B Action")
    ax.set_ylabel("Player A Action")
    
    # Add percentage annotations
    for i in range(3):
        for j in range(3):
            val = data[i, j]
            # Change text color based on background intensity for readability
            text_color = "white" if val > np.max(data) / 2 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", color=text_color)


# --- 4. MAIN CONTROLLER ---

def main():
    NUM_RUNS = 250
    NUM_STEPS = 50

    game_fn = game.action_price

    # 1. Run the experiments
    avg_A, avg_B, fin_A, fin_B, joint_avg, joint_fin = run_experiments(game_fn, NUM_RUNS, NUM_STEPS)

    # 2. Project marginal data
    xa_avg, ya_avg = project_to_ternary(avg_A)
    xb_avg, yb_avg = project_to_ternary(avg_B)
    xa_fin, ya_fin = project_to_ternary(fin_A)
    xb_fin, yb_fin = project_to_ternary(fin_B)

    # 3. Setup dashboard
    fig, axes = setup_dashboard_figure()
    ax_A_avg, ax_B_avg, ax_joint_avg = axes[0, 0], axes[0, 1], axes[0, 2]
    ax_A_fin, ax_B_fin, ax_joint_fin = axes[1, 0], axes[1, 1], axes[1, 2]

    # 4. Plot Average Strategies (Top Row)
    format_ternary_axis(ax_A_avg, f"Player A: AVERAGE strategy")
    plot_density(ax_A_avg, xa_avg, ya_avg, colormap="Blues")

    format_ternary_axis(ax_B_avg, f"Player B: AVERAGE strategy")
    plot_density(ax_B_avg, xb_avg, yb_avg, colormap="Reds")
    
    plot_heatmap(ax_joint_avg, joint_avg, "Empirical Joint Profile\n(All Steps Average)", colormap="Purples")

    # 5. Plot Final Strategies (Bottom Row)
    format_ternary_axis(ax_A_fin, f"Player A: FINAL strategy")
    plot_density(ax_A_fin, xa_fin, ya_fin, colormap="Blues")

    format_ternary_axis(ax_B_fin, f"Player B: FINAL strategy")
    plot_density(ax_B_fin, xb_fin, yb_fin, colormap="Reds")
    
    plot_heatmap(ax_joint_fin, joint_fin, "Empirical Joint Profile\n(Final Step Only)", colormap="Purples")

    g = game_fn(FullInfoRegretMatchingActions)
    title_text = f"{g.name} game with random initial strategies"
    subtitle_text = f"Total Runs: {NUM_RUNS} | Steps per Run: {NUM_STEPS} | Algorithm: Regret Matching"

    # Main Title
    fig.suptitle(title_text, fontsize=20, fontweight='bold', y=0.98)

    # Subtitle (placed slightly lower)
    fig.text(0.5, 0.93, subtitle_text, ha='center', fontsize=12, color='dimgray')

    # Adjust layout to prevent the title from overlapping the top plots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()