import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from typing import Tuple
import multiprocessing as mp
from scipy.spatial import Voronoi, KDTree

from game_theory.full_information_regret_matching import FullInfoRegretMatchingStrategies
import game_theory.game as game

GLOBAL_STEP = 0.05
PLAYER_A_COLOR = "blue"
PLAYER_A_CMAP = "Blues"
PLAYER_B_COLOR = "red"
PLAYER_B_CMAP = "Reds"


def plot_player_triangle(ax, X, Y, U, V, Mag, title: str):
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], "k-", lw=1.5, alpha=0.8) # Thinner line to match MC

    # Consistent font sizing with MC plots
    ax.text(0, -0.05, "Action 0", ha="center", va="top", fontsize=9)
    ax.text(1, -0.05, "Action 1", ha="center", va="top", fontsize=9)
    ax.text(0.5, np.sqrt(3) / 2 + 0.05, "Action 2", ha="center", va="bottom", fontsize=9)

    ax.streamplot(
        X, Y, U, V,
        color=PLAYER_A_COLOR if "Player A" in title else PLAYER_B_COLOR,
        linewidth=1,
        arrowsize=1.2,
        density=1.5,
    )

    ax.text(0.5, 1.2, title, transform=ax.transAxes, 
            ha="center", va="top", fontsize=11, fontweight='semibold')

    ax.axis("off")
    ax.set_aspect("equal")

def barycentric_to_cartesian(p: np.ndarray) -> Tuple[float, float]:
    x = p[1] + 0.5 * p[2]
    y = (np.sqrt(3) / 2) * p[2]
    return x, y


def cartesian_to_barycentric(x: float, y: float) -> np.ndarray:
    p2 = y / (np.sqrt(3) / 2)
    p1 = x - 0.5 * p2
    p0 = 1 - p1 - p2
    return np.array([p0, p1, p2])


def run_simulation_local(
    g, pA_start: np.ndarray, pB_start: np.ndarray, iterations: int = 100
):
    g.reset(pA_start, pB_start)
    final_state = next(g.selfplay(num_iterations=iterations))
    for state in g.selfplay(num_iterations=iterations):
        final_state = state
    (pA, max_extA, max_intA, avg_stratA), (pB, max_extB, max_intB, avg_stratB) = (
        final_state
    )
    return pA, pB


def compute_flow_task(
    game_instance, player: str, opp_strat_tuple: tuple, grid_size: int
):
    """The heavy lifting, isolated for the background process."""
    opp_strat = np.array(opp_strat_tuple)
    X, Y = np.meshgrid(
        np.linspace(0, 1, grid_size), np.linspace(0, np.sqrt(3) / 2, grid_size)
    )

    U, V = np.zeros_like(X), np.zeros_like(Y)
    mask = np.zeros_like(X, dtype=bool)

    for i in range(grid_size):
        for j in range(grid_size):
            p_init = cartesian_to_barycentric(X[i, j], Y[i, j])
            if np.all(p_init >= -0.01):
                p_init = np.clip(p_init, 0, 1) / np.clip(p_init, 0, 1).sum()
                if player == "A":
                    final_A, _ = run_simulation_local(game_instance, p_init, opp_strat)
                    dp = final_A - p_init
                else:
                    _, final_B = run_simulation_local(game_instance, opp_strat, p_init)
                    dp = final_B - p_init
                du, dv = barycentric_to_cartesian(dp)
                U[i, j], V[i, j] = du, dv
            else:
                mask[i, j] = True

    mag = np.sqrt(U**2 + V**2)
    return (
        X,
        Y,
        np.ma.masked_array(U, mask=mask),
        np.ma.masked_array(V, mask=mask),
        np.ma.masked_array(mag, mask=mask),
    )


def background_worker(game_instance, shared_cache, grid_size, step):
    """The function that runs inside the dedicated background process."""
    steps = np.round(np.arange(0, 1.01, step), 3)

    for p0 in steps:
        for p1 in steps:
            if p0 + p1 <= 1.0:
                strat = (p0, p1, round(1.0 - p0 - p1, 3))
                for p in ["A", "B"]:
                    key = (p, strat)
                    if key not in shared_cache:
                        # Compute and store in the inter-process dictionary
                        shared_cache[key] = compute_flow_task(
                            game_instance, p, strat, grid_size
                        )


# ==========================================
# ENGINE AND UI
# ==========================================


class FlowEngine:
    def __init__(self, game_instance, grid_size=25):
        self.g = game_instance
        self.grid_size = grid_size

        self.manager = mp.Manager()
        self.shared_cache = self.manager.dict()
        self.worker_process = None

    def get_flow_data(self, player: str, opp_strat_tuple: tuple):
        key = (player, opp_strat_tuple)
        if key in self.shared_cache:
            return self.shared_cache[key]

        # Synchronous fallback
        data = compute_flow_task(self.g, player, opp_strat_tuple, self.grid_size)
        self.shared_cache[key] = data
        return data

    def start_background_precompute(self):
        self.worker_process = mp.Process(
            target=background_worker,
            args=(self.g, self.shared_cache, self.grid_size, GLOBAL_STEP),
            daemon=True,
        )
        self.worker_process.start()


class SimplexControl:
    """A draggable 2D simplex that shows coordinates on hover, Voronoi cells, and precomputation status."""

    def __init__(self, ax, title, player_to_update, engine, step_size):
        self.ax = ax
        self.title = title
        self.player_to_update = player_to_update
        self.engine = engine
        self.step_size = step_size

        self.strats = []
        self.points = []
        self.on_change_callback = None

        # State tracking
        self.is_pressed = False  # New: track if mouse button is held

        self._setup_grid()
        self._draw_voronoi()

        self.kdtree = KDTree(self.points)
        # Start roughly in the middle
        self.current_idx = self.kdtree.query(
            barycentric_to_cartesian((0.33, 0.33, 0.33))
        )[1]
        self.last_cache_size = -1

        # UI elements
        (self.marker,) = self.ax.plot(
            [], [], "o", color="white", markeredgecolor="black", markersize=8, zorder=10
        )

        # New: Hover Annotation (Tooltip)
        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8, ec="gray"),
            arrowprops=dict(arrowstyle="->"),
            zorder=20,
        )
        self.annot.set_visible(False)  # Hide initially

        self._update_marker()
        self.ax.text(0.5, 1.15, self.title, transform=self.ax.transAxes, 
             ha="center", va="top", fontsize=10, fontweight='semibold')
        subtitle_text = f"Click or drag to update strategies"
        self.ax.text(0.5, 0.93, subtitle_text, ha='center', fontsize=8, color='dimgray')
        
        # button_press handles initial click and initiates dragging
        self.ax.figure.canvas.mpl_connect("button_press_event", self.on_click)
        # motion_notify handles both hovering (tooltips) and dragging movement
        self.ax.figure.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        # button_release ends dragging
        self.ax.figure.canvas.mpl_connect("button_release_event", self.on_release)

        # Timer for background cache updates (shading cells)
        self.timer = self.ax.figure.canvas.new_timer(interval=500)
        self.timer.add_callback(self.poll_cache_and_update_colors)
        self.timer.start()

    def _setup_grid(self):
        steps = np.round(np.arange(0, 1.01, self.step_size), 3)
        for p0 in steps:
            for p1 in steps:
                if p0 + p1 <= 1.0:
                    p2 = round(1.0 - p0 - p1, 3)
                    self.strats.append((p0, p1, p2))
                    self.points.append(barycentric_to_cartesian((p0, p1, p2)))
        self.points = np.array(self.points)

    def _draw_voronoi(self):
        dummy_points = np.array([[-3, -3], [4, -3], [0.5, 4]])
        all_pts = np.vstack([self.points, dummy_points])
        vor = Voronoi(all_pts)

        patches = []
        for i in range(len(self.points)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            polygon = [vor.vertices[v] for v in region]
            patches.append(Polygon(polygon, closed=True))

        self.collection = PatchCollection(
            patches, edgecolors="white", linewidths=0.5, zorder=1
        )
        self.ax.add_collection(self.collection)

        # --- THE FIX: Clip the Voronoi cells to the triangle exactly ---
        triangle_pts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
        clip_poly = Polygon(triangle_pts, transform=self.ax.transData)
        self.collection.set_clip_path(clip_poly)

        # Draw the simplex outline on top
        triangle_outline = np.vstack([triangle_pts, triangle_pts[0]])
        self.ax.plot(
            triangle_outline[:, 0], triangle_outline[:, 1], "k-", lw=2, zorder=5
        )

        # --- THE FIX: Lock the camera/axes to the triangle so it doesn't zoom out ---
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(-0.05, np.sqrt(3) / 2 + 0.05)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        self.colors = np.zeros(len(self.points))
        self.collection.set_array(self.colors)
        self.collection.set_cmap(PLAYER_B_CMAP if self.player_to_update == "A" else PLAYER_A_CMAP)
        self.collection.set_clim(0, 1)

    def on_click(self, event):
        """Initiates dragging and updates point."""
        if event.inaxes != self.ax:
            return
        self.is_pressed = True  # Start drag state
        self._handle_point_update(event.xdata, event.ydata)

    def on_release(self, event):
        """Ends dragging state."""
        self.is_pressed = False

    def on_mouse_move(self, event):
        """Handles both dragging (updates plot) and hovering (shows tooltip)."""
        if event.inaxes != self.ax:
            # If mouse leaves the simplex area, hide the tooltip
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.ax.figure.canvas.draw_idle()
            return

        # Always find nearest point for hovering/tooltips
        _, idx = self.kdtree.query([event.xdata, event.ydata])
        strat = self.strats[idx]

        # --- PART 1: Dragging (Slide functionality) ---
        if self.is_pressed:
            # If button is held, update the selected point and the main flow plot
            self._handle_point_update(event.xdata, event.ydata)

        # --- PART 2: Hover Animation (Tooltip functionality) ---
        # Get cartesian center point of Voronoi cell for tooltip placement
        cell_center_xy = self.points[idx]

        # Update tooltip content and position
        self.annot.xy = cell_center_xy
        # Round the strategy coordinates for nice display
        p = np.round(strat, 2)
        text = f"p0: {p[0]:.2f}\np1: {p[1]:.2f}\np2: {p[2]:.2f}"
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_alpha(0.8)  # set opacity
        self.annot.set_visible(True)

        # Need to redraw to show the updated tooltip
        self.ax.figure.canvas.draw_idle()

    def _handle_point_update(self, x, y):
        """Helper to snap to nearest valid point and trigger flow updates."""
        _, idx = self.kdtree.query([x, y])

        # Only trigger update if we actually moved to a *different* cell center
        if self.current_idx != idx:
            self.current_idx = idx
            self._update_marker()
            if self.on_change_callback:
                # This calls update_A or update_B in the main script, recomputing the flow.
                self.on_change_callback(self.strats[idx])

    def _update_marker(self):
        x, y = self.points[self.current_idx]
        self.marker.set_data([x], [y])
        # We don't call draw_idle here because it's called immediately after in on_mouse_move

    def poll_cache_and_update_colors(self):
        """Timer callback: Checks background progress and shades completed cells."""
        current_cache_size = len(self.engine.shared_cache)
        if current_cache_size == self.last_cache_size:
            return

        self.last_cache_size = current_cache_size

        # Map cache status to color intensity
        for i, strat in enumerate(self.strats):
            key = (self.player_to_update, strat)
            self.colors[i] = 0.6 if key in self.engine.shared_cache else 0.1

        self.collection.set_array(self.colors)
        # Note: Polling updates are redraws. Hovering/dragging also trigger redraws.
        # This is generally fast enough if the FlowEngine cache is populated.
        self.ax.figure.canvas.draw_idle()

    def get_current_strat(self):
        return self.strats[self.current_idx]


def plot_interactive_dual_ternary_flow(g):
    engine = FlowEngine(g)
    engine.start_background_precompute()

    # Use the same figure size and layout logic as the MC dashboard
    fig = plt.figure(figsize=(16, 9))
    
    # 1. Main Title (Bold)
    title_text = f"Strategy flow for game: {g.name}"
    fig.suptitle(title_text, fontsize=20, fontweight='bold', y=0.98)

    # 2. Subtitle (Gray)
    subtitle_text = f"Algorithm: Regret Matching | Interaction: Drag bottom triangles to update opponent strategies"
    fig.text(0.5, 0.93, subtitle_text, ha='center', fontsize=12, color='dimgray')

    # Define axes with better spacing for titles
    ax1 = fig.add_axes([0.05, 0.42, 0.4, 0.40])
    ax2 = fig.add_axes([0.55, 0.42, 0.4, 0.40])
     
    # Control Simplexes
    ax_ctrl_A = fig.add_axes([0.15, 0.05, 0.2, 0.25])
    ax_ctrl_B = fig.add_axes([0.65, 0.05, 0.2, 0.25])

    ctrl_A = SimplexControl(ax_ctrl_A, "Player B Initial Strategy", "A", engine, GLOBAL_STEP)
    ctrl_B = SimplexControl(ax_ctrl_B, "Player A Initial Strategy", "B", engine, GLOBAL_STEP)

    def update_A(strat_B):
        X, Y, U, V, Mag = engine.get_flow_data("A", strat_B)
        ax1.clear()
        p = np.round(strat_B, 2).tolist()
        plot_player_triangle(ax1, X, Y, U, V, Mag, f"Player A: Strategy Flow\n(vs initial B: {p})")
        fig.canvas.draw_idle()

    def update_B(strat_A):
        X, Y, U, V, Mag = engine.get_flow_data("B", strat_A)
        ax2.clear()
        p = np.round(strat_A, 2).tolist()
        plot_player_triangle(ax2, X, Y, U, V, Mag, f"Player B: Strategy Flow\n(vs initial A: {p})")
        fig.canvas.draw_idle()

    ctrl_A.on_change_callback = update_A
    ctrl_B.on_change_callback = update_B

    update_A(ctrl_A.get_current_strat())
    update_B(ctrl_B.get_current_strat())

    plt.show()


if __name__ == "__main__":
    mp.freeze_support()  # Good practice if we ever compile to an executable
    g = game.action_price(FullInfoRegretMatchingStrategies)
    plot_interactive_dual_ternary_flow(g)
