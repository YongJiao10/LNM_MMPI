import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
from collections import OrderedDict
from matplotlib.patches import Arc, PathPatch, Wedge, Patch
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, ListedColormap, to_hex
# from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go
import plotly.io as pio
from utils import *


NET = ['VN', 'SMN', 'DAN', 'VAN', 'LN', 'FPCN', 'DMN']
# PALLETE = [(.467, .204, .506), (.388, .588, .761), (.173, .494, .067), (.863, .361, 1), 
#                 (.541, .278, .094), (.969, .675, .29), (.847, .361, .42)] # colors for Functional Networks
# PALLETE = ["#773481", "#6396c2", "#2c7e11", "#dc5cff", 
#            "#8a4718", "#f7ac4a", "#d85c6b"] # colors for Functional Networks

PALLETE = ["#4A3432", "#5E5B8A", "#4F8FB5", "#83934E",
           "#A25C8A", "#8a4718", "#C79A3B", "#B24A43"] # adopted: SC, VN, SMN, DAN, VAN, LN, FPCN, DMN


# DIMENSION = ['Neurologically-Linked Despair', 'Cognitive Strain-Disaffiliation']
DIMENSION = ['Somato-Cognitive Compromise', 'Social-Behavioral Dysregulation']

SUBTYPE_COLOR = ['#c0158f', '#158fc0', '#8fc015'] # purple, green, blue
SCALE_COLOR = ['#1a7097', '#97411a', '#7f1a97', '#31971a']   # navy, brown, purple / cognition, mental health, personality
DIMENSION_COLOR = ['#835bd6', '#c8971c']  # purple, yellow

# MULTI_COLOR = sns.color_palette()
MULTI_COLOR = ['#459395', '#866f85', '#fda638', '#6b93c6', '#EB7C69',  "#bd7dc4", '#4ca866'] # green, orange, blue, red, purple

PAIR_COLOR = [['#374E55', '#B2768A'], 
              ['#00A1D5', '#B24745'],  
              ['#79AF97', '#6A6599']]  

PANEL_TITLE_STYLE = {
    'fontsize': 24,
    'fontweight': 'bold',
    'fontfamily': plt.rcParamsDefault['font.sans-serif'][0],
    'color': 'black',
}
SQUARE_PANEL_SIZE = (8.4, 8.4)

TEXT_COLOR = {
    'Cognition': SCALE_COLOR[0],
    'MH': SCALE_COLOR[1],
    'Personality': SCALE_COLOR[2],
    'Environment': SCALE_COLOR[3],
}
SUBTYPE_COLORMAP = {
    'subtype 1': [(0, '#ffcaef'), (0.5, '#e17dc0'), (1, SUBTYPE_COLOR[0])],
    'subtype 2': [(0, '#9fe5ff'), (0.5, '#64b9df'), (1, SUBTYPE_COLOR[1])],
    'subtype 3': [(0, '#d5e3aa'), (0.5, '#b4d269'), (1, SUBTYPE_COLOR[2])],
}

def get_significance_symbol(p_value: float) -> str:
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


def plot_permutation_hist(
    r_perm,
    r_real,
    idx,
    color="#1f77b4",
    bins=20,
    save_path=None,
    figsize=(4, 4),
    font_delta=12,
    transparent=True,
):
    """
    Shared permutation histogram plot used across read_rcca notebooks.

    Parameters
    ----------
    r_perm : ndarray, shape (n_perm, eval_dim)
        Permutation correlation matrix.
    r_real : ndarray, shape (eval_dim,)
        Real CV correlation vector.
    idx : int
        Canonical dimension index.
    color : str
        Histogram color.
    bins : int
        Histogram bins.
    save_path : str or None
        Target path stem (without extension). Save as PDF when provided.
    figsize : tuple
        Matplotlib figure size.
    font_delta : int
        Font size increment over base settings.
    transparent : bool
        Whether to save figure with transparent background.
    """
    vals = r_perm[:, idx]
    real = r_real[idx]

    p_fontsize = 12 + font_delta
    tick_fontsize = 11 + font_delta
    xlabel_fontsize = 14 + font_delta

    # Right-tail permutation p-value.
    p_val = (np.sum(vals >= real) + 1) / (len(vals) + 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(vals, bins=bins, color=color, alpha=0.7, edgecolor="none")
    ax.axvline(real, color="red", linestyle="--", linewidth=2)

    p_label = rf"$\mathit{{P}}_{{\mathrm{{permutation}}}}={p_val:.3f}$"

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_yticks([])

    current_ticks = ax.get_xticks()
    x_min, x_max = ax.get_xlim()
    visible_ticks = current_ticks[(current_ticks >= x_min - 1e-12) & (current_ticks <= x_max + 1e-12)]
    neg_ticks = visible_ticks[visible_ticks < -1e-12]
    pos_ticks = visible_ticks[visible_ticks > 1e-12]

    if neg_ticks.size > 0:
        left_tick = np.min(neg_ticks)
    elif pos_ticks.size > 0:
        left_tick = -np.max(pos_ticks)
    else:
        left_tick = -np.max(np.abs(ax.get_xlim()))

    if pos_ticks.size > 0:
        right_tick = np.max(pos_ticks)
    elif neg_ticks.size > 0:
        right_tick = -np.min(neg_ticks)
    else:
        right_tick = np.max(np.abs(ax.get_xlim()))

    if np.isclose(left_tick, 0.0) and np.isclose(right_tick, 0.0):
        left_tick, right_tick = -1.0, 1.0

    ax.set_xticks([left_tick, 0.0, right_tick])
    ax.set_xticklabels([f"{left_tick:.1f}", "0", f"{right_tick:.1f}"])
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_xlabel(r"$r_{cv}$", fontsize=xlabel_fontsize)

    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.19, top=0.83)
    fig.text(
        0.5,
        0.95,
        p_label,
        ha="center",
        va="top",
        fontsize=p_fontsize,
        clip_on=False,
    )

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(f"{save_path}.pdf", bbox_inches="tight", transparent=transparent)

    return fig, ax


def plot_net_ring(
    name_value,
    full_scale=0.15,
    palette=PALLETE,
    network_order=("DMN", "FPCN", "LN", "VAN", "DAN", "SMN", "VN", "SC"),
    title="Average\nbrain loading",
    bg_color="none",
    track_color="#cfd4d5",
    start_angle=90,
    clockwise=True,
    linewidth=16,
    radius_outer=1.0,
    radius_step=0.10,
    label_mode="arc_start_abbr",
    value_fmt="{:.2f}",
    show_values=False,
    clip_overflow=True,
    auto_optimize=True,
    optimize_iters=8,
    value_fontsize=None,
    network_fontsize=None,
    title_fontsize=None,
    ax=None,
    ring_width=None,
    ring_gap=None,
    outer_radius=None,
    show_track=False,
    abbr_color="black",
    value_color="black",
):
    """
    Plot SC+7 ring loading chart (outer->inner fixed order by ``network_order``).

    Parameters
    ----------
    name_value : dict or sequence of (name, value)
        Network names and values. Supports aliases such as:
        Default mode->DMN, Frontoparietal->FPCN, Visual->VN, SC/Subcortical->SC.
    full_scale : float
        Value corresponding to a full 360-degree ring. Example: 0.15 means 0.15 == full circle.
    show_values : bool
        Whether to show numeric value labels near arc ends. Default False.
    clip_overflow : bool
        If True, values above ``full_scale`` are clipped to full ring.
        If False, values above ``full_scale`` raise ValueError.
    ring_width, ring_gap, outer_radius : float or None
        New geometry controls in data coordinates. If None, inferred from legacy
        ``linewidth`` and ``radius_step`` for backward compatibility.
    show_track : bool
        If True, draws background tracks for each ring. Default False.

    Returns
    -------
    fig, ax, ordered_values
        Matplotlib figure/axes and ordered values as OrderedDict.
    """
    canonical_set = {"DMN", "FPCN", "LN", "VAN", "DAN", "SMN", "VN", "SC"}
    ordered_networks = tuple(str(n).upper() for n in network_order)

    if set(ordered_networks) != canonical_set or len(ordered_networks) != 8:
        raise ValueError(
            "network_order must contain exactly one of each: "
            "DMN, FPCN, LN, VAN, DAN, SMN, VN, SC."
        )
    if not np.isfinite(full_scale) or full_scale <= 0:
        raise ValueError(f"full_scale must be a positive finite number, got {full_scale}.")
    if outer_radius is None:
        outer_radius = radius_outer
    if not np.isfinite(outer_radius) or outer_radius <= 0:
        raise ValueError(f"outer_radius must be positive finite, got {outer_radius}.")
    if not np.isfinite(linewidth) or linewidth <= 0:
        raise ValueError(f"linewidth must be a positive finite number, got {linewidth}.")
    if not np.isfinite(radius_step) or radius_step <= 0:
        raise ValueError(f"radius_step must be a positive finite number, got {radius_step}.")
    if palette is None:
        raise ValueError("palette cannot be None.")

    def _normalize_name(name: str) -> str:
        key = str(name).strip().lower()
        key = re.sub(r"[\s_\-/]+", "", key)
        key = re.sub(r"[^a-z0-9]", "", key)
        return key

    def build_network_color_map_from_brain_order(palette_):
        brain_order = tuple(globals().get("NETWORK_ORDER_BRAIN", ['SC', 'VN', 'SMN', 'DAN', 'VAN', 'LN', 'FPCN', 'DMN']))
        if len(palette_) < len(brain_order):
            raise ValueError(
                f"Brain network palette requires >= {len(brain_order)} colors, got {len(palette_)}."
            )
        return {net: palette_[i] for i, net in enumerate(brain_order)}

    def compute_ring_geometry(n_rings, outer_r, ring_w, ring_g):
        geoms = []
        r_outer = outer_r
        for _ in range(n_rings):
            r_inner = r_outer - ring_w
            if r_inner <= 0:
                raise ValueError(
                    "Ring geometry invalid (non-positive inner radius). "
                    "Reduce ring_width/ring_gap or increase outer_radius."
                )
            geoms.append((r_outer, r_inner, 0.5 * (r_outer + r_inner)))
            r_outer = r_inner - ring_g
        return geoms

    def value_to_span(value, scale, allow_clip):
        try:
            value = float(value)
        except Exception as exc:
            raise ValueError(f"Non-numeric value: {value!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"Value must be finite, got {value}.")
        if value < 0:
            raise ValueError(f"Value must be non-negative, got {value}.")
        ratio = value / scale
        if allow_clip:
            ratio = float(np.clip(ratio, 0.0, 1.0))
        elif ratio > 1:
            raise ValueError(
                f"Value {value:.6g} exceeds full_scale={scale:.6g}. "
                "Set clip_overflow=True to clip to one full ring."
            )
        return ratio, 360.0 * ratio

    alias_map = {
        "DMN": ["dmn", "defaultmode", "defaultmodenetwork"],
        "FPCN": ["fpcn", "frontoparietal", "frontoparietalnetwork"],
        "LN": ["ln", "limbic", "limbicnetwork"],
        "VAN": ["van", "ventralattention", "ventralattentionnetwork"],
        "DAN": ["dan", "dorsalattention", "dorsalattentionnetwork"],
        "SMN": ["smn", "somatomotor", "somatomotornetwork", "sensorimotor"],
        "VN": ["vn", "visual", "visualnetwork"],
        "SC": ["sc", "subcortical", "subcortex", "subcorticalnetwork"],
    }
    alias_to_canonical = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            alias_to_canonical[_normalize_name(alias)] = canonical

    if isinstance(name_value, dict):
        raw_items = list(name_value.items())
    elif isinstance(name_value, (list, tuple)):
        raw_items = []
        for i, pair in enumerate(name_value):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"name_value[{i}] must be a pair like ('DMN', 0.12), got: {pair!r}")
            raw_items.append((pair[0], pair[1]))
    else:
        raise ValueError("name_value must be a dict or a sequence of (name, value) pairs.")

    mapped_values = {}
    unknown_names = []
    for raw_name, raw_value in raw_items:
        canonical = alias_to_canonical.get(_normalize_name(raw_name))
        if canonical is None:
            unknown_names.append(str(raw_name))
            continue
        if canonical in mapped_values:
            raise ValueError(f"Duplicate network after alias mapping: {canonical} (from {raw_name!r}).")
        mapped_values[canonical] = raw_value

    if unknown_names:
        raise ValueError(
            "Unknown network name(s): "
            + ", ".join(unknown_names)
            + ". Supported canonical names: "
            + ", ".join(ordered_networks)
            + "."
        )

    missing = [n for n in ordered_networks if n not in mapped_values]
    if missing:
        raise ValueError(f"Missing network(s): {', '.join(missing)}.")

    if ring_width is None:
        # Default wider rings so the center hole is smaller in the common 8-ring layout.
        ring_width = min(radius_step, max(0.045, 0.0055 * linewidth))
    if ring_gap is None:
        # Exact tangency: adjacent rings share one boundary (no gap, no overlap).
        ring_gap = 0.0
    if not np.isfinite(ring_width) or ring_width <= 0:
        raise ValueError(f"ring_width must be positive finite, got {ring_width}.")
    if not np.isfinite(ring_gap) or ring_gap < 0:
        raise ValueError(f"ring_gap must be non-negative finite, got {ring_gap}.")

    n_rings = len(ordered_networks)
    min_hole_radius = max(0.22, 1.8 * ring_width) if title else 0.06
    used_radius = n_rings * ring_width + (n_rings - 1) * ring_gap
    max_used = outer_radius - min_hole_radius
    if max_used <= 0:
        raise ValueError("outer_radius is too small for requested center hole.")
    if used_radius > max_used:
        scale = max_used / used_radius
        ring_width *= scale
        ring_gap *= scale

    geoms = compute_ring_geometry(n_rings, outer_radius, ring_width, ring_gap)
    inner_hole_radius = geoms[-1][1]

    network_color_map = build_network_color_map_from_brain_order(list(palette))
    ordered_values = OrderedDict((n, float(mapped_values[n])) for n in ordered_networks)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
    else:
        fig = ax.figure

    is_transparent_bg = (
        bg_color is None
        or (isinstance(bg_color, str) and bg_color.strip().lower() in {"none", "transparent"})
    )
    if is_transparent_bg:
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)
    else:
        fig.patch.set_facecolor(bg_color)
        fig.patch.set_alpha(1.0)
        ax.set_facecolor(bg_color)
        ax.patch.set_alpha(1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Compatibility knobs retained in signature but not used in current style pipeline.
    _ = auto_optimize, optimize_iters

    if value_fontsize is None:
        value_fontsize = int(np.clip(10 + 30 * ring_width, 10, 14)) - 1
    if network_fontsize is None:
        network_fontsize = int(np.clip(10 + 28 * ring_width, 10, 14)) + 1
    if title_fontsize is None:
        title_fontsize = int(np.clip(22 * inner_hole_radius + 8, 11, 20)) + 2

    direction = -1 if clockwise else 1
    text_offset = max(0.015, 0.30 * ring_width)
    abbr_clearance_pts = max(4.0, 30.0 * ring_width)
    value_end_clearance_pts = max(2.0, 10.0 * ring_width, 0.32 * value_fontsize)

    margin = max(0.08, 0.5 * ring_width + text_offset)
    lim = outer_radius + margin
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _points_to_pixels(points):
        return float(points) * fig.dpi / 72.0

    def _text_bbox_pixels(text_str, fontsize):
        ghost = ax.text(0.0, 0.0, text_str, ha="center", va="center", fontsize=fontsize, alpha=0.0)
        bbox = ghost.get_window_extent(renderer=renderer)
        ghost.remove()
        return float(bbox.width), float(bbox.height)

    def _arc_pixels_per_degree(radius, angle_deg):
        eps_deg = 0.5
        a0 = np.deg2rad(angle_deg)
        a1 = np.deg2rad(angle_deg + direction * eps_deg)
        p0 = np.array(ax.transData.transform((radius * np.cos(a0), radius * np.sin(a0))), dtype=float)
        p1 = np.array(ax.transData.transform((radius * np.cos(a1), radius * np.sin(a1))), dtype=float)
        dist = np.hypot(*(p1 - p0))
        return dist / eps_deg

    for i, network in enumerate(ordered_networks):
        value = ordered_values[network]
        outer_r, _, mid_r = geoms[i]
        _, span = value_to_span(value, full_scale, clip_overflow)

        theta_start = float(start_angle)
        theta_end = float(theta_start + direction * span)
        theta1, theta2 = (theta_end, theta_start) if clockwise else (theta_start, theta_end)

        if show_track:
            ax.add_patch(
                Wedge(
                    center=(0, 0),
                    r=outer_r,
                    theta1=0,
                    theta2=360,
                    width=ring_width,
                    facecolor=track_color,
                    edgecolor="none",
                    antialiased=False,
                )
            )

        if span > 0:
            ax.add_patch(
                Wedge(
                    center=(0, 0),
                    r=outer_r,
                    theta1=theta1,
                    theta2=theta2,
                    width=ring_width,
                    facecolor=network_color_map[network],
                    edgecolor="none",
                    antialiased=False,
                )
            )

        if label_mode not in ("none", None):
            # Abbreviation placement rule:
            # y aligned to ring midline at arc start; text right edge aligned to start x.
            rs = mid_r
            x_start = rs * np.cos(np.deg2rad(theta_start))
            y_start = rs * np.sin(np.deg2rad(theta_start))
            ax.annotate(
                network,
                xy=(x_start, y_start),
                xytext=(-abbr_clearance_pts, 0),
                textcoords="offset points",
                ha="right", va="center",
                fontsize=network_fontsize,
                color=abbr_color,
                clip_on=False,
            )

        if show_values:
            # Keep value label center exactly on ring centerline, then shift along
            # arc direction past arc-end by text-size-aware distance to avoid overlap.
            value_text = value_fmt.format(value)
            text_w_px, text_h_px = _text_bbox_pixels(value_text, value_fontsize)
            angle_rad = np.deg2rad(theta_end)
            tangent_data = direction * np.array([-np.sin(angle_rad), np.cos(angle_rad)], dtype=float)
            origin_disp = np.array(ax.transData.transform((0.0, 0.0)), dtype=float)
            tangent_disp = np.array(ax.transData.transform(tuple(tangent_data)), dtype=float) - origin_disp
            tangent_norm = np.hypot(tangent_disp[0], tangent_disp[1])
            if tangent_norm > 0:
                tangent_unit = tangent_disp / tangent_norm
            else:
                tangent_unit = np.array([1.0, 0.0], dtype=float)

            half_proj_px = 0.5 * (abs(tangent_unit[0]) * text_w_px + abs(tangent_unit[1]) * text_h_px)
            shift_px = half_proj_px + _points_to_pixels(value_end_clearance_pts)
            px_per_deg = _arc_pixels_per_degree(mid_r, theta_end)
            delta_deg = 0.0 if px_per_deg <= 1e-9 else shift_px / px_per_deg
            if span < 40.0:
                delta_deg += 0.45 * (40.0 - span)
            theta_value = theta_end + direction * delta_deg
            x_value = mid_r * np.cos(np.deg2rad(theta_value))
            y_value = mid_r * np.sin(np.deg2rad(theta_value))
            ax.text(
                x_value, y_value, value_text,
                ha="center", va="center",
                fontsize=value_fontsize,
                color=value_color,
                clip_on=False,
            )

    if title:
        ax.text(
            0, 0, title,
            ha="center", va="center",
            fontsize=title_fontsize,
            color="black",
            linespacing=1.1,
        )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    return fig, ax, ordered_values
        
def kde(x):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制第一个数据（散点图）
    sns.scatterplot(x=x, y=range(len(x)), ax=ax1, color='blue', label='Scatter')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Scatter', color='blue')

    # 创建第二个 y 轴
    ax2 = ax1.twinx()

    # 绘制第二个数据（KDE 曲线）
    sns.kdeplot(x, fill=True, ax=ax2, color='red', label='KDE')
    ax2.set_ylabel('KDE', color='red')

    # 显示图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, loc='upper left')

def visualize_pca_variance(arr, max_components=None, cumulative_threshold=0.9):
    """
    Visualize PCA variance explained by different dimensions to help select optimal n_components
    
    Parameters:
    -----------
    arr : numpy.ndarray
        Input data array with shape (n_samples, n_features)
    max_components : int, optional
        Maximum number of components to consider, defaults to n_features
    cumulative_threshold : float, optional
        Cumulative variance threshold line, default 0.95
    
    Returns:
    --------
    pca : PCA object
        Fitted PCA object
    """
    from sklearn.decomposition import PCA

    n_samples, n_features = arr.shape
    
    if max_components is None:
        max_components = min(n_samples, n_features)
    else:
        max_components = min(max_components, n_samples, n_features)
    
    # Fit PCA
    pca = PCA(n_components=max_components)
    pca.fit(arr)
    
    # Get variance explained ratios
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Individual variance explained by each component
    ax1.bar(range(1, len(explained_variance_ratio) + 1), 
            explained_variance_ratio, 
            alpha=0.7, 
            color='steelblue',
            edgecolor='gray')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Variance Explained Ratio', fontsize=12)
    ax1.set_title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right plot: Cumulative variance explained
    ax2.plot(range(1, len(cumulative_variance_ratio) + 1), 
             cumulative_variance_ratio, 
             'o-', 
             linewidth=2, 
             markersize=4,
             color='coral')
    ax2.axhline(y=cumulative_threshold, 
                color='red', 
                linestyle='--', 
                linewidth=2,
                label=f'{cumulative_threshold*100:.0f}% threshold')
    
    # Find minimum components needed to reach threshold
    n_components_threshold = np.argmax(cumulative_variance_ratio >= cumulative_threshold) + 1
    ax2.axvline(x=n_components_threshold, 
                color='green', 
                linestyle='--', 
                linewidth=2,
                alpha=0.7,
                label=f'n={n_components_threshold} components')
    ax2.scatter([n_components_threshold], 
                [cumulative_variance_ratio[n_components_threshold-1]], 
                s=100, 
                color='green', 
                zorder=5)
    
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Data shape: {arr.shape}")
    print(f"Max components: {max_components}")
    print(f"\nTop 10 components variance explained:")
    for i in range(min(10, len(explained_variance_ratio))):
        print(f"  PC{i+1}: {explained_variance_ratio[i]:.4f} (cumulative: {cumulative_variance_ratio[i]:.4f})")
    print(f"\nComponents needed for {cumulative_threshold*100:.0f}% variance: {n_components_threshold}")
    
    return pca
    

def custom_sci_format(value):
    exponent = int(math.floor(math.log10(abs(value))))
    coefficient = value / (10 ** exponent)
    return f"{coefficient:.2f} \\times 10^{{{exponent}}}"


def plot_correlation(X, Y, dims=[0, 1], dot_size=20, alpha=0.5, num_scales=1, x_label='SFC',
                    save_path=None, title=DIMENSION, colors=MULTI_COLOR):
    """
    Scatter + regression plots for specified dimensions.

    Parameters:
    -----------
    X : numpy.ndarray, shape (n_samples, n_dimensions)
    Y : numpy.ndarray, shape (n_samples, n_dimensions)
    dims : list of int
        Indices of dimensions to plot, e.g., [0, 3, 5]
    dot_size : int
        Marker size for scatter
    alpha : float
        Scatter alpha
    num_scales : int or float
        Scaling factor applied to X and Y (visual only)
    save_path : str or None
        If provided, saves as PDF at the given path (no extension)
    title : list or None
        Optional list of titles per dimension (global). Will be mapped by dims.
        If None, titles will be empty.
    colors : list
        Color palette per dimension index; color = colors[dim % len(colors)]
    """
    n_dims = len(dims)
    fig, axes = plt.subplots(1, n_dims, figsize=(7 * n_dims, 6))
    axes = [axes] if n_dims == 1 else axes

    # Map titles by dims (fallback to empty strings)
    if title is None:
        _titles = [''] * n_dims
    else:
        if isinstance(title, (list, tuple)) and len(title) >= (max(dims) + 1):
            _titles = [title[d] for d in dims]
        else:
            _titles = [''] * n_dims

    # Scale copies (do not modify inputs)
    Xp = num_scales * X
    Yp = num_scales * Y

    for i, dim in enumerate(dims):
        ax = axes[i]
        x = Xp[:, dim]
        y = Yp[:, dim]

        # Pearson r and Cohen's d (guard against r -> ±1)
        r, p_value = pearsonr(x, y)
        denom = max(1e-12, np.sqrt(max(0.0, 1 - r**2)))
        d = 2 * r / denom

        # Color by true dimension index
        c = colors[i]

        # Plot with regression
        sns.regplot(x=x, y=y, ax=ax, color=c,
                    scatter_kws={'s': dot_size, 'alpha': alpha})

        # Labels and title
        ax.set_title(_titles[i], fontsize=22)
        ax.set_xlabel(f'{x_label} score (a.u.)', fontsize=18)
        ax.set_ylabel('Behavior score (a.u.)', fontsize=18)
        ax.tick_params(labelsize=16)

        # Stat text
        if p_value < 1e-4:
            stat_text = (fr'$\mathit{{r}}={r:.3f}$' + '\n' +
                         r'$\mathit{p}<0.0001$' + '\n' +
                         fr'$\mathit{{d}}={d:.3f}$')
        else:
            stat_text = (fr'$\mathit{{r}}={r:.3f}$' + '\n' +
                         fr'$\mathit{{p}}={custom_sci_format(p_value)}$' + '\n' +
                         fr'$\mathit{{d}}={d:.3f}$')
        ax.text(0.65, 0.05, stat_text, transform=ax.transAxes, fontsize=18)

        # Clean style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_box_aspect(1)

    # Layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2, wspace=0.3)

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')



def plot_corr_with_label(X, Y, labels, n_dims=2, dot_size=10, alpha=0.3, num_scales=100, label_mapping={0: 'Male', 1: 'Female'},
                     title=DIMENSION, colors=SUBTYPE_COLOR, save_path=None):
    if (title is None) or (len(title) < n_dims):
        title = DIMENSION
    
    fig, axes = plt.subplots(1, n_dims, figsize=(7*n_dims, 6))
    axes = [axes] if n_dims == 1 else axes
    
    X = num_scales * X
    Y = num_scales * Y
    
    for i in range(n_dims):
        x = X[:, i]
        y = Y[:, i]
        
        # Get unique labels
        unique_labels = np.unique(labels)
        stats = {}
        diff_line = None
        
        for j, label in enumerate(unique_labels):
            # Get data for current label
            mask = labels == label
            x_group = x[mask]
            y_group = y[mask]
            n_group = int(mask.sum())
            
            # Calculate statistics for this group
            r, p_value = pearsonr(x_group, y_group)
            d = 2 * r / np.sqrt(1 - r**2) if abs(r) < 1 else np.nan
            stats[label] = (r, n_group)
            
            # Format p-value string
            if p_value < 0.0001:
                p_str = "$P<0.0001$"
            elif p_value > 0.01:
                p_str = f"$P={p_value:.3f}$"
            else:
                p_str = f"$P={custom_sci_format(p_value)}$"
            
            # Create label with statistics
            group_name = label_mapping.get(label, str(label))
            label_with_stats = f"{group_name} (N={n_group}, $r$={r:.3f}, {p_str}, $d$={d:.3f})"
            
            # Plotting with updated label
            sns.regplot(
                x=x_group, y=y_group, ax=axes[i],
                color=colors[j % len(colors)],
                scatter_kws={'s': dot_size, 'alpha': alpha},
                line_kws={'label': label_with_stats}
            )

        # Difference test between 2 groups (Fisher Z)
        if len(unique_labels) == 2:
            l1, l2 = unique_labels
            r1, n1 = stats[l1]
            r2, n2 = stats[l2]
            if (n1 > 3) and (n2 > 3) and np.isfinite(r1) and np.isfinite(r2) and (abs(r1) < 1) and (abs(r2) < 1):
                p_diff, sig = fisher_ztest(r1, r2, n1, n2)
                sig = sig.replace("N.S.", "ns")
                if p_diff < 1e-4:
                    p_diff_str = "$P<0.0001$"
                elif p_diff > 1e-2:
                    p_diff_str = f"$P={p_diff:.3f}$"
                else:
                    p_diff_str = f"$P={custom_sci_format(p_diff)}$"
                diff_line = f"Difference test (Fisher Z): {p_diff_str} ({sig})"
            else:
                diff_line = "Difference test (Fisher Z): N/A"
        
        axes[i].set_title(title[i], fontsize=22)
        axes[i].set_xlabel('Brain score (a.u.)', fontsize=18)
        axes[i].set_ylabel('Behavior score (a.u.)', fontsize=18)
        axes[i].tick_params(labelsize=16)
        
        # Rebuild legend and append Fisher Z row when available.
        handles, labels_list = axes[i].get_legend_handles_labels()
        if diff_line is not None:
            handles = list(handles)
            labels_list = list(labels_list)
            handles.append(plt.Line2D([], [], linestyle='None'))
            labels_list.append(diff_line)
        axes[i].legend(handles, labels_list, loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=False)
        
        # Remove top and right spines
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.1, wspace=0.4)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #plt.savefig(f'{save_path}.svg', bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')


# def align(U1, U2, along_symp=True):
#     if along_symp:
#         df = pd.DataFrame(U2)
#     else: df = pd.DataFrame(U1)
#     # 计算行间的相关性矩阵
#     corr_matrix = df.T.corr()
#     reverse_idx = corr_matrix[0] < 0
#     U1[reverse_idx, :] *= -1
#     U2[reverse_idx, :] *= -1
#     return U1, U2


def plot_corr(U1, U2, scale=1):
    fig, ax = plt.subplots(1, 2, figsize=(16*scale, 8*scale), gridspec_kw={'width_ratios': [1, 1]})

    # 计算并绘制U1的相关性矩阵
    df1 = pd.DataFrame(U1)
    corr_matrix1 = df1.T.corr()
    icc1 = compute_icc(U1)
    mask1 = np.triu(np.ones_like(corr_matrix1, dtype=bool), 1)
    sns.heatmap(corr_matrix1, mask=mask1, annot=True, fmt=".2f", cmap='Blues', 
                cbar=True, square=True, linewidths=.5, ax=ax[0], cbar_kws={'shrink': 0.5})
    ax[0].set_title(f'Correlation Across 10 Folds for FC loading', fontsize=14)
    ax[0].text(6, 3, f'ICC: {icc1}', verticalalignment='center', fontsize=14)

    # 计算并绘制U2的相关性矩阵
    df2 = pd.DataFrame(U2)
    corr_matrix2 = df2.T.corr()
    icc2 = compute_icc(U2)
    mask2 = np.triu(np.ones_like(corr_matrix2, dtype=bool), 1)
    sns.heatmap(corr_matrix2, mask=mask2, annot=True, fmt=".2f", cmap='Blues', 
                cbar=True, square=True, linewidths=.5, ax=ax[1], cbar_kws={'shrink': 0.5})
    ax[1].set_title(f'Correlation Across 10 Folds for Symptom loading', fontsize=14)
    ax[1].text(6, 3, f'ICC: {icc2}', verticalalignment='center', fontsize=14)

    plt.show()

def plot_loadings(loadings, dims=[0, 1], variants=SCALES, colors=['#EB7C69',  # 橙
    '#6b93c6',  # 蓝
    '#bd7dc4',  # 紫
    '#4ca866']  # 绿
    , cv=True, save_path=None,
    title=DIMENSION, title_colors=MULTI_COLOR,
    variant_categories=None,
    category_colors=None,
    category_order=None,
    x_label='Behavior Loadings (a.u.)',
    error_mode='std',
    legend_loc='lower right',
    legend_fontsize=15,
    y_tick_fontsize=13,
    x_label_fontsize=16,
    x_tick_fontsize=11):
    """
    Plot loadings with optional CV statistics in horizontal layout.

    Parameters:
    -----------
    loadings : numpy.ndarray
        If cv=True: shape (n_CVfolds, n_variants, n_dimensions)
        If cv=False: shape (n_variants, n_dimensions)
    dims : list of int
        Indices of dimensions to plot, for example [0, 1]
    variants : list
        List of variant names
    colors : list
        Unused for bar coloring in this version, kept for API compatibility
    cv : bool
        Whether to plot with CV statistics (error bars)
    error_mode : str
        Error bar mode when cv=True. One of {'std', 'sem'}.
    legend_loc : str
        'auto' compares lower-right and upper-left data overlap, preferring
        lower right when both positions have equal overlap.
    save_path : str or None
        If provided, saves as PDF at the given path without extension
    title : list or None
        Optional list of titles per dimension, aligned with dims by position
    title_colors : list
        Colors for titles, aligned with dims by position
    """

    # ----------- behavior default categories -----------
    somatic_raw = {
        "Malaise (MLS)", "Gastro-Intestinal Complaints (GIC)", "Head Pain Complaints (HPC)",
        "Neurological Complaints (NUC)", "Cognitive Complaints (COG)"
    }
    internalizing_raw = {
        "Suicidal/Death Ideation (SUI)", "Helplessness/Hopelessness (HLP)", "Self-Doubt (SFD)",
        "Inefficacy (NFC)", "Stress/Worry (STW)", "Anxiety (AXY)", "Anger Proneness (ANP)",
        "Behavior-Restricting Fears (BRF)", "Multiple Specific Fears (MSF)"
    }
    externalizing_raw = {
        "Juvenile Conduct Problems (JCP)", "Substance Abuse (SUB)", "Aggression (AGG)", "Activation (ACT)"
    }
    interpersonal_raw = {
        "Family Problems (FML)", "Interpersonal Passivity (IPP)", "Social Avoidance (SAV)",
        "Shyness (SHY)", "Disaffiliativeness (DSF)"
    }
    somatic = {mmpi_strip_abbr(s) for s in somatic_raw}
    internalizing = {mmpi_strip_abbr(s) for s in internalizing_raw}
    externalizing = {mmpi_strip_abbr(s) for s in externalizing_raw}
    interpersonal = {mmpi_strip_abbr(s) for s in interpersonal_raw}

    def infer_behavior_category(scale_name):
        normalized = mmpi_strip_abbr(scale_name)
        if normalized in somatic:
            return "Somatic/Cognitive"
        if normalized in internalizing:
            return "Internalizing"
        if normalized in externalizing:
            return "Externalizing"
        if normalized in interpersonal:
            return "Interpersonal"
        return None

    variants_arr = np.array(variants, dtype=object)

    # Build category for each variant (generic), while preserving default behavior style.
    use_default_category_mode = (variant_categories is None)
    if use_default_category_mode:
        variant_category_arr = np.array(
            [infer_behavior_category(str(v)) for v in variants_arr], dtype=object
        )
        # Backward-compatible fallback: unknown names are grouped as "Other"
        # instead of raising errors.
        variant_category_arr = np.array(
            ["Other" if c is None else c for c in variant_category_arr], dtype=object
        )
        default_colors = {
            "Somatic/Cognitive": colors[0],
            "Internalizing": colors[1],
            "Externalizing": colors[2],
            "Interpersonal": colors[3],
            "Other": "#9E9E9E",
        }
        effective_category_colors = default_colors if category_colors is None else dict(category_colors)
        if category_order is None:
            effective_category_order = ["Somatic/Cognitive", "Internalizing", "Externalizing", "Interpersonal"]
            if np.any(variant_category_arr == "Other"):
                effective_category_order.append("Other")
        else:
            effective_category_order = list(category_order)
    else:
        # Accept either dict(name->category) or list aligned with variants.
        if isinstance(variant_categories, dict):
            resolved = []
            for v in variants_arr:
                if v in variant_categories:
                    resolved.append(variant_categories[v])
                elif str(v) in variant_categories:
                    resolved.append(variant_categories[str(v)])
                else:
                    raise ValueError(f"variant_categories missing key for variant: {v}")
            variant_category_arr = np.array(resolved, dtype=object)
        else:
            if len(variant_categories) != len(variants_arr):
                raise ValueError(
                    f"Length mismatch: len(variant_categories)={len(variant_categories)} "
                    f"but len(variants)={len(variants_arr)}"
                )
            variant_category_arr = np.array(list(variant_categories), dtype=object)

        if category_colors is None:
            uniq_cat = list(dict.fromkeys([str(c) for c in variant_category_arr]))
            palette = list(colors)
            if len(palette) < len(uniq_cat):
                palette = [to_hex(c) for c in sns.color_palette("tab10", len(uniq_cat))]
            effective_category_colors = {cat: palette[i] for i, cat in enumerate(uniq_cat)}
        else:
            effective_category_colors = dict(category_colors)

        effective_category_order = (
            list(dict.fromkeys([str(c) for c in variant_category_arr]))
            if category_order is None else list(category_order)
        )

    # --------------------------------------------------------------

    n_dims = len(dims)
    fig, axes = plt.subplots(1, n_dims, figsize=(8 * n_dims, 6))
    axes = [axes] if n_dims == 1 else axes

    # ---- 保留你的标题逻辑 ----
    if title is None:
        _titles = [''] * n_dims
        _title_colors = ['black'] * n_dims
    else:
        if isinstance(title, (list, tuple)) and len(title) >= max(dims) + 1:
            _titles = [title[d] for d in dims]
        else:
            _titles = list(title) if isinstance(title, (list, tuple)) else [''] * n_dims
            if len(_titles) < n_dims:
                _titles += [''] * (n_dims - len(_titles))

        if isinstance(title_colors, (list, tuple)) and len(title_colors) >= max(dims) + 1:
            _title_colors = [title_colors[d] for d in dims]
        else:
            _title_colors = list(title_colors) if isinstance(title_colors, (list, tuple)) else ['black'] * n_dims
            if len(_title_colors) < n_dims:
                _title_colors += ['black'] * (n_dims - len(_title_colors))

    # ---------------- 绘图 ----------------
    for i, dim in enumerate(dims):
        ax = axes[i]

        # Extract & normalize
        if cv:
            U = loadings[:, :, dim]
            denom = np.abs(U.mean(axis=0)).max()
            denom = denom if denom not in [0, None] else 1.0
            U_norm = U / denom
            plot_values = U_norm.mean(axis=0)
            if error_mode == 'std':
                error_values = U_norm.std(axis=0)
            elif error_mode == 'sem':
                n_fold = U_norm.shape[0]
                if n_fold <= 1:
                    error_values = np.zeros(U_norm.shape[1], dtype=U_norm.dtype)
                else:
                    error_values = U_norm.std(axis=0, ddof=1) / np.sqrt(n_fold)
            else:
                raise ValueError(f"Unsupported error_mode: {error_mode}. Use 'std' or 'sem'.")
        else:
            U = loadings[:, dim]
            denom = np.abs(U).max()
            denom = denom if denom not in [0, None] else 1.0
            U_norm = U / denom
            plot_values = U_norm
            error_values = None

        # Remove zero variates
        nonzero_mask = ~(np.isclose(plot_values, 0))
        if nonzero_mask.sum() == 0:
            ax.set_axis_off()
            continue

        plot_values = plot_values[nonzero_mask]
        variants_nonzero = variants_arr[nonzero_mask]
        categories_nonzero = variant_category_arr[nonzero_mask]
        if error_values is not None:
            error_values = error_values[nonzero_mask]

        # Sort
        indices = np.argsort(plot_values)[::-1]
        sorted_values = plot_values[indices]
        sorted_scales = variants_nonzero[indices]
        sorted_categories = categories_nonzero[indices]
        sorted_errors = error_values[indices] if error_values is not None else None

        # ----------- color by category -----------
        missing_colors = [c for c in sorted_categories if c not in effective_category_colors]
        if missing_colors:
            raise ValueError(
                "Missing colors for categories: "
                f"{sorted(set([str(c) for c in missing_colors]))}. "
                "Please pass `category_colors`."
            )
        bar_colors = [effective_category_colors[c] for c in sorted_categories]

        # Keep seaborn visual style, but avoid aggregation when duplicate names exist.
        # We draw with unique internal labels, then map ticks back to original names.
        sorted_scales_internal = np.array(
            [f"{s}__{j}" for j, s in enumerate(sorted_scales)],
            dtype=object
        )

        sns.barplot(
            x=sorted_values,
            y=sorted_scales_internal,
            ax=ax,
            dodge=False,
            orient='h',
            color='white',
            errorbar=None
        )

        for p, c in zip(ax.patches, bar_colors):
            p.set_facecolor(c)
            p.set_edgecolor('none')

        # Restore original labels for display (without internal suffixes).
        y_pos = np.arange(len(sorted_scales_internal))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_scales)

        if cv and sorted_errors is not None:
            for x_val, y_val, err_val, bar_color in zip(sorted_values, y_pos, sorted_errors, bar_colors):
                ax.errorbar(
                    x=x_val,
                    y=y_val,
                    xerr=err_val,
                    fmt='none',
                    ecolor=bar_color,
                    elinewidth=1.0,
                    capsize=3
                )

        for idx, label in enumerate(ax.get_yticklabels()):
            label.set_color(bar_colors[idx] if idx < len(bar_colors) else 'black')
            label.set_fontsize(y_tick_fontsize)

        ax.grid(True, axis='x', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ax.legend_:
            ax.legend_.remove()

        ax.set_xlabel(x_label, fontsize=x_label_fontsize)
        if x_tick_fontsize is not None:
            ax.tick_params(axis='x', labelsize=x_tick_fontsize)
        if _titles[i]:
            ax.set_title(_titles[i], fontsize=14, color=_title_colors[i], fontweight='bold')

        # ----------- per-axis legend -----------
        if use_default_category_mode:
            # Keep legacy behavior-style legend (always 4 fixed groups).
            legend_cats = list(effective_category_order)
        else:
            # For generic mode: if order provided, respect it; otherwise only show present categories.
            if category_order is not None:
                legend_cats = list(effective_category_order)
            else:
                present_cats = list(dict.fromkeys([str(c) for c in sorted_categories]))
                legend_cats = [c for c in effective_category_order if c in present_cats]
                for c in present_cats:
                    if c not in legend_cats:
                        legend_cats.append(c)
        handles = [
            plt.Line2D(
                [0], [0], marker='s', linestyle='',
                markerfacecolor=effective_category_colors[cat],
                markeredgecolor='none',
                markersize=10,
                label=cat
            )
            for cat in legend_cats
        ]
        ax.legend(
            handles,
            legend_cats,
            loc='lower right' if legend_loc == 'auto' else legend_loc,
            fontsize=legend_fontsize,
            frameon=True,
            borderpad=0.25,
            handlelength=1,     # 色块-文字间距缩短
            handletextpad=0.7,    # 更紧凑
        )

    plt.subplots_adjust(wspace=0.75)

    if legend_loc == 'auto':
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in axes:
            if ax.legend_ is None:
                continue
            data_boxes = [p.get_window_extent(renderer) for p in ax.patches]
            data_boxes.extend(
                collection.get_transform().transform_path(path).get_extents().padded(0.5)
                for collection in ax.collections
                for path in collection.get_paths()
            )
            best_location, least_overlap = 'lower right', float('inf')
            for location in ('lower right', 'upper left'):
                ax.legend_.set_loc(location)
                legend_box = ax.legend_.get_window_extent(renderer)
                overlap = 0.0
                for data_box in data_boxes:
                    intersection = Bbox.intersection(legend_box, data_box)
                    if intersection is not None:
                        overlap += intersection.width * intersection.height
                if overlap < least_overlap:
                    best_location, least_overlap = location, overlap
            ax.legend_.set_loc(best_location)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')


NETWORK_ORDER_BRAIN = ['SC', 'VN', 'SMN', 'DAN', 'VAN', 'LN', 'FPCN', 'DMN']


def extract_network_code(label: str) -> str | None:
    """Extract tail network token like '(FPCN/R)' from an atlas label."""
    if label is None:
        return None
    m = re.search(r'\(([A-Za-z0-9]+)/(?:L|R)\)\s*$', str(label))
    return m.group(1) if m else None


def build_brain_network_color_map(palette: list[str]) -> dict[str, str]:
    """Map SC + 7 cortical networks to the first eight colors in palette."""
    if len(palette) < len(NETWORK_ORDER_BRAIN):
        raise ValueError(
            f"Brain network palette requires >= {len(NETWORK_ORDER_BRAIN)} colors, got {len(palette)}"
        )
    return {net: palette[i] for i, net in enumerate(NETWORK_ORDER_BRAIN)}


def apply_brain_label_colors(fig, network_color_map: dict[str, str], default_color: str = 'black') -> None:
    """Color y-axis label text by parsed brain network code."""
    for ax in fig.axes:
        for label in ax.get_yticklabels():
            net = extract_network_code(label.get_text())
            label.set_color(network_color_map.get(net, default_color))


def _resolve_variant_category_array(variants, variant_categories):
    """Resolve variant_categories to an array aligned with variants."""
    if variant_categories is None:
        return None

    variants_arr = np.array(variants, dtype=object)
    if isinstance(variant_categories, dict):
        resolved = []
        for v in variants_arr:
            if v in variant_categories:
                resolved.append(variant_categories[v])
            elif str(v) in variant_categories:
                resolved.append(variant_categories[str(v)])
            else:
                raise ValueError(f"variant_categories missing key for variant: {v}")
        return np.array(resolved, dtype=object)

    if len(variant_categories) != len(variants_arr):
        raise ValueError(
            f"Length mismatch: len(variant_categories)={len(variant_categories)} "
            f"but len(variants)={len(variants_arr)}"
        )
    return np.array(list(variant_categories), dtype=object)


def _resolve_modality_legend_categories(variant_category_arr, category_order):
    """Resolve legend category order with optional user order override."""
    if variant_category_arr is None:
        return []

    present = []
    for cat in variant_category_arr.tolist():
        c = str(cat)
        if c not in present:
            present.append(c)

    if category_order is None:
        return present

    ordered = [str(c) for c in category_order if str(c) in present]
    for cat in present:
        if cat not in ordered:
            ordered.append(cat)
    return ordered


def _select_legend_axes(axes, each_axis, ax_index, legend_name):
    """Pick legend target axes with bounds checking."""
    if not axes:
        raise ValueError(f"No axes available to place {legend_name}.")
    if each_axis:
        return list(axes)
    idx = int(ax_index)
    if idx < 0 or idx >= len(axes):
        raise IndexError(f"{legend_name} out of range: {idx} (n_axes={len(axes)})")
    return [axes[idx]]


def _compute_sorted_variant_order_for_dim(loadings, dim, variants, variant_category_arr, cv):
    """Reproduce plot_loadings sorting order for one dimension."""
    variants_arr = np.array(variants, dtype=object)

    if cv:
        U = loadings[:, :, dim]
        denom = np.abs(U.mean(axis=0)).max()
        denom = denom if denom not in [0, None] else 1.0
        plot_values = (U / denom).mean(axis=0)
    else:
        U = loadings[:, dim]
        denom = np.abs(U).max()
        denom = denom if denom not in [0, None] else 1.0
        plot_values = U / denom

    nonzero_mask = ~(np.isclose(plot_values, 0))
    if nonzero_mask.sum() == 0:
        empty = np.array([], dtype=object)
        return empty, (None if variant_category_arr is None else empty)

    values_nonzero = plot_values[nonzero_mask]
    variants_nonzero = variants_arr[nonzero_mask]
    cats_nonzero = None if variant_category_arr is None else variant_category_arr[nonzero_mask]

    indices = np.argsort(values_nonzero)[::-1]
    sorted_scales = variants_nonzero[indices]
    sorted_categories = None if cats_nonzero is None else cats_nonzero[indices]
    return sorted_scales, sorted_categories


def _recolor_errorbars(ax, y_centers, bar_colors):
    """Recolor existing errorbars by matching their y position to target bar centers."""
    if not y_centers or not bar_colors:
        return

    y_ref = np.array(y_centers, dtype=float)
    colors_ref = list(bar_colors)

    def _pick_color(y_val):
        idx = int(np.argmin(np.abs(y_ref - float(y_val))))
        return colors_ref[idx]

    for coll in ax.collections:
        if not hasattr(coll, "get_segments"):
            continue
        segs = coll.get_segments()
        if not segs:
            continue
        ys = []
        for seg in segs:
            arr = np.asarray(seg, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            ys.append(float(np.nanmean(arr[:, 1])))
        if not ys:
            continue
        c = _pick_color(float(np.nanmean(ys)))
        if hasattr(coll, "set_color"):
            coll.set_color(c)
        if hasattr(coll, "set_edgecolor"):
            coll.set_edgecolor(c)

    for line in ax.lines:
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if ydata.size == 0:
            continue
        c = _pick_color(float(np.nanmean(ydata)))
        line.set_color(c)
        if hasattr(line, "set_markeredgecolor"):
            line.set_markeredgecolor(c)


def plot_brain_loadings(
    loadings,
    dims,
    variants,
    x_label,
    save_path=None,
    title='',
    cv=True,
    error_mode='std',
    variant_categories=None,
    category_colors=None,
    category_order=None,
    network_palette=PALLETE,
    default_label_color='black',
    show_legend=True,
    legend_loc='lower right',
    legend_fontsize=15,
    y_tick_fontsize=13,
    x_label_fontsize=16,
    x_tick_fontsize=11,
    show_network_legend=False,
    network_legend_loc='upper left',
    network_legend_bbox=None,
    network_legend_ncol=1,
    network_legend_title=None,
    network_legend_fontsize=12,
    network_legend_ax_index=0,
    network_legend_each_axis=False,
    style_bars_by_network_and_modality=False,
    hollow_categories=('Structure',),
    hollow_bar_hatch='//',
    hollow_bar_linewidth=1.8,
    filled_bar_linewidth=0.8,
    modality_legend_loc='lower right',
    modality_legend_bbox=None,
    modality_legend_ncol=1,
    modality_legend_fontsize=None,
    modality_legend_hollow_linewidth=None,
    modality_legend_ax_index=0,
    modality_legend_each_axis=False,
):
    """
    Brain-loading wrapper over plot_loadings:
    1) plots bars via plot_loadings (without file saving)
    2) recolors y labels by network code
    3) optionally restyles bars: network color + hollow/filled by modality
    4) optionally controls network/modality legends
    5) saves the final figure if save_path is provided
    """
    network_color_map = build_brain_network_color_map(list(network_palette))
    variant_category_arr = _resolve_variant_category_array(variants, variant_categories)

    plot_loadings(
        loadings=loadings,
        dims=dims,
        variants=variants,
        cv=cv,
        save_path=None,
        title=title,
        variant_categories=variant_categories,
        category_colors=category_colors,
        category_order=category_order,
        x_label=x_label,
        error_mode=error_mode,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        y_tick_fontsize=y_tick_fontsize,
        x_label_fontsize=x_label_fontsize,
        x_tick_fontsize=x_tick_fontsize,
    )

    fig = plt.gcf()
    fig.canvas.draw()
    apply_brain_label_colors(fig, network_color_map, default_color=default_label_color)

    axes = list(fig.axes)

    if style_bars_by_network_and_modality:
        hollow_set = {str(c) for c in hollow_categories} if hollow_categories is not None else set()

        # Recolor bars by network and encode modality as hollow/filled.
        for i_ax, ax in enumerate(axes):
            if i_ax >= len(dims):
                continue
            dim = int(dims[i_ax])
            sorted_scales, sorted_categories = _compute_sorted_variant_order_for_dim(
                loadings=loadings,
                dim=dim,
                variants=variants,
                variant_category_arr=variant_category_arr,
                cv=cv,
            )

            bars = sorted(
                list(ax.patches),
                key=lambda p: float(p.get_y() + 0.5 * p.get_height())
            )
            n_apply = min(len(bars), len(sorted_scales))
            ordered_colors = []
            y_centers = []
            for j in range(n_apply):
                bar = bars[j]
                label = str(sorted_scales[j])
                cat = None if sorted_categories is None else str(sorted_categories[j])
                net = extract_network_code(label)
                net_color = network_color_map.get(net, default_label_color)
                y_centers.append(float(bar.get_y() + 0.5 * bar.get_height()))
                if (cat is not None) and (str(cat) in hollow_set):
                    bar.set_facecolor('none')
                    bar.set_edgecolor(net_color)
                    bar.set_linewidth(hollow_bar_linewidth)
                    bar.set_hatch(hollow_bar_hatch)
                else:
                    bar.set_facecolor(net_color)
                    bar.set_edgecolor(net_color)
                    bar.set_linewidth(filled_bar_linewidth)
                    bar.set_hatch(None)
                ordered_colors.append(net_color)

            if cv and ordered_colors:
                _recolor_errorbars(ax, y_centers, ordered_colors)

        # Remove legends created by plot_loadings; add custom legends below.
        for ax in axes:
            if ax.legend_:
                ax.legend_.remove()

        network_targets = []
        modality_targets = []

        if show_network_legend:
            network_targets = _select_legend_axes(
                axes=axes,
                each_axis=network_legend_each_axis,
                ax_index=network_legend_ax_index,
                legend_name="network_legend_ax_index",
            )

            # Legend display order: DMN at top, SC at bottom.
            network_cats = list(reversed(NETWORK_ORDER_BRAIN))
            network_handles = [
                plt.Line2D(
                    [0], [0], marker='s', linestyle='',
                    markerfacecolor=network_color_map[cat],
                    markeredgecolor='none',
                    markersize=9,
                    label=cat
                )
                for cat in network_cats
            ]

            network_kwargs = dict(
                loc=network_legend_loc,
                frameon=True,
                fontsize=network_legend_fontsize,
                ncol=network_legend_ncol,
                borderpad=0.3,
                handlelength=1.0,
                handletextpad=0.6,
                labelspacing=0.3,
            )
            if network_legend_title:
                network_kwargs["title"] = network_legend_title
                network_kwargs["title_fontsize"] = network_legend_fontsize
            if network_legend_bbox is not None:
                network_kwargs["bbox_to_anchor"] = network_legend_bbox

            need_add_artist = show_legend
            network_legends = {}
            for target_ax in network_targets:
                leg = target_ax.legend(network_handles, network_cats, **network_kwargs)
                network_legends[target_ax] = leg
                if need_add_artist:
                    target_ax.add_artist(leg)

        if show_legend:
            modality_targets = _select_legend_axes(
                axes=axes,
                each_axis=modality_legend_each_axis,
                ax_index=modality_legend_ax_index,
                legend_name="modality_legend_ax_index",
            )
            legend_cats = _resolve_modality_legend_categories(variant_category_arr, category_order)
            if legend_cats:
                handles = []
                legend_hollow_lw = (
                    hollow_bar_linewidth
                    if modality_legend_hollow_linewidth is None
                    else float(modality_legend_hollow_linewidth)
                )
                for cat in legend_cats:
                    if cat in hollow_set:
                        handles.append(
                            Patch(
                                facecolor='none',
                                edgecolor='black',
                                hatch=hollow_bar_hatch,
                                linewidth=legend_hollow_lw,
                                label=cat
                            )
                        )
                    else:
                        handles.append(
                            Patch(facecolor='black', edgecolor='black', linewidth=filled_bar_linewidth, label=cat)
                        )
                mfontsize = legend_fontsize if modality_legend_fontsize is None else modality_legend_fontsize
                modality_kwargs = dict(
                    loc=modality_legend_loc,
                    frameon=True,
                    fontsize=mfontsize,
                    ncol=modality_legend_ncol,
                    borderpad=0.3,
                    handlelength=1.2,
                    handletextpad=0.6,
                    labelspacing=0.3,
                )
                if modality_legend_bbox is not None:
                    modality_kwargs["bbox_to_anchor"] = modality_legend_bbox
                for target_ax in modality_targets:
                    target_ax.legend(handles, legend_cats, **modality_kwargs)
    else:
        if not show_legend:
            for ax in axes:
                if ax.legend_:
                    ax.legend_.remove()

        if show_network_legend:
            # Legend display order: DMN at top, SC at bottom.
            legend_cats = list(reversed(NETWORK_ORDER_BRAIN))
            handles = [
                plt.Line2D(
                    [0], [0], marker='s', linestyle='',
                    markerfacecolor=network_color_map[cat],
                    markeredgecolor='none',
                    markersize=9,
                    label=cat
                )
                for cat in legend_cats
            ]

            target_axes = _select_legend_axes(
                axes=axes,
                each_axis=network_legend_each_axis,
                ax_index=network_legend_ax_index,
                legend_name="network_legend_ax_index",
            )

            legend_kwargs = dict(
                loc=network_legend_loc,
                frameon=True,
                fontsize=network_legend_fontsize,
                ncol=network_legend_ncol,
                borderpad=0.3,
                handlelength=1.0,
                handletextpad=0.6,
                labelspacing=0.3,
            )
            if network_legend_title:
                legend_kwargs["title"] = network_legend_title
                legend_kwargs["title_fontsize"] = network_legend_fontsize
            if network_legend_bbox is not None:
                legend_kwargs["bbox_to_anchor"] = network_legend_bbox

            for target_ax in target_axes:
                target_ax.legend(
                    handles,
                    legend_cats,
                    **legend_kwargs,
                )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(f"{save_path}.pdf", bbox_inches='tight')
        
def subtype_stacked_bar(diag_labels, subtype_labels, title=None, save_path=None):
    """
    Create a stacked bar chart based on diagnostic and subtype labels.
    :param diag_labels: A vector of diagnostic labels (0: Anxiety, 1: ADHD, 2: Comorbid)
    :param subtype_labels: A vector of subtype labels (0, 1, 2 for three subtypes)
    :param title: Optional title for the chart
    """
    
    # Process the data
    subtypes = np.unique(subtype_labels)
    diags = np.array([2, 1, 0])  # Changed order to Anxiety, ADHD, Comorbid
    data = []
    for subtype in subtypes:
        subtype_data = []
        for diag in diags:
            count = np.sum((subtype_labels == subtype) & (diag_labels == diag))
            subtype_data.append(count)
        data.append(subtype_data)

    diag_names = ['Anxiety Disorders', 'ADHD', 'Comorbid']

    # Check if there are any comorbid cases
    has_comorbid = any(diag_labels == 0)
    # If there are no comorbid cases, remove it from diags and diag_names
    if not has_comorbid:
        diags = diags[:-1]
        diag_names = diag_names[:-1]
        data = [subtype_data[:-1] for subtype_data in data]

    # Create the chart
    fig, ax = plt.subplots(figsize=(8, 6))
    bottoms = np.zeros(len(subtypes))
    width = 0.6

    for i, diag in enumerate(diags):
        values = [subtype[i] for subtype in data]
        bars = ax.bar(SUBTYPE_NAME, values, width, label=diag_names[i], bottom=bottoms, color=MULTI_COLOR[i])
        bottoms += values

        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2,
                       f'{int(height)}', ha='center', va='center',
                       color='white', fontweight='bold')

    # Customize the chart
    ax.set_ylabel('Number of Patients', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(bottoms) * 1.1)  # Set y-axis limit with some padding
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(True, linestyle='-', alpha=0.7)
    ax.set_axisbelow(True)

    # Set x-tick labels with custom colors
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(SUBTYPE_COLOR[i])
        tick.set_fontsize(12)
        tick.set_fontweight('bold')
    
    plt.yticks(fontsize=10)

    # Customize the legend and move it closer to x-axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
             ncol=3, frameon=False, fontsize=12)

    if title:
        plt.title(title, fontsize=16, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #plt.savefig(f'{save_path}.svg', bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')


def plot_fc_heatmap(fc_values, pos_value=False, normalize=False, dark_mode=True,
                    threshold=None, cmap=None, title='', scale=.6, cb_title='',
                    horizontal_layout=False, save_path=None):
    """
    Plot functional connectivity heatmap with flexible layout orientation.
    
    Args:
        fc_values: FC matrices or list of FC matrices
        pos_value: bool, whether to use positive values only
        normalize: bool, whether to normalize the matrices
        dark_mode: bool, use dark theme colormap
        threshold: float, threshold for matrix values
        cmap: colormap for heatmap
        title: str, figure title
        scale: float, scaling factor for figure size
        cb_title: str, colorbar title
        horizontal_layout: bool, if True, display subplots in n x m layout (datasets x timepoints),
                         if False, display in m x n layout (timepoints x datasets)
        save_path: str, path to save the figure
    """
    gridcolor = 'white' if dark_mode else 'black'

    if cmap is None:
        if dark_mode:
            colors = [
                '#00ffff', '#0081ff','#0000ff', '#4b0082', 
                '#000000', '#7f0000', '#FF0000', '#FF8C00', '#FFFF00'
            ]
            cmap = ListedColormap(colors)
        else:
            cmap = 'bwr'

    def apply_threshold(mat, threshold):
        mat_copy = mat.copy()
        flat_mat = np.abs(mat_copy).flatten()
        
        if threshold >= 1:
            num_elements = int(threshold)
            threshold_value = np.partition(flat_mat, -num_elements)[-num_elements]
        else:
            threshold_value = np.percentile(flat_mat, 100 * (1 - threshold))
        mat_copy[np.abs(mat_copy) < threshold_value] = 0
        return mat_copy

    def normalize_matrix(mat, pos_value):
        if pos_value:
            return mat / np.max(mat)
        return mat / np.max(np.abs(mat))

    fc_values_list = [fc_values] if isinstance(fc_values[0], np.ndarray) else fc_values

    processed_mats_list = []
    for fc_values in fc_values_list:
        processed_mats = []
        for fc in fc_values:
            if threshold is not None:
                fc = apply_threshold(fc, threshold)
            mat = FC_vec2mat(fc)
            mat, pos = rearrange_FC(mat)
            if normalize:
                mat = normalize_matrix(mat, pos_value)
            processed_mats.append(mat)
        processed_mats_list.append(processed_mats)

    masks = [np.triu(np.ones_like(mat, dtype=bool), 1) for mat in processed_mats_list[0]]

    n_datasets = len(processed_mats_list)
    n_timepoints = len(processed_mats_list[0])

    if horizontal_layout:
        fig, axes = plt.subplots(n_datasets, n_timepoints,
                               figsize=(9 * n_timepoints * scale, 7.5 * n_datasets * scale))
    else:
        fig, axes = plt.subplots(n_timepoints, n_datasets,
                               figsize=(8.5 * n_datasets * scale, 7.5 * n_timepoints * scale))

    if n_datasets == 1 or n_timepoints == 1:
        axes = np.array(axes).reshape((n_datasets, n_timepoints) if horizontal_layout else (n_timepoints, n_datasets))

    for i in range(n_timepoints):
        for j in range(n_datasets):
            if horizontal_layout:
                ax = axes[j, i]
            else:
                ax = axes[i, j]

            mat = processed_mats_list[j][i]

            vmin, vmax = np.min(mat[~masks[i]]), np.max(mat[~masks[i]])
            if normalize:
                vmin, vmax = (0, 1) if pos_value else (-1, 1)

            if vmin == vmax:
                if pos_value:
                    vmin = 0
                    vmax = 1
                else:
                    vmin = -1
                    vmax = 1

            if not dark_mode:
                ax.plot([0, len(mat)], [0, len(mat)], color=gridcolor, linewidth=scale)
                ax.axvline(x=0, color=gridcolor, linewidth=scale)
                ax.axhline(y=len(mat), color=gridcolor, linewidth=scale+1)
            
            for p in pos:
                ax.plot([0, p], [p, p], color=gridcolor, linewidth=scale)
                ax.plot([p, p], [p, len(mat)], color=gridcolor, linewidth=scale)
            
            norm = Normalize(vmin=max(0, vmin), vmax=vmax) if pos_value else TwoSlopeNorm(vmin=-max(abs(vmin), abs(vmax)), vcenter=0, vmax=max(abs(vmin), abs(vmax)))

            sns.heatmap(mat, mask=masks[i], fmt=".2f", cmap=cmap, norm=norm,
                        cbar=True, square=True, linewidths=0, linecolor=gridcolor,
                        cbar_kws={'shrink': 0.8, 'pad': 0.0}, ax=ax)

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=12*scale)
            if cb_title:
                cbar.ax.set_title(cb_title, fontsize=14*scale, pad=10)

            ax.set_xticks([])
            ax.set_yticks([])

            # Network labels positioning
            tick_pos = pos + [len(mat)]
            middle_pos = [(tick_pos[i] + (tick_pos[i-1] if i > 0 else 0)) / 2 for i in range(len(tick_pos))]
            fontsize = 20 * scale

            for k, p in enumerate(middle_pos):
                color = PALLETE[k % len(PALLETE)]
                net_label = NET[k]
                
                # Top labels (rotated 45 degrees)
                y_offset = len(mat) + 1
                ax.text(p, y_offset, net_label, 
                       ha='center', va='top', color=color, 
                       fontsize=fontsize, rotation=45, weight='bold')
                
                # Left labels (horizontal)
                x_offset = -1
                ax.text(x_offset, p, net_label, 
                       ha='right', va='center', color=color, 
                       fontsize=fontsize, weight='bold')

    if title:
        fig.suptitle(title, fontsize=24*scale, y=0.95)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #plt.savefig(f'{save_path}.svg', bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')

        
def clustering(features, n_repeat=1, percent=0.9, n_clusters_max=7, display_mode='separate', show_y=True, savefig=False, seed=SEED):
    def get_bic_aic(data, max_clusters=10):
        bics, aics = np.zeros(max_clusters), np.zeros(max_clusters)
        for n_clusters in range(1, max_clusters + 1):
            gmm = GMM(n_components=n_clusters, n_init=10, random_state=seed)
            gmm.fit(data)
            bics[n_clusters-1] = gmm.bic(data)
            aics[n_clusters-1] = gmm.aic(data)
        return bics, aics

    if percent == 1:
        sample_data = features
        n_samples = len(features)
    else:
        n_samples = int(percent * len(features))

    bic_values = np.zeros((n_repeat, n_clusters_max))
    aic_values = np.zeros((n_repeat, n_clusters_max))

    for i in tqdm(range(n_repeat)):
        if percent < 1:
            sample_indices = np.random.choice(len(features), n_samples, replace=False)
            sample_data = features[sample_indices]
        bic_values[i, :], aic_values[i, :] = get_bic_aic(sample_data, max_clusters=n_clusters_max)

    mean_bic, min_bic, max_bic = np.mean(bic_values, axis=0), np.min(bic_values, axis=0), np.max(bic_values, axis=0)
    mean_aic, min_aic, max_aic = np.mean(aic_values, axis=0), np.min(aic_values, axis=0), np.max(aic_values, axis=0)

    xs = np.arange(1, n_clusters_max + 1)

    def plot_criterion(ax, xs, mean_values, min_values, max_values, title, ylabel):
        ax.plot(xs, mean_values, 'k-', linewidth=2.5)
        ax.fill_between(xs, y1=min_values, y2=max_values, alpha=0.3, facecolor=[0, 0, 0])
        
        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of clusters', fontsize=14, fontweight='bold')
        
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.1)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        
        if show_y:
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([f'{x:.0f}' for x in ax.get_yticks()], fontsize=12, fontweight='bold')
        else:
            ax.set_yticks([])

        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
        
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    if display_mode == 'separate':
        for criterion, mean_values, min_values, max_values, title, ylabel in [
            ('BIC', mean_bic, min_bic, max_bic, 'BIC by Clusters', 'BIC\n(lower is better)'),
            ('AIC', mean_aic, min_aic, max_aic, 'AIC by Clusters', 'AIC\n(lower is better)')
        ]:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            plot_criterion(ax, xs, mean_values, min_values, max_values, title, ylabel)
            plt.tight_layout()
            if savefig:
                plt.savefig(f'{criterion.lower()}.pdf', bbox_inches='tight')
            plt.show()
    elif display_mode == 'together':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        plot_criterion(ax1, xs, mean_bic, min_bic, max_bic, 'BIC by Clusters', 'BIC\n(lower is better)')
        plot_criterion(ax2, xs, mean_aic, min_aic, max_aic, 'AIC by Clusters', 'AIC\n(lower is better)')
        plt.tight_layout()
        if savefig:
            plt.savefig('bic_aic.pdf', bbox_inches='tight')
        plt.show()
    else:
        raise ValueError("Invalid display_mode. Choose 'separate' or 'together'.")
    return min_bic.min()

def clustering(features, n_repeat=1, percent=0.9, n_clusters_max=7, display_mode='separate', show_y=True, save_path=None, seed=SEED):
    def get_bic_aic(data, max_clusters=10):
        bics, aics = np.zeros(max_clusters), np.zeros(max_clusters)
        for n_clusters in range(1, max_clusters + 1):
            gmm = GMM(n_components=n_clusters, n_init=10, random_state=seed)
            gmm.fit(data)
            bics[n_clusters-1] = gmm.bic(data)
            aics[n_clusters-1] = gmm.aic(data)
        return bics, aics

    if percent == 1:
        sample_data = features
        n_samples = len(features)
    else:
        n_samples = int(percent * len(features))

    bic_values = np.zeros((n_repeat, n_clusters_max))
    aic_values = np.zeros((n_repeat, n_clusters_max))

    for i in tqdm(range(n_repeat)):
        if percent < 1:
            sample_indices = np.random.choice(len(features), n_samples, replace=False)
            sample_data = features[sample_indices]
        bic_values[i, :], aic_values[i, :] = get_bic_aic(sample_data, max_clusters=n_clusters_max)

    mean_bic, min_bic, max_bic = np.mean(bic_values, axis=0), np.min(bic_values, axis=0), np.max(bic_values, axis=0)
    mean_aic, min_aic, max_aic = np.mean(aic_values, axis=0), np.min(aic_values, axis=0), np.max(aic_values, axis=0)

    xs = np.arange(1, n_clusters_max + 1)

    def plot_criterion(ax, xs, mean_values, min_values, max_values, title, ylabel):
        ax.plot(xs, mean_values, 'k-', linewidth=2.5)
        ax.fill_between(xs, y1=min_values, y2=max_values, alpha=0.3, facecolor=[0, 0, 0])
        
        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of clusters', fontsize=14, fontweight='bold')
        
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.1)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        
        if show_y:
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([f'{x:.0f}' for x in ax.get_yticks()], fontsize=12, fontweight='bold')
        else:
            ax.set_yticks([])

        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
        
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    if display_mode == 'separate':
        for criterion, mean_values, min_values, max_values, title, ylabel in [
            ('BIC', mean_bic, min_bic, max_bic, 'BIC by Clusters', 'BIC\n(lower is better)'),
            ('AIC', mean_aic, min_aic, max_aic, 'AIC by Clusters', 'AIC\n(lower is better)')
        ]:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            plot_criterion(ax, xs, mean_values, min_values, max_values, title, ylabel)
            plt.tight_layout()
            if save_path:
                plt.savefig(f'{criterion.lower()}.pdf', bbox_inches='tight')
    elif display_mode == 'together':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        plot_criterion(ax1, xs, mean_bic, min_bic, max_bic, 'BIC by Clusters', 'BIC\n(lower is better)')
        plot_criterion(ax2, xs, mean_aic, min_aic, max_aic, 'AIC by Clusters', 'AIC\n(lower is better)')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #plt.savefig(f'{save_path}.svg', bbox_inches='tight')
            plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    else:
        raise ValueError("Invalid display_mode. Choose 'separate' or 'together'.")
    return min_bic.min()


def plot_chord(conn_matrix, max_line_width=12, min_line_width=0.5, colormap='coolwarm', nodewidth=3,
                          colorbar_pos='bottom', colorbar_width=0.018, colorbar_length=0.3, colorbar_pad=0.45, 
                          max_strength=None, shadow=False, node_names=NET, colors=PALLETE, save_path=None):
    from matplotlib.patches import Arc, PathPatch
    from matplotlib.path import Path

    n_nodes = len(node_names)
    # Constants
    node_names = [str(i) for i in range(n_nodes)] if node_names is None else node_names
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n_nodes] if colors is None else colors
    
    # Setup figure and axes
    fig = plt.figure(figsize=(8, 7))
    start, end = 0, 0.9
    center = (end - start) / 2
    ax = fig.add_axes([start, start, end, end])
    ax.set_aspect('equal')
    
    # Calculate node positions
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False) + np.pi/2
    pos = {node: (np.cos(angle), np.sin(angle)) 
           for node, angle in zip(node_names, angles)}
    
    # Setup color normalization based on actual data range
    cmap = plt.colormaps[colormap]
    
    # Calculate min and max values in the connection matrix
    data_min = np.min(conn_matrix)
    data_max = np.max(conn_matrix)
    # Find the maximum absolute value to create symmetric color range
    max_abs_value = max(abs(data_min), abs(data_max))
    
    # Create symmetric color normalization range
    norm = plt.Normalize(-max_abs_value, max_abs_value)
    
    # Setup line width scaling based on connection strength
    max_strength = np.max(np.abs(conn_matrix)) if max_strength is None else max_strength
    
    def get_line_width(strength):
        return min_line_width + (max_line_width - min_line_width) * abs(strength) / max_strength
    
    # Draw nodes with shadow effect
    arc_width = 40  # Width of each arc segment in degrees
    radius = 1.0
    for i, (node, (x, y)) in enumerate(pos.items()):
        node_angle = np.degrees(angles[i])
        start_angle = node_angle - arc_width/2
        end_angle = node_angle + arc_width/2
        
        # Draw self-connection
        self_strength = conn_matrix[i, i]
        outer_x, outer_y = np.cos(angles[i]), np.sin(angles[i])
        if self_strength != 0:
            ax.add_patch(Arc((outer_x, outer_y), width=0.4, height=0.4,
                            angle=node_angle, theta1=-90, theta2=90,
                            color=cmap(norm(self_strength)),  # Use original strength value for color mapping
                            linewidth=get_line_width(self_strength)))
        if shadow:
            # Draw shadow
            shadow = Arc((0.015, -0.015),  # Offset for shadow
                        width=2*radius,
                        height=2*radius,
                        angle=0,
                        theta1=start_angle,
                        theta2=end_angle,
                        color='gray',
                        alpha=0.3,
                        linewidth=nodewidth)
            ax.add_patch(shadow)
        
        # Draw main arc
        arc = Arc((0, 0),
                 width=2*radius,
                 height=2*radius,
                 angle=0,
                 theta1=start_angle,
                 theta2=end_angle,
                 color=colors[i],
                 linewidth=nodewidth)
        ax.add_patch(arc)
        
        # Add node label
        text_radius = 0.88
        text_x = text_radius * np.cos(angles[i])
        text_y = text_radius * np.sin(angles[i])
        text_angle = node_angle - 90 + (180 if text_y < 0 else 0)
        plt.text(text_x, text_y, node, color=colors[i],
                rotation=text_angle, ha='center', va='center',
                rotation_mode='anchor', fontsize=28, fontweight='bold')
    
    # Draw connections between nodes
    link_radius = 0.8

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if conn_matrix[i, j] != 0:
                strength = conn_matrix[i, j]
                start_pos = (link_radius*np.cos(angles[i]), link_radius*np.sin(angles[i]))
                end_pos = (link_radius*np.cos(angles[j]), link_radius*np.sin(angles[j]))
                
                # Calculate control point for curved line
                mid_x, mid_y = (start_pos[0] + end_pos[0])/2, (start_pos[1] + end_pos[1])/2
                control_point = (mid_x * 0.3, mid_y * 0.3)
                
                path = Path([start_pos, control_point, end_pos],
                          [Path.MOVETO, Path.CURVE3, Path.CURVE3])
                ax.add_patch(PathPatch(path, facecolor='none',
                                     edgecolor=cmap(norm(strength)),  # Use original strength value for color mapping
                                     linewidth=get_line_width(strength)))
    
    # Set plot limits and remove axes
    scale = 1.25
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.axis('off')
    
    # Add colorbar
    cax_pos = {'right': [center + colorbar_pad, center - colorbar_length/2, colorbar_width, colorbar_length],
               'bottom': [center - colorbar_length/2, center - colorbar_pad, colorbar_length, colorbar_width]}
    cax = fig.add_axes(cax_pos[colorbar_pos])
    
    # Create colorbar with symmetric range showing actual data values
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                       cax=cax,
                       orientation='vertical' if colorbar_pos in ['right', 'left'] else 'horizontal',
                       ticks=[-max_abs_value, 0, max_abs_value])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=16, length=0)
    
    # Format colorbar labels with appropriate decimal places
    cbar.ax.set_xticklabels([f'{-max_abs_value:.1f}', '0', f'{max_abs_value:.1f}'], fontsize=20)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #plt.savefig(f'{save_path}.svg', bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')


def ternary_plot(prob, mask=None, dot_size=5, save_path=None, title=None):

    pio.kaleido.scope.default_format = "svg"

    df = pd.DataFrame(prob, columns=['subtype 1', 'subtype 2', 'subtype 3'])
    df['subtype'] = df[['subtype 1', 'subtype 2', 'subtype 3']].idxmax(axis=1)
    df['probability'] = df[['subtype 1', 'subtype 2', 'subtype 3']].max(axis=1)

    fig = go.Figure()


    for subtype in SUBTYPE_COLORMAP.keys():

        df_subtype = df[df['subtype'] == subtype]
        
        # set color based on subtype
        colors = df_subtype['probability']
        
        if mask is not None:
            mask_subtype = mask[df_subtype.index]
            colors = ['gray' if m else c for m, c in zip(mask_subtype, colors)]
        
        fig.add_trace(go.Scatterternary({
            'mode': 'markers',
            'a': df_subtype['subtype 1'],
            'b': df_subtype['subtype 2'],
            'c': df_subtype['subtype 3'],
            'marker': {
                'symbol': 'circle',  
                'color': colors,
                'colorscale': SUBTYPE_COLORMAP[subtype],
                'size': dot_size,
            },
            'name': subtype
        }))

    fig.update_layout({
        'ternary': {
            'sum': 1,
            'bgcolor': 'whitesmoke',  # Set background color of ternary plot area
            'aaxis': {
                'gridcolor': SUBTYPE_COLOR[0],  # Color of coordinate grid lines
                'griddash': 'dash',
                'gridwidth': 1.5,
                'title': {
                    'text': SUBTYPE_NAME[0],
                    'font': {
                        'size': 16,   # Font size
                        'color': SUBTYPE_COLOR[0]  # Font color
                    }
                }
            },
            'baxis': {
                'gridcolor': SUBTYPE_COLOR[1],
                'griddash': 'dash',
                'gridwidth': 1.5,
                'title': {
                    'text': SUBTYPE_NAME[1],
                    'font': {
                        'size': 16,  # Font size
                        'color': SUBTYPE_COLOR[1]  # Font color
                    }
                }
            },
            'caxis': {
                'gridcolor': SUBTYPE_COLOR[2],
                'griddash': 'dash',
                'gridwidth': 1.5,
                'title': {
                    'text': SUBTYPE_NAME[2],
                    'font': {
                        'size': 16,  # Font size
                        'color': SUBTYPE_COLOR[2]  # Font color
                    }
                }
            },
        },
        'title': title,
        'showlegend': False,
        'width': 750,  # Chart width
        'height': 600  # Chart height
    })

    if save_path:
        fig.write_image(f'{save_path}.pdf')

    fig.show()


def plot_LME(df, df_anova, subtype_names=None, subtype_colors=None, titles=None, errorbar='ci',
             num_cols=5, p_loc=[0.5, 0.95], save_path=None):
    """
    Plot Linear Mixed Effects analysis results
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to plot
    anova_results : dict
        Dictionary containing ANOVA results with p-values
    subtype_names : dict, optional
        Dictionary mapping subtype indices to names
    subtype_colors : dict, optional
        Dictionary mapping subtype indices to colors
    titles : list, optional
        List of titles for each subplot
    errorbar: str, default='ci'
        Type of error bars to plot. Can be 'ci' for confidence intervals or 'sd' for std.
    num_cols : int, default=5
        Number of columns in the subplot grid
    p_loc : list, default=[0.5, 0.95]
        Location of p-value text [x, y] in axis coordinates
    """
    def format_p_value(p_value):
        """Format p-value for plotting"""
        if isinstance(p_value, float):
            if p_value < 0.001:
                return '< 0.001'
            elif p_value < 0.01:
                return '< 0.01'
            elif p_value < 0.05:
                return '< 0.05'
            else:
                return f'= {p_value:.4f}'
        return str(p_value)

    df['Subtype'] = df['Subtype'].astype('Int64')
    # Set default subtype names and colors if not provided
    if subtype_names is None:
        subtype_names = {0: SUBTYPE_NAME[0], 1: SUBTYPE_NAME[1], 2: SUBTYPE_NAME[2]}
    if subtype_colors is None:
        subtype_colors = {0: SUBTYPE_COLOR[0], 1: SUBTYPE_COLOR[1], 2: SUBTYPE_COLOR[2]}
    
    # Get unique time points from data
    time_points = sorted(df['Age'].unique())
    
    # Determine the scale columns to plot
    keys = list(df_anova.keys())
    num_scales = len(keys)

    # Calculate required number of rows
    num_rows = (num_scales + 1 + num_cols - 1) // num_cols
    
    # Create a temporary figure to get legend handles
    temp_fig, temp_ax = plt.subplots()
    temp_plot = sns.lineplot(
        data=df,
        x="Age",
        y=keys[0],  # Use the first variable
        hue="Subtype",
        palette=subtype_colors
    )
    handles, _ = temp_plot.get_legend_handles_labels()
    plt.close(temp_fig)
    
    # Create main figure with subplot grid
    fig = plt.figure(figsize=(3 * num_cols, 3 * num_rows))
    gs = fig.add_gridspec(num_rows, num_cols)
    
    # Plot each scale's line plot
    for i, key in enumerate(keys):
        # Calculate row and column position
        row = i // num_cols
        col = i % num_cols
        
        title = titles[i] if titles else key
        # Create subplot at the calculated position
        ax = fig.add_subplot(gs[row, col])

        # Create line plot
        sns.lineplot(
            data=df,
            x="Age",
            y=key,
            hue="Subtype",
            palette=subtype_colors, 
            errorbar=errorbar,
            legend=False  # No legend for individual subplots
        )

        # Format p-value
        fdr_p_value = df_anova.loc['Age:Subtype_p_fdr', key]
        p_label = format_p_value(fdr_p_value)

        # Set title and labels
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Age in Years')
        ax.set_ylabel('Predicted Scale Mean')
        ax.set_xticks(time_points)
        ax.set_xticklabels(time_points)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add p-value text
        ax.text(p_loc[0], p_loc[1], f'p$_{{FDR}}$ {p_label}',
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10)

    # Remove empty subplot before legend if needed
    if num_scales < num_rows * num_cols - 1:
        empty_ax = fig.add_subplot(gs[num_rows-1, num_cols-2])
        empty_ax.remove()

    # Add legend in the last position using handles from temporary plot
    legend_ax = fig.add_subplot(gs[num_rows-1, num_cols-1])
    legend_ax.axis('off')
    legend_ax.legend(handles=handles, 
                    labels=[subtype_names[i] for i in range(len(subtype_names))],
                    loc='center', 
                    fontsize=12)

    # Adjust layout
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #plt.savefig(f'{save_path}.svg', bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')


def plot_scales_by_time(df, subtype_names=None, subtype_colors=None, titles=None, df_tc=None, 
                num_cols=5, fmt='o', save_path=None):
    """
    Plot line charts of LME results with error bars for subtypes and optionally add a TC reference line.
    Optionally remove outliers separately for each subtype (not separated by time points).
    
    Parameters:
    df (DataFrame): Main dataframe containing the data to plot.
    p_values (dict): Dictionary of FDR-adjusted p-values for each scale.
    subtype_names (dict): Labels for each subtype.
    subtype_colors (dict): Colors for each subtype.
    titles (list): Optional, custom titles for each subplot.
    df_tc (DataFrame): Optional, dataframe for TC data to add reference lines.
    num_cols (int): Number of columns in the subplot grid.
    p_loc (list): Coordinates for p-value text placement in each subplot.
    fmt (str): Format string for the plot markers.
    """
    
    # Set default subtype names and colors if not provided
    df['Subtype'] = df['Subtype'].astype('Int64')
    if subtype_names is None:
        subtype_names = {0: SUBTYPE_NAME[0], 1: SUBTYPE_NAME[1], 2: SUBTYPE_NAME[2]}
    if subtype_colors is None:
        subtype_colors = {0: SUBTYPE_COLOR[0], 1: SUBTYPE_COLOR[1], 2: SUBTYPE_COLOR[2]}
    
    # Determine the scale columns to plot (exclude 'eventname', 'Time', 'Subtype')
    keys = [k for k in df.keys() if k not in ['src_subject_id', 'eventname', 'Subtype','Age', 'Time', 'Sex', 'Parental_Education', 'Income', 'Marital', 'Treatment', 'ICV']]
    num_scales = len(keys)

    # Calculate required number of rows
    num_rows = (num_scales + 1 + num_cols - 1) // num_cols
    
    # Create figure with subplot grid
    fig = plt.figure(figsize=(3 * num_cols, 3 * num_rows))
    gs = fig.add_gridspec(num_rows, num_cols)
    
    # Plot each scale's line plot
    for i, key in enumerate(keys):
        # Calculate row and column position
        row = i // num_cols
        col = i % num_cols
        
        # Create subplot at the calculated position
        ax = fig.add_subplot(gs[row, col])

        # Initialize a list to store cleaned data for all subtypes
        df_key_cleaned = []
        
        # Process each subtype separately
        for subtype in df['Subtype'].unique():
            # Get data for current scale and subtype (all time points)
            mask = (df['Subtype'] == subtype)
            current_data = df[mask].dropna(subset=[key])
            
            if len(current_data) > 0:  # Only process if we have data
                df_key_cleaned.append(current_data)
        
        # Combine all cleaned data
        df_key = pd.concat(df_key_cleaned) if df_key_cleaned else pd.DataFrame()

        # Process TC data if provided
        if df_tc is not None:
            # Process TC data as a single group (not separated by time points)
            current_tc_data = df_tc.dropna(subset=[key])
            
            if len(current_tc_data) > 0:
                
                df_tc_key = current_tc_data
            
                # Aggregate TC data
                if not df_tc_key.empty:
                    df_tc_agg = df_tc_key.groupby('Time')[key].agg(['mean', 'sem']).reset_index()
                    
                    # Plot TC values
                    sns.lineplot(
                        x='Time', 
                        y='mean', 
                        data=df_tc_agg,
                        label='Typical Control',
                        color='gray',
                        linestyle='--',
                        legend=False,
                        ax=ax
                    )
                    ax.errorbar(
                        x=df_tc_agg['Time'], 
                        y=df_tc_agg['mean'], 
                        yerr=df_tc_agg['sem'], 
                        fmt=fmt, 
                        color='gray', 
                        capsize=3,
                        alpha=0.5
                    )

        # Create aggregated data for plotting
        if not df_key.empty:
            df_agg = df_key.groupby(['Time', 'Subtype'], observed=False)[key].agg(['mean', 'sem']).reset_index()

            # Plot each subtype's line
            for subtype, color in subtype_colors.items():
                df_subtype = df_agg[df_agg['Subtype'] == subtype]
                if not df_subtype.empty:
                    sns.lineplot(
                        x='Time', 
                        y='mean', 
                        data=df_subtype, 
                        label=subtype_names[subtype], 
                        color=color, 
                        ax=ax,
                        legend=False
                    )
                    ax.errorbar(
                        x=df_subtype['Time'], 
                        y=df_subtype['mean'], 
                        yerr=df_subtype['sem'], 
                        fmt=fmt, 
                        color=color, 
                        capsize=3
                    )
                
            ax.set_xticks([0, 1, 2, 3, 4])


        # Set title and labels
        title = titles[i] if titles else key
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Follow-up Year')
        ax.set_ylabel('Score Mean')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Remove empty subplot before legend if needed
    if num_scales < num_rows * num_cols - 1:
        empty_ax = fig.add_subplot(gs[num_rows-1, num_cols-2])
        empty_ax.remove()

    # Add legend in the last position
    legend_ax = fig.add_subplot(gs[num_rows-1, num_cols-1])
    legend_ax.axis('off')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles=handles, labels=labels, loc='center', fontsize=12)

    # Adjust layout
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #plt.savefig(f'{save_path}.svg', bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')


# def plot_partial_eta_squared(
#         df_anova, 
#         scale_names=None, 
#         colors=MULTI_COLOR,     
#         save_path=None,
#         effect_labels=None,
#         legend_loc='upper right',
#         star_offset=0.): 
#     """
#     Plot grouped bar chart (horizontal layout → rotated to vertical bars).
#     """
#     # ---------- Data wrangling ----------
#     effects = list(dict.fromkeys(
#         [re.sub(r'_p_fdr$|_eta_sq$', '', idx) for idx in df_anova.index]))
#     colors = colors[:len(effects)]

#     # Eta-squared
#     eta_sq_data = df_anova.loc[[f"{e}_eta_sq" for e in effects]]
#     eta_sq_data = eta_sq_data.reset_index().melt(
#         id_vars='index', var_name='Scale', value_name='Eta_Squared')
#     eta_sq_data['Effect'] = eta_sq_data['index'].str.replace('_eta_sq', '')

#     # p-values
#     p_values = df_anova.loc[[f"{e}_p_fdr" for e in effects]]
#     p_values = p_values.reset_index().melt(
#         id_vars='index', var_name='Scale', value_name='p_value')
#     p_values['Effect'] = p_values['index'].str.replace('_p_fdr', '')

#     # Merge
#     plot_data = pd.merge(
#         eta_sq_data, p_values[['Scale', 'Effect', 'p_value']],
#         on=['Scale', 'Effect'])
#     plot_data['Significance'] = plot_data['p_value'].apply(
#         get_significance_symbol)

#     # Apply effect name replacements (only affects plot)
#     if effect_labels:
#         plot_data['Effect'] = plot_data['Effect'].replace(effect_labels)

#     # ---------- Plot ----------
#     plt.style.use('default')
#     fig, ax = plt.subplots(figsize=(10, 5))          

#     sns.barplot(
#         data=plot_data,
#         x='Scale',                 
#         y='Eta_Squared',
#         hue='Effect',
#         errorbar=None,
#         width=0.7,
#         palette=colors if colors else None,
#         ax=ax)

#     # Clean axes
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     # Tick labels
#     if scale_names:
#         ax.set_xticks(range(len(scale_names)))
#         ax.set_xticklabels(scale_names, rotation=45, ha='right')
#     ax.set_ylabel("Partial Eta Squared", fontsize=14)
#     ax.set_xlabel("")

#     # ---------- Significance markers ----------
#     max_height = plot_data['Eta_Squared'].max()
#     marker_offset = max_height * 0.02

#     n_effects = len(effects)
#     n_scales = len(plot_data['Scale'].unique())

#     for scale_idx in range(n_scales):
#         scale_data = plot_data[
#             plot_data['Scale'] == plot_data['Scale'].unique()[scale_idx]]

#         for effect_idx, (_, row) in enumerate(scale_data.iterrows()):
#             if row['Significance']:
#                 x_pos = scale_idx + (effect_idx - (n_effects - 1) / 2) * (0.7 / n_effects) + star_offset
#                 y_pos = row['Eta_Squared'] + marker_offset

#                 ax.text(
#                     x_pos, y_pos,
#                     row['Significance'],
#                     ha='center', va='bottom',
#                     rotation=90,                 
#                     fontsize=10)

#     # Vertical dashed lines between scale groups
#     for i in range(1, n_scales):
#         ax.axvline(i - 0.5, linestyle='--', color='gray', linewidth=0.7)

#     # Legend in upper-right corner
#     ax.legend(
#         title='Effect',
#         loc=legend_loc,
#         fontsize=10)

#     plt.tight_layout()
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         #plt.savefig(f'{save_path}.svg', bbox_inches='tight')
#         plt.savefig(f'{save_path}.pdf', bbox_inches='tight')

def plot_partial_eta_squared(
        df_anova, 
        scale_names=None, 
        colors=MULTI_COLOR,     
        save_path=None,
        effect_labels=None,
        legend_config=None,  # 改为更灵活的配置参数
        star_offset=0.): 
    """
    Plot grouped bar chart (horizontal layout → rotated to vertical bars).
    
    legend_config参数说明:
    - None: 使用默认位置 'upper right'
    - str: 使用matplotlib标准位置 ('upper right', 'lower left'等)
    - tuple: 使用精确坐标 (x, y)，相对于axes的比例坐标
    - dict: 完整配置，如 {'loc': 'upper right', 'bbox_to_anchor': (1, 1), 'ncol': 2}
    """
    # ---------- Data wrangling ----------
    effects = list(dict.fromkeys(
        [re.sub(r'_p_fdr$|_eta_sq$', '', idx) for idx in df_anova.index]))
    colors = colors[:len(effects)]

    # Eta-squared
    eta_sq_data = df_anova.loc[[f"{e}_eta_sq" for e in effects]]
    eta_sq_data = eta_sq_data.reset_index().melt(
        id_vars='index', var_name='Scale', value_name='Eta_Squared')
    eta_sq_data['Effect'] = eta_sq_data['index'].str.replace('_eta_sq', '')

    # p-values
    p_values = df_anova.loc[[f"{e}_p_fdr" for e in effects]]
    p_values = p_values.reset_index().melt(
        id_vars='index', var_name='Scale', value_name='p_value')
    p_values['Effect'] = p_values['index'].str.replace('_p_fdr', '')

    # Merge
    plot_data = pd.merge(
        eta_sq_data, p_values[['Scale', 'Effect', 'p_value']],
        on=['Scale', 'Effect'])
    plot_data['Significance'] = plot_data['p_value'].apply(
        get_significance_symbol)

    # Apply effect name replacements (only affects plot)
    if effect_labels:
        plot_data['Effect'] = plot_data['Effect'].replace(effect_labels)

    # ---------- Plot ----------
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 5))          

    sns.barplot(
        data=plot_data,
        x='Scale',                 
        y='Eta_Squared',
        hue='Effect',
        errorbar=None,
        width=0.7,
        palette=colors if colors else None,
        ax=ax)

    # Clean axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tick labels
    if scale_names:
        ax.set_xticks(range(len(scale_names)))
        ax.set_xticklabels(scale_names, rotation=45, ha='right')
    ax.set_ylabel("Partial Eta Squared", fontsize=14)
    ax.set_xlabel("")

    # ---------- Significance markers ----------
    max_height = plot_data['Eta_Squared'].max()
    marker_offset = max_height * 0.02

    n_effects = len(effects)
    n_scales = len(plot_data['Scale'].unique())

    for scale_idx in range(n_scales):
        scale_data = plot_data[
            plot_data['Scale'] == plot_data['Scale'].unique()[scale_idx]]

        for effect_idx, (_, row) in enumerate(scale_data.iterrows()):
            if row['Significance']:
                x_pos = scale_idx + (effect_idx - (n_effects - 1) / 2) * (0.7 / n_effects) + star_offset
                y_pos = row['Eta_Squared'] + marker_offset

                ax.text(
                    x_pos, y_pos,
                    row['Significance'],
                    ha='center', va='bottom',
                    rotation=90,                 
                    fontsize=10)

    # Vertical dashed lines between scale groups
    for i in range(1, n_scales):
        ax.axvline(i - 0.5, linestyle='--', color='gray', linewidth=0.7)

    # ---------- 精确控制Legend位置 ----------
    legend_kwargs = {'title': 'Effect', 'fontsize': 10}
    
    if legend_config is None:
        # 默认位置
        legend_kwargs['loc'] = 'upper right'
    elif isinstance(legend_config, str):
        # 字符串位置
        legend_kwargs['loc'] = legend_config
    elif isinstance(legend_config, tuple):
        # 精确坐标 (x, y)
        legend_kwargs['loc'] = 'upper left'  # 基准位置
        legend_kwargs['bbox_to_anchor'] = legend_config
    elif isinstance(legend_config, dict):
        # 完整配置字典
        legend_kwargs.update(legend_config)
    
    ax.legend(**legend_kwargs)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')


def remove_outliers_std(data, outlier_std=3):
    """
    Remove outliers based on standard deviation method.
    
    Parameters:
    data: numpy array of values
    outlier_std: number of standard deviations to use as threshold
    
    Returns:
    tuple: (cleaned data, number of outliers removed)
    """
    mean = np.mean(data)
    std = np.std(data)
    lower_bound = mean - outlier_std * std
    upper_bound = mean + outlier_std * std
    mask = (data >= lower_bound) & (data <= upper_bound)
    return data[mask], mask

def remove_outliers_mad(data, outlier_threshold=4):
    """
    Remove outliers based on Median Absolute Deviation (MAD) method.
    Falls back to std method if MAD is 0.
    
    Parameters:
    data: numpy array of values
    outlier_threshold: threshold for modified z-scores
    
    Returns:
    tuple: (cleaned data, number of outliers removed)
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return remove_outliers_std(data)
    modified_zscores = 0.6745 * (data - median) / mad
    mask = np.abs(modified_zscores) <= outlier_threshold
    return data[mask], mask

def subtype_radar(
    group_label: np.ndarray,
    scale: np.ndarray,
    scale_list: List[str],
    scale_tc: np.ndarray = None,
    y_lim: float = 0.8, 
    same_mean: float = 0.,
    remove_outliers: bool = True,
    remove_method: Literal['std', 'mad'] = 'std',
    remove_method_list: List[str] = None,
    outlier_threshold: float = 4,
    normalize_method: Literal['minmax', 'maxdiv'] = 'minmax',
    stats: Literal['anova', 'kruskal'] = 'kruskal',
    fdr: bool = True,
    subtype_name = SUBTYPE_NAME,
    subtype_color = SUBTYPE_COLOR,
    type: Literal['Cognition', 'MH', 'Personality', 'Environment'] = 'Cognition',
    type_list = None,
    title: str = '',
    legend_position: Literal['right', 'bottom'] = 'bottom',
    save_path: str = None,
):
    """
    Draw a radar plot of subtype profiles.
    Only the legend auto-shrink feature is new; everything else remains unchanged.
    """

    # ---------- helper: normalization ----------
    def normalize_scale(data: np.ndarray, method: str) -> np.ndarray:
        if method == 'minmax':
            min_val, max_val = np.min(data), np.max(data)
            return (data - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(data)
        # method == 'maxdiv'
        max_val = np.max(np.abs(data))
        return data / max_val if max_val != 0 else np.zeros_like(data)

    # ---------- plotly setup ----------
    pio.kaleido.scope.default_format = "svg"
    subtype_color = [to_hex(c) for c in subtype_color]

    # ---------- merge typical control ----------
    n_groups = len(np.unique(group_label))
    if scale_tc is not None:
        group_label = np.concatenate((group_label, np.ones(scale_tc.shape[0]) * n_groups))
        scale = np.concatenate((scale, scale_tc))
        subtype_name = subtype_name + ['Typical Control']
        subtype_color = subtype_color + ['gray']

    if type_list is None:
        type_list = [type] * len(scale_list)
    if remove_method_list is None:
        remove_method_list = [remove_method] * len(scale_list)

    unique_labels = list(np.unique(group_label))
    n_scales = scale.shape[1]

    # ---------- outlier removal ----------
    cleaned_data_by_scale = [[] for _ in range(n_scales)]
    cleaned_labels_by_scale = [[] for _ in range(n_scales)]
    outlier_counts = np.zeros((n_scales, len(unique_labels)))

    for i, subtype in enumerate(unique_labels):
        subtype_data = scale[group_label == subtype]
        for j in range(n_scales):
            data_j = subtype_data[:, j]
            data_j = data_j[~np.isnan(data_j)]
            if remove_outliers:
                if remove_method_list[j] == 'mad':
                    data_j, mask = remove_outliers_mad(data_j, outlier_threshold)
                else:
                    data_j, mask = remove_outliers_std(data_j)
                outlier_counts[j, i] = np.sum(~mask)
            cleaned_data_by_scale[j].extend(data_j)
            cleaned_labels_by_scale[j].extend([subtype] * len(data_j))

    # ---------- global statistics ----------
    p_values = []
    for j in range(n_scales):
        scale_data = [np.array(cleaned_data_by_scale[j])[np.array(cleaned_labels_by_scale[j]) == s]
                      for s in unique_labels[:n_groups]] # exclude TC
        p_val = scipy.stats.kruskal(*scale_data)[1] if stats == 'kruskal' else scipy.stats.f_oneway(*scale_data)[1]
        p_values.append(p_val)

    if fdr:
        _, p_corrected, *_ = multipletests(p_values, method='fdr_bh')
        significance_symbols = [get_significance_symbol(p) for p in p_corrected]
    else:
        significance_symbols = [get_significance_symbol(p) for p in p_values]

    # ---------- normalization by scale ----------
    normalized_data_by_scale, scale_mean = [], np.zeros(n_scales)
    for j in range(n_scales):
        normed = normalize_scale(np.array(cleaned_data_by_scale[j]), normalize_method)
        normalized_data_by_scale.append((normed, cleaned_labels_by_scale[j]))
        scale_mean[j] = np.mean(normed)

    # ---------- aggregate per subtype ----------
    group_means, group_sems, group_ns, legend_texts = [], [], [], []
    for subtype in unique_labels:
        means, sems = [], []
        n_samples = len(scale[group_label == subtype])
        for j in range(n_scales):
            normed, labels = normalized_data_by_scale[j]
            mask = np.array(labels) == subtype
            data_sub = normed[mask]
            means.append(np.mean(data_sub))
            sems.append(np.std(data_sub) / np.sqrt(len(data_sub)))
        group_means.append(np.array(means))
        group_sems.append(np.array(sems))
        group_ns.append(n_samples)
        legend_texts.append(f'{subtype_name[len(group_ns)-1]} (N={n_samples})')

    # ---------- angles for radar ----------
    angles = np.linspace(0, 360, n_scales, endpoint=False)
    angles = np.append(angles, angles[0])        # close loop

    # ---------- polar traces ----------
    data = []
    if same_mean > 0:
        group_means = np.array(group_means) - scale_mean + same_mean

    for i, (means, sems) in enumerate(zip(group_means, group_sems)):
        color = subtype_color[i % len(subtype_color)]
        means = np.append(means, means[0])
        errors = np.append(1.96 * sems, 1.96 * sems[0])

        data.append(go.Scatterpolar(
            r=means, theta=angles, mode='lines+markers',
            name=legend_texts[i],
            line=dict(color=color), marker=dict(color=color, symbol='circle', size=8)
        ))

        # error bars
        for j in range(len(means)-1):
            r0, rp, rm = means[j], means[j] + errors[j], means[j] - errors[j]
            theta = angles[j]
            for r_pair in [(r0, rp), (r0, rm)]:
                data.append(go.Scatterpolar(
                    r=list(r_pair), theta=[theta]*2, mode='lines',
                    line=dict(color=color, width=2), showlegend=False, hoverinfo='skip'
                ))
            for r_cap in [rp, rm]:
                data.append(go.Scatterpolar(
                    r=[r_cap]*2, theta=[theta-2, theta+2], mode='lines',
                    line=dict(color=color, width=2), showlegend=False, hoverinfo='skip'
                ))

    # ---------- significance markers ----------
    for ang, sym in zip(angles[:-1], significance_symbols):
        data.append(go.Scatterpolar(
            r=[y_lim], theta=[ang], mode='text', text=[sym],
            textfont=dict(size=16, color='black'),
            showlegend=False, hoverinfo='skip'
        ))

    # ---------- colored tick labels ----------
    colored_ticktext = [
        f'<span style="color:{TEXT_COLOR[type_list[i]]}; font-size:16px">{scale_list[i]}</span>'
        for i in range(n_scales)
    ]

    FIG_WIDTH = 1200

    # ---------- legend auto-shrink ----------
    if legend_position == 'bottom':
        font_size = 16
        def est_width(fs: int) -> float:
            # ~0.6*fs px per char, plus 40 px marker/padding
            return sum(len(txt) * fs * 0.6 + 40 for txt in legend_texts)
        while est_width(font_size) > 0.9 * FIG_WIDTH and font_size > 10:
            font_size -= 1
        legend_cfg = dict(
            orientation="h", yanchor="top", y=-0.1,
            xanchor="center", x=0.5,
            font=dict(size=font_size)
        )
        fig_height = 650
    else:
        legend_cfg = dict(font=dict(size=16))
        fig_height = 600

    # ---------- layout ----------
    layout = go.Layout(
        width=FIG_WIDTH, height=fig_height,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, y_lim*1.1],
                            tickvals=[0, 0.2, 0.4, 0.6],
                            ticktext=['']*4, tickfont=dict(size=14)),
            angularaxis=dict(tickmode='array', tickvals=angles[:-1],
                             ticktext=colored_ticktext,
                             rotation=0, direction='clockwise',
                             tickfont=dict(size=14))
        ),
        showlegend=True, legend=legend_cfg,
        title=dict(text=title, font=dict(size=22))
    )

    fig = go.Figure(data=data, layout=layout)
    if save_path:
        fig.write_image(f'{save_path}.pdf')
        fig.write_image(f'{save_path}.svg')
    fig.show()

    return np.array(group_means).T, outlier_counts


def raincloud_plot(ax, data, x_pos, color, dot_size, width, alpha, cloud_width, rain_offset, jitter_width):
    """Helper function to create a raincloud plot for one group
    """
    # Remove NaN values for plotting
    data = data[~np.isnan(data)]
    
    if len(data) > 0:  # Only plot if there is data
        # Kernel density estimate (cloud)
        kernel = scipy.stats.gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 200)
        density = kernel(x_range)
        density = density / density.max() * cloud_width
        
        # Plot violin plot (cloud) - only half cloud
        ax.fill_betweenx(x_range, x_pos - density, x_pos, alpha=alpha, color=color)
        
        # Box plot
        box_props = dict(facecolor='white', edgecolor='black', linewidth=1.5)
        ax.boxplot(data, positions=[x_pos], widths=[width*0.3], 
                    patch_artist=True, boxprops=box_props, notch=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    showfliers=False, vert=True)
        
        # Scatter plot
        y = data
        x = np.repeat(x_pos + rain_offset, len(y))
        x_jitter = np.random.normal(0, jitter_width, size=len(y))
        ax.scatter(x + x_jitter, y, alpha=0.5, s=dot_size, color=color)


def plot_kde2d(scores, labels, box_distance=0.5, box_width=0.5, box_linewidth=0.8, maxalpha=0.8,
                         showfliers=False, num_scale=100,
                         colors=SUBTYPE_COLOR, group_names=SUBTYPE_NAME, 
                         save_path=None):
    
    plt.figure(figsize=(6,6))
    gs = plt.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                     hspace=0.01, wspace=0.01,
                     left=0.1, right=0.9, bottom=0.1, top=0.9)
    
    ax_top_box = plt.subplot(gs[0, 0])
    ax_right_box = plt.subplot(gs[1, 1])
    ax_scatter = plt.subplot(gs[1, 0])
    
    # Create DataFrame with mapped group names
    scores = num_scale * scores
    df = pd.DataFrame({
        'x': scores[:, 0],
        'y': scores[:, 1],
        'Subtype': [group_names[l] for l in labels]  # Map numeric labels to group names
    })

    # Define density levels and transparency
    alphas = [0.1, 0.3, maxalpha]
    levels_sets = [[0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]
        
    # Plot density contours (must use for loop for better color effect))
    for levels, alpha in zip(levels_sets, alphas):
        sns.kdeplot(
            data=df,
            x='x',
            y='y',
            hue='Subtype',
            hue_order=group_names,
            levels=levels,
            palette=colors,
            alpha=alpha,
            common_norm=False,
            ax=ax_scatter,
            fill=True,
            bw_adjust=1.0,
            legend=False
        )

    # Create legend to self-define
    handles = []
    for color in colors:
        handle = plt.Rectangle((0,0), 1, 1, color=color, alpha=maxalpha)
        handles.append(handle)

    ax_scatter.legend(
            handles,
            group_names,
            loc='lower right',
            title='',
            fontsize=10,
            title_fontsize=10,
            # bbox_to_anchor=(1.05, 1),
            frameon=True
        )

    # Create boxplots using seaborn
    # Right boxplot
    right_box = sns.boxplot(data=df, y='y', x='Subtype', hue='Subtype', hue_order=group_names,
               palette=colors, order=group_names, ax=ax_right_box, 
               width=box_width, linewidth=box_linewidth, boxprops = dict(edgecolor='black'),
               showfliers=showfliers, notch=True, legend=False)
    
    # Top boxplot
    top_box = sns.boxplot(data=df, x='x', y='Subtype', hue='Subtype', hue_order=group_names,
               palette=colors, order=group_names, ax=ax_top_box, 
               width=box_width, linewidth=box_linewidth, boxprops = dict(edgecolor='black'),
               showfliers=showfliers, notch=True, legend=False)
    
    for box in right_box.patches:
        box.set_alpha(maxalpha)  
    
    for box in top_box.patches:
        box.set_alpha(maxalpha) 

    # Adjust axes labels and ticks
    ax_scatter.set_xlabel(f'{DIMENSION[0]} (a.u.)', fontsize=12)
    ax_scatter.set_ylabel(f'{DIMENSION[1]} (a.u.)', fontsize=12)
    ax_scatter.tick_params(axis='both', which='major', labelsize=10)
    
    # Remove labels and ticks from boxplots
    ax_top_box.set(xticklabels=[], yticklabels=[], xlabel='', ylabel='')
    ax_right_box.set(xticklabels=[], yticklabels=[], xlabel='', ylabel='')

    ax_top_box.set_xticklabels([])
    ax_top_box.set_yticklabels([])
    ax_right_box.set_xticklabels([])
    ax_right_box.set_yticklabels([])
    
    # Sync axis limits
    xlim = ax_scatter.get_xlim()
    ylim = ax_scatter.get_ylim()
    ax_top_box.set_xlim(xlim)
    ax_right_box.set_ylim(ylim)
    
    # Remove unnecessary ticks
    ax_top_box.set_xticks([])
    ax_top_box.set_yticks([])
    ax_right_box.set_xticks([])
    ax_right_box.set_yticks([])

    # Remove unnecessary spines
    for ax in [ax_top_box, ax_right_box]:
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)

    # Adjust subplot positions using box_distance
    pos_top = ax_top_box.get_position()
    pos_right = ax_right_box.get_position()
    ax_top_box.set_position([pos_top.x0, pos_top.y0, pos_top.width, pos_top.height * box_distance])
    ax_right_box.set_position([pos_right.x0, pos_right.y0, pos_right.width * box_distance, pos_right.height])
    
    # Save figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #plt.savefig(f'{save_path}.svg', bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')


def plot_pivot_heatmap(pivot_df, xlabel, ylabel, title, cbar_label="score"):
    plt.figure()
    im = plt.imshow(pivot_df.values, aspect='auto', origin='lower')
    plt.xticks(np.arange(pivot_df.shape[1]), [f"{v:g}" for v in pivot_df.columns])
    plt.yticks(np.arange(pivot_df.shape[0]), [f"{v:g}" for v in pivot_df.index])
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(im, label=cbar_label)
    plt.tight_layout()
    plt.show()


def plot_icc(
    icc_view1, icc_view2,
    dim_idx,
    col_names=["Brain ICC", "Behavior ICC"],
    circle_color="#f47c20",
    circle_lw=4,
    text_color="#000000",
    font_size=13,
    label_size=14,
    spine_color="#000000",
    spine_lw=1.2,
    gap_pts=4,
    label_pad_pts=6,
    save_path=None
):
    """
    ICC grid plot with:
    - fixed per-cell physical size across row counts
    - gapless layout (cells touch exactly)
    - equal-aspect arcs matching original 2x2 appearance
    - extensible to arbitrary row/column counts
    """

    # ----- dimensions -----
    dim_idx = np.asarray(dim_idx)
    n_dim = len(dim_idx)
    n_col = len(col_names)

    # ----- fixed physical cell size -----
    cell_size_inch = 1.0
    fig_width  = cell_size_inch * n_col
    fig_height = cell_size_inch * n_dim
    fig = plt.figure(figsize=(fig_width, fig_height))

    # ----- gridspec matching the original layout -----
    gs = fig.add_gridspec(
        n_dim, n_col,
        left=0.15, right=0.85,
        top=0.85, bottom=0.15,
        wspace=0.0, hspace=0.0
    )

    # ----- value conversion -----
    def to_val(v):
        v = float(v)
        return v * 100 if v <= 1.2 else v

    def fmt(v):
        return f"{v:.2g}"

    data = np.vstack([
        [to_val(icc_view1[d]) for d in dim_idx],
        [to_val(icc_view2[d]) for d in dim_idx]
    ]).T

    # ----- axes -----
    axes = np.empty((n_dim, n_col), dtype=object)
    for i in range(n_dim):
        for j in range(n_col):
            ax = fig.add_subplot(gs[i, j])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])

            for side in ("left", "right", "top", "bottom"):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color(spine_color)
                ax.spines[side].set_linewidth(spine_lw)

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            axes[i, j] = ax

    fig.canvas.draw()

    # ----- arcs -----
    for i in range(n_dim):
        for j in range(n_col):
            ax = axes[i, j]
            ax_bbox = ax.get_window_extent()
            w_px, h_px = ax_bbox.width, ax_bbox.height

            margin_pts = circle_lw / 2 + gap_pts
            margin_px = margin_pts * (fig.dpi / 72)

            margin_x = 2 * (margin_px / w_px)
            margin_y = 2 * (margin_px / h_px)
            r = 1 - max(margin_x, margin_y)

            v = data[i, j]
            theta_start = 90
            theta_end = theta_start - (v / 100) * 360

            arc = Arc(
                (0, 0),
                width=2*r, height=2*r,
                theta1=theta_end, theta2=theta_start,
                linewidth=circle_lw, color=circle_color,
                capstyle="round"
            )
            ax.add_patch(arc)

            ax.text(
                0, 0, fmt(v / 100.0),
                ha="center", va="center",
                fontsize=font_size, color=text_color
            )

    # ----- row labels (right aligned) -----
    dim_names = [f"CD {d+1}" for d in dim_idx]
    fig.canvas.draw()

    row_y = []
    for i in range(n_dim):
        ymin, ymax = axes[i, -1].get_position(fig).get_points()[:, 1]
        row_y.append((ymin + ymax) / 2)

    x_right = axes[0, -1].get_position(fig).get_points()[1, 0]
    pad_x = label_pad_pts / (fig.get_size_inches()[0] * fig.dpi)

    for i in range(n_dim):
        fig.text(
            x_right + pad_x,
            row_y[i],
            dim_names[i],
            ha="left", va="center",
            fontsize=label_size
        )

    # ----- column labels (bottom) -----
    col_x = []
    for j in range(n_col):
        xmin, xmax = axes[-1, j].get_position(fig).get_points()[:, 0]
        col_x.append((xmin + xmax) / 2)

    pad_y = label_pad_pts / (fig.get_size_inches()[1] * fig.dpi)
    bottom_y = min(
        axes[-1, j].get_position(fig).y0 for j in range(n_col)
    ) - pad_y

    for j in range(n_col):
        fig.text(
            col_x[j], bottom_y,
            col_names[j],
            ha="right", va="top",
            fontsize=label_size, rotation=45
        )

    # ----- save -----
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.pdf", bbox_inches="tight")


def plot_brain_maps(model, output_dir):
    """Render the adopted Figure 2–4 maps from saved Top 20 display inputs."""
    from pathlib import Path
    import subprocess

    from matplotlib.colorbar import ColorbarBase
    from subcortex_visualization.plotting import plot_subcortical_data
    from cca401.preferred import CURRENT

    code = Path(__file__).resolve().parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_number = {"nc": 2, "fc": 3, "nc+fc": 4}[model]
    inputs = CURRENT / "figure_inputs"
    colors = ["#835bd6", "#ffffff", "#c8971c"]
    cmap = LinearSegmentedColormap.from_list("PurpleWhiteGold", colors)
    subcortical_order = [
        "hippocampus", "amygdala", "thalamus_posterior", "thalamus_anterior",
        "accumbens", "pallidum", "putamen", "caudate",
    ]

    if model == "nc+fc":
        records = pd.read_csv(inputs / "fig4_top20_records.csv")
        cortical_input = inputs / "fig4_top20_cortical_input.csv"
        cortical = pd.read_csv(cortical_input)
        modalities = (("struct", 0), ("func", 116))
    else:
        records = pd.read_csv(inputs / "fig23_top20_records.csv")
        records = records.loc[records.figure == figure_number]
        cortical_input = inputs / "fig23_top20_cortical_input.csv"
        modalities = (("struct" if model == "nc" else "func", 0),)

    for dimension in (1, 2):
        selected = records.loc[records.dimension == dimension]
        assert len(selected) == 20
        if model == "nc+fc":
            bounds = cortical.loc[cortical.cd == dimension, ["vmin", "vmax"]].drop_duplicates()
            values = np.zeros(232)
            for row in selected.itertuples(index=False):
                values[row.global_feature_index] = row.reference_loading
            assert np.count_nonzero(values) == 20
        else:
            bounds = selected[["color_min", "color_max"]].drop_duplicates()
            values = np.zeros(116)
            for row in selected.itertuples(index=False):
                values[row.feature_index] = row.reference_loading
        assert len(bounds) == 1
        vmin, vmax = bounds.iloc[0].tolist()

        for modality, offset in modalities:
            block = values[offset:offset + 116]
            frame = pd.DataFrame([
                {
                    "region": subcortical_order[index % 8],
                    "value": float(block[index]),
                    "Hemisphere": "R" if index < 8 else "L",
                }
                for index in range(16)
            ])
            figure = plot_subcortical_data(
                subcortex_data=frame, atlas="Melbourne_S1", line_thickness=1.5,
                line_color="black", hemisphere="both", fill_title="", cmap=cmap,
                vmin=vmin, vmax=vmax, midpoint=0, show_legend=False,
                show_figure=False,
            )
            if model == "nc+fc":
                figure.savefig(
                    output_dir / f"subcortical_cd{dimension}_{modality}.pdf",
                    bbox_inches="tight",
                )
            else:
                figure.savefig(
                    output_dir / f"fig{figure_number}_cd{dimension}_subcortical.pdf",
                    bbox_inches="tight", transparent=True,
                )
            plt.close(figure)

        if model != "nc+fc":
            figure, ax = plt.subplots(figsize=(4.0, 0.42))
            figure.subplots_adjust(left=0.04, right=0.96, bottom=0.50, top=0.90)
            zero_position = -vmin / (vmax - vmin)
            bar_cmap = LinearSegmentedColormap.from_list(
                "PurpleWhiteGoldLinearScale",
                [(0, colors[0]), (zero_position, colors[1]), (1, colors[2])],
            )
            bar = ColorbarBase(
                ax, cmap=bar_cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                orientation="horizontal", ticks=[vmin, 0, vmax],
            )
            bar.outline.set_visible(False)
            bar.ax.set_xticklabels([f"{vmin:.1f}", "0.0", f"{vmax:.1f}"])
            bar.ax.tick_params(axis="x", length=0, labelsize=9, pad=1)
            figure.savefig(
                output_dir / f"fig{figure_number}_cd{dimension}_colorbar.pdf",
                bbox_inches="tight", pad_inches=0.02, transparent=True,
            )
            plt.close(figure)

    r_home = Path("/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources")
    r_env = os.environ.copy()
    r_env["R_HOME"] = str(r_home)
    subprocess.run(
        [str(r_home / "bin/exec/R"), "--vanilla", "--slave", "-f",
         str(code / "R/cortical_loadings_nested.R"), "--args",
         "--input", str(cortical_input), "--output-dir", str(output_dir),
         "--model", model],
        check=True, cwd=code, env=r_env,
    )


def crop_cortical_views(figs_dir=None):
    """Crop four-view cortical rows into individual view PDFs for PowerPoint assembly.

    Parameters and geometry follow LNM_MMPI_code/figs/CORTICAL_CROPS.md.
    Source rows:
      - fig2/fig2_cd{1,2}_cortical.pdf
      - fig3/fig3_cd{1,2}_cortical.pdf
      - fig4/brain116_cd{1,2}_{sLNM,fLNM}_cortical.pdf
    Output directories:
      - fig2/cortical_views/
      - fig3/cortical_views/
      - fig4/cortical_views/
    """
    from pathlib import Path
    import sys

    try:
        from pypdf import Transformation
        import pypdf
    except (ImportError, AttributeError):
        sys.modules.pop('pypdf', None)
        for fallback in [
            "/Users/jy/Library/Python/3.9/lib/python/site-packages",
            "/opt/homebrew/lib/python3.11/site-packages",
        ]:
            if fallback not in sys.path and Path(fallback).exists():
                sys.path.insert(0, fallback)
        import pypdf
        from pypdf import Transformation

    if figs_dir is None:
        figs_dir = Path(__file__).resolve().parent / "figs"
    else:
        figs_dir = Path(figs_dir).resolve()

    # Standard crop rectangles from CORTICAL_CROPS.md
    std_crops = {
        1: (37.636, 103.317, 136.673, 97.375),
        2: (200.554, 103.317, 141.626, 97.375),
        3: (385.757, 103.317, 141.626, 97.375),
        4: (547.189, 103.317, 135.685, 97.375),
    }

    # Figure 2 CD1 small coordinate adjustments per CORTICAL_CROPS.md
    fig2_cd1_crops = {
        1: (37.635, 103.317, 136.679, 97.375),
        2: (200.560, 103.320, 141.630, 97.370),
        3: (385.757, 103.317, 141.626, 97.375),
        4: (547.189, 103.317, 135.685, 97.375),
    }

    def _crop_single_view(src_path, dst_path, left, bottom, width, height):
        reader = pypdf.PdfReader(str(src_path))
        writer = pypdf.PdfWriter()
        page = reader.pages[0]
        page.add_transformation(Transformation().translate(-left, -bottom))
        page.mediabox.lower_left = (0, 0)
        page.mediabox.upper_right = (width, height)
        page.cropbox.lower_left = (0, 0)
        page.cropbox.upper_right = (width, height)
        writer.add_page(page)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "wb") as f:
            writer.write(f)

    generated = []

    # 1. Figure 2
    fig2_dir = figs_dir / "fig2"
    fig2_cv = fig2_dir / "cortical_views"
    for cd in (1, 2):
        src = fig2_dir / f"fig2_cd{cd}_cortical.pdf"
        crops = fig2_cd1_crops if cd == 1 else std_crops
        for v in (1, 2, 3, 4):
            dst = fig2_cv / f"fig2_cd{cd}_view{v}.pdf"
            l, b, w, h = crops[v]
            _crop_single_view(src, dst, l, b, w, h)
            generated.append(dst)

    # 2. Figure 3
    fig3_dir = figs_dir / "fig3"
    fig3_cv = fig3_dir / "cortical_views"
    for cd in (1, 2):
        src = fig3_dir / f"fig3_cd{cd}_cortical.pdf"
        for v in (1, 2, 3, 4):
            dst = fig3_cv / f"fig3_cd{cd}_view{v}.pdf"
            l, b, w, h = std_crops[v]
            _crop_single_view(src, dst, l, b, w, h)
            generated.append(dst)

    # 3. Figure 4
    fig4_dir = figs_dir / "fig4"
    fig4_cv = fig4_dir / "cortical_views"
    panels = [
        ("cd1", "struct", "sLNM", [1, 2, 3, 4]),
        ("cd1", "func", "fLNM", [1, 2, 3, 4]),
        ("cd2", "struct", "sLNM", [1, 2, 3, 4]),
        ("cd2", "func", "fLNM", [1, 2, 3, 4]),
    ]
    for cd_label, modality_name, lnm_name, views in panels:
        src = fig4_dir / f"brain116_{cd_label}_{lnm_name}_cortical.pdf"
        for v in views:
            dst = fig4_cv / f"fig4_{cd_label}_{modality_name}_view{v}.pdf"
            l, b, w, h = std_crops[v]
            _crop_single_view(src, dst, l, b, w, h)
            generated.append(dst)

    return generated


def plot_subcortical(weights, 
                   title="", 
                   line_thickness=1.5, 
                   line_color='black', 
                   save_path=None, 
                   cmap=None,
                   show_label=False,
                   vmin=None,
                   vmax=None,
                   midpoint=0,
                   show_colorbar=True):
    """
    Visualize weights for the Melbourne S1 Subcortex Atlas (16 regions).

    Parameters:
    ----------
    weights : np.array or list
        Vector of length 16. 
        Order: [Rh-HIP, Rh-AMY, Rh-pTHA, Rh-aTHA, Rh-NAc, Rh-GP, Rh-PUT, Rh-CAU, Lh-HIP...]
    title : str
        Title of the colorbar/plot. Default is "plot_subcortical".
    line_thickness : float
        Thickness of the outline.
    line_color : str
        Color of the outline.
    save_path : str
        Path prefix to save the figure (e.g., './figs/result'). The code will append '.pdf'.
    cmap : str
        Matplotlib colormap name.
    show_label : bool
        If True, generates an additional reference plot (atlas only).
    vmin : float or None
        Lower bound for colormap normalization.
    vmax : float or None
        Upper bound for colormap normalization.
    midpoint : float or None
        Midpoint used for diverging normalization.
    show_colorbar : bool
        Whether to draw colorbar in the subcortical figure.
    """
    from subcortex_visualization.plotting import plot_subcortical_data

    # 1. Input validation
    weights = np.array(weights)
    if weights.shape[0] != 16:
        raise ValueError(f"Input vector length must be 16, got {weights.shape[0]}")

    # 2. Define mapping (Melbourne S1 official names)
    # Order matches the provided specific sequence: 
    # HIP, AMY, pTHA, aTHA, NAc, GP, PUT, CAU
    region_order = [
        'hippocampus',          # 1
        'amygdala',             # 2
        'thalamus_posterior',   # 3
        'thalamus_anterior',    # 4
        'accumbens',            # 5
        'pallidum',             # 6
        'putamen',              # 7
        'caudate'               # 8
    ]
    
    data_list = []

    for i in range(16):
        # 0-7 is Right Hemisphere (R), 8-15 is Left Hemisphere (L)
        hemi = 'R' if i < 8 else 'L'
        
        # Cycle through region names
        region_name = region_order[i % 8]
        val = weights[i]
        
        data_list.append({
            'region': region_name,
            'value': val,
            'Hemisphere': hemi
        })

    # 3. Convert to DataFrame
    plot_df = pd.DataFrame(data_list)

    target_pdf = None
    if save_path:
        target_pdf = f"{save_path}.pdf"
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # 4. Reference Plot (If requested)
    if show_label:
        plot_subcortical_data(
            atlas='Melbourne_S1', 
            hemisphere='both', 
            fill_title="", 
            line_color=line_color, 
            line_thickness=line_thickness,
            cmap='viridis',
            show_figure=True
        )

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("PurpleWhiteGold", ['#835bd6', "white", '#c8971c'])
    # 5. Main Data Plotting
    fig = plot_subcortical_data(
        subcortex_data=plot_df,
        atlas='Melbourne_S1', 
        line_thickness=line_thickness,
        line_color=line_color,
        hemisphere='both',
        fill_title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        midpoint=midpoint,
        show_legend=show_colorbar,
        show_figure=False
    )

    # 6. Save using returned matplotlib Figure.
    if target_pdf:
        fig.savefig(target_pdf, bbox_inches='tight')

    # Keep previous behavior: display figure in notebook.
    plt.show()


def plot_cortical_network_legend(
    save_path=None,
    order="canonical",
    include_subcortex=False,
    use_full_names=False,
    markersize=8.0,
    markeredgewidth=1.5,
    fontsize=9.0,
    labelspacing=0.35,
    handlelength=1.0,
    handletextpad=0.25,
    borderpad=0.05,
    transparent=True,
    padding_pt=1.5,
):
    """Render a standalone vertical network legend with hollow square colored borders.

    Matches the parcel outline colors used in the Figure 2–4 cortical maps
    (Schaefer 100 7-network parcellation).

    Parameters
    ----------
    save_path : str or Path, optional
        Path to save the generated PDF. If provided without .pdf suffix, it is added.
    order : str, default 'canonical'
        'canonical': VN at top down to DMN at bottom (standard Yeo network ordering).
        'hierarchical': DMN at top down to VN at bottom (gradient cool->warm / transmodal->unimodal).
    include_subcortex : bool, default False
        Whether to include Subcortex ('SC', #4A3432). Default is False because cortical maps
        only outline the 7 cortical networks.
    use_full_names : bool, default False
        Whether to spell out full network names instead of abbreviations.
    markersize : float, default 8.0
        Size of the square marker in points.
    markeredgewidth : float, default 1.5
        Stroke width of the square colored border.
    fontsize : float, default 9.0
        Font size for network labels.
    labelspacing : float, default 0.35
        Vertical spacing between legend entries.
    handlelength : float, default 1.0
        Length of legend handle area in font units. Kept compact for standalone markers.
    handletextpad : float, default 0.25
        Padding between the square marker and label text.
    borderpad : float, default 0.05
        Padding inside the legend border in font units.
    transparent : bool, default True
        Whether the figure background is transparent.
    padding_pt : float, default 1.5
        Padding around the tight bounding box in points.

    Returns
    -------
    tuple of (Figure, Path or None)
    """
    network_color_map = build_brain_network_color_map(list(PALLETE))

    if order == "canonical":
        cats = ["VN", "SMN", "DAN", "VAN", "LN", "FPCN", "DMN"]
        if include_subcortex:
            cats.insert(0, "SC")
    elif order == "hierarchical":
        cats = ["DMN", "FPCN", "LN", "VAN", "DAN", "SMN", "VN"]
        if include_subcortex:
            cats.append("SC")
    else:
        raise ValueError(f"Unknown order: {order!r}. Choose 'canonical' or 'hierarchical'.")

    full_names_map = {
        "DMN": "Default Mode",
        "FPCN": "Frontoparietal Control",
        "LN": "Limbic",
        "VAN": "Ventral Attention",
        "DAN": "Dorsal Attention",
        "SMN": "Somatomotor",
        "VN": "Visual",
        "SC": "Subcortical",
    }

    labels = [full_names_map.get(c, c) if use_full_names else c for c in cats]

    handles = [
        plt.Line2D(
            [0], [0],
            marker="s",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor=network_color_map[c],
            markeredgewidth=markeredgewidth,
            markersize=markersize,
            label=lbl,
        )
        for c, lbl in zip(cats, labels)
    ]

    fig = plt.figure(figsize=(1.0, 2.0))
    leg = fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        frameon=False,
        fontsize=fontsize,
        handlelength=handlelength,
        handletextpad=handletextpad,
        borderpad=borderpad,
        labelspacing=labelspacing,
    )

    out_file = None
    if save_path:
        import pathlib
        out_file = pathlib.Path(save_path).resolve()
        if out_file.suffix.lower() != ".pdf":
            out_file = out_file.with_suffix(".pdf")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if padding_pt > 0:
            bbox = bbox.padded(padding_pt / 72.0)
        fig.savefig(out_file, bbox_inches=bbox, transparent=transparent)

    return fig, out_file

