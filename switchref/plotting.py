"""Plotting helpers."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Optional default for sub_buses; experiments may override.
SUB_BUSES_DEFAULT = [0, 17, 18, 19, 20, 21, 22, 23]


def plot_dQ_and_V(
    record_dQ,
    record_V,
    record_Vref=None,
    *,
    bus_colors,
    dc_buses,
    dt=0.1,
    vmin=-0.05,
    vmax=0.05,
    plot_buses=None,
    show_legend=False,
    legend_ncol=4,
    save_path=None,
    publication_style=False,
):
    """
    Plot deltaQ, voltage deviation, and optional reference voltage.

    Differences from the original notebook function:
      - ``bus_colors`` is required (was a global in the notebook).
      - ``dc_buses`` is required (was a global with a default).
    Output and return values match the notebook one-for-one.

    ``publication_style=True`` switches to column-width publication
    settings: smaller figsize, CMR serif via LaTeX usetex, ~9pt fonts,
    thinner lines. Default ``False`` keeps the validation style (large
    figures, sans-serif, big fonts).
    """
    record_dQ = np.asarray(record_dQ, dtype=np.float32)
    record_V = np.asarray(record_V, dtype=np.float32)
    if record_Vref is not None:
        record_Vref = np.asarray(record_Vref, dtype=np.float32)

    Tq, n = record_dQ.shape
    Tv, nv = record_V.shape
    assert nv == n
    if record_Vref is not None:
        Tr, nr = record_Vref.shape
        assert nr == n

    if plot_buses is None:
        plot_buses = list(range(n))
    else:
        plot_buses = list(plot_buses)
        assert all(0 <= i < n for i in plot_buses), \
            "plot_buses contains invalid bus index"

    t_q = dt * np.arange(1, Tq + 1)
    t_v = dt * np.arange(Tv)

    if publication_style:
        # Column-width publication settings: CMR serif, ~9pt fonts,
        # thinner lines, LaTeX usetex so math renders in Computer Modern.
        rc_overrides = {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
            "font.family": "serif",
        }
        # Save at full text-width (~7") 1:1 with the print size; the
        # exp figures are placed in figure* (double-column) in main.tex
        # with \includegraphics[width=0.95\linewidth] so the printed
        # figure ends up roughly 6.7" wide, close to 1:1 scaling.
        figsize = (7.0, 1.7)
        save_dpi = 300
        TITLE_FS, LABEL_FS, TICK_FS = 10, 10, 8
        LEGEND_FS = 8
        DC_LW, NON_DC_LW = 1.0, 0.5
        BAND_LW = 0.7
    else:
        rc_overrides = {}
        figsize = (18, 5.5)
        save_dpi = 120
        TITLE_FS, LABEL_FS, TICK_FS = 29, 27, 25
        LEGEND_FS = 19
        DC_LW, NON_DC_LW = 1.8, 0.8
        BAND_LW = 1.2

    # Apply rcParams overrides (publication mode only) and remember
    # the originals so we can restore them in the finally block --
    # otherwise text.usetex / font.family would leak into subsequent
    # matplotlib calls in the same Python session.
    _saved_rc = {k: plt.rcParams[k] for k in rc_overrides}
    if rc_overrides:
        plt.rcParams.update(rc_overrides)

    fig, axes = plt.subplots(
        1,
        3 if record_Vref is not None else 2,
        figsize=figsize,
        dpi=save_dpi,
    )
    axes = np.atleast_1d(axes)

    dQ_ylim = (-0.011, 0.011)
    dQ_yticks = [-0.01, -0.005, 0.0, 0.005, 0.01]

    V_ylim = (-0.1, 0.1)
    V_yticks = [-0.1, -0.05, 0.0, 0.05, 0.1]

    def plus_one_formatter(x, pos):
        return f"{x + 1:.2f}"

    def style_axis(ax):
        ax.tick_params(labelsize=TICK_FS, direction="in")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def line_style(i):
        if i in dc_buses:
            return dict(lw=DC_LW, alpha=1.0, zorder=3)
        return dict(dashes=(20, 10), lw=NON_DC_LW, alpha=1.0, zorder=1)

    def finish_axis(ax, ylabel, ylim, yticks):
        ax.set_ylabel(ylabel, fontsize=TITLE_FS + 4)
        ax.set_xlabel("Time (s)", fontsize=LABEL_FS)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        style_axis(ax)

    def draw_bus_curves(ax, t, data, add_labels=False):
        # Draw non-DC buses first.
        non_dc = [i for i in plot_buses if i not in dc_buses]
        dc = [i for i in plot_buses if i in dc_buses]

        for i in non_dc:
            ax.plot(
                t,
                data[:, i],
                color=bus_colors[i],
                label=f"Bus {i+2}" if add_labels else None,
                **line_style(i),
            )

        # Draw DC buses on top.
        for i in dc:
            ax.plot(
                t,
                data[:, i],
                color=bus_colors[i],
                label=f"Bus {i+2}" if add_labels else None,
                **line_style(i),
            )

    axQ = axes[0]
    draw_bus_curves(axQ, t_q, record_dQ, add_labels=False)
    finish_axis(axQ, r"$\Delta q\ (\mathrm{p.u.})$", dQ_ylim, dQ_yticks)

    axV = axes[1]
    axV.axhspan(vmin, vmax, color="0.85", alpha=0.2, zorder=0)
    axV.axhline(vmax, color="0.6", ls="--", lw=BAND_LW, zorder=1)
    axV.axhline(vmin, color="0.6", ls="--", lw=BAND_LW, zorder=1)
    draw_bus_curves(axV, t_v, record_V, add_labels=show_legend)
    finish_axis(axV, r"$v\,(\mathrm{p.u.})$", V_ylim, V_yticks)

    axV.yaxis.set_major_formatter(FuncFormatter(plus_one_formatter))

    if show_legend:
        handles, labels = axV.get_legend_handles_labels()

        order = np.argsort([int(lbl.split()[-1]) for lbl in labels])
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]

        # Anchor the LOWER edge of the legend just above the axes top
        # (axes_y = 1.02), so the legend sits OUTSIDE the data area
        # rather than overlapping the trace spikes that reach 1.05.
        leg = axV.legend(
            handles, labels,
            fontsize=LEGEND_FS, ncol=legend_ncol,
            loc="lower center", bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            handlelength=1.5, handletextpad=0.5, columnspacing=1.2,
            borderpad=0.3, labelspacing=0.3,
        )

    if record_Vref is not None:
        axR = axes[2]
        axR.axhspan(vmin, vmax, color="0.85", alpha=0.2, zorder=0)
        axR.axhline(vmax, color="0.6", ls="--", lw=BAND_LW, zorder=1)
        axR.axhline(vmin, color="0.6", ls="--", lw=BAND_LW, zorder=1)
        draw_bus_curves(axR, t_q, record_Vref, add_labels=show_legend)
        finish_axis(axR, r"$v_{\mathrm{ref}}$ (p.u.)", V_ylim, V_yticks)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    plt.show()

    # Restore rcParams (publication mode only).
    if rc_overrides:
        plt.rcParams.update(_saved_rc)

    if record_Vref is None:
        return record_dQ, record_V
    return record_dQ, record_V, record_Vref


def plot_publication_grid_2x2(
    runs,
    *,
    bus_colors,
    plot_buses,
    save_path,
    dt=0.1,
    vmin=-0.05,
    vmax=0.05,
    legend_ncol=4,
):
    """Render four runs grouped by scenario, with each scenario block
    showing v above Delta q.

    Layout (4 rows of panels x 2 columns of controllers):

        +-------------------- shared bus legend ---------------------+
        |  Fixed-reference            |   Switching-reference        |
        +--- Single DC block ---------+------------------------------+
        |  v,        Single DC, fixed |  v,        Single DC, switch |
        |  Delta q,  Single DC, fixed |  Delta q,  Single DC, switch |
        +--- Two DC block ------------+------------------------------+
        |  v,        Two DC,    fixed |  v,        Two DC,    switch |
        |  Delta q,  Two DC,    fixed |  Delta q,  Two DC,    switch |
        +-------------------------------------------------------------+

    Putting v above Delta q in each block emphasizes the regulation
    outcome (the band-compliance story) before the control cost.

    ``runs`` is a sequence of four dicts (order: single-DC fixed,
    single-DC switching, two-DC fixed, two-DC switching), each with:
      - ``dQ``       : (T, n) array of reactive increments
      - ``V``        : (T, n) array of voltage deviation (already v - 1)
      - ``dc_buses`` : list of "DC" bus indices to draw on top + bold
      - ``label``    : short label (now unused for inner panel titles,
                       kept for backward compatibility)

    ``bus_colors`` and ``plot_buses`` follow ``plot_dQ_and_V``'s contract.
    """
    if len(runs) != 4:
        raise ValueError(f"Expected 4 runs, got {len(runs)}")

    rc_overrides = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
        "font.family": "serif",
    }
    _saved_rc = {k: plt.rcParams[k] for k in rc_overrides}
    plt.rcParams.update(rc_overrides)

    TITLE_FS, LABEL_FS, TICK_FS = 10, 9, 7
    LEGEND_FS = 8
    DC_LW, NON_DC_LW = 1.0, 0.5
    BAND_LW = 0.6

    fig = plt.figure(figsize=(7.0, 5.0), dpi=300)
    # Outer grid:
    #   row 0:  legend strip
    #   row 1:  column-header strip ("Fixed-reference" / "Switching-reference")
    #   rows 2-3: single-DC block (v above Delta q), tight pair
    #   row 4:  empty spacer to visually separate the two scenario blocks
    #   rows 5-6: two-DC    block (v above Delta q), tight pair
    outer = fig.add_gridspec(
        7, 2,
        height_ratios=[0.55, 0.10, 1.0, 1.0, 0.30, 1.0, 1.0],
        hspace=0.18, wspace=0.10,
        left=0.085, right=0.985, top=0.98, bottom=0.06,
    )

    legend_ax = fig.add_subplot(outer[0, :])
    legend_ax.axis("off")
    # Column-header strip in row 1.
    header_axL = fig.add_subplot(outer[1, 0]); header_axL.axis("off")
    header_axR = fig.add_subplot(outer[1, 1]); header_axR.axis("off")
    header_axL.text(0.5, 0.0, "Fixed-reference",
                     ha="center", va="bottom", fontsize=TITLE_FS,
                     transform=header_axL.transAxes)
    header_axR.text(0.5, 0.0, "Switching-reference",
                     ha="center", va="bottom", fontsize=TITLE_FS,
                     transform=header_axR.transAxes)

    dQ_ylim = (-0.011, 0.011)
    dQ_yticks = [-0.01, 0.0, 0.01]
    V_ylim = (-0.1, 0.1)
    V_yticks = [-0.05, 0.0, 0.05]

    def plus_one_formatter(x, pos):
        return f"{x + 1:.2f}"

    def style_axis(ax):
        ax.tick_params(labelsize=TICK_FS, direction="in", length=2.5, width=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.8)

    # Row indices: each scenario block stacks v on top of Delta q.
    # Single-DC block: v in row 2, Delta q in row 3.
    # Two-DC    block: v in row 5, Delta q in row 6.
    panel_map = [
        # (run_idx, dQ_row, v_row, col)
        (0, 3, 2, 0),
        (1, 3, 2, 1),
        (2, 6, 5, 0),
        (3, 6, 5, 1),
    ]

    # Create all axes upfront so we can apply sharex consistently and
    # control which panels show tick labels.
    axQs = {}
    axVs = {}
    for run_idx, qr, vr, col in panel_map:
        # Delta-q axes share x-axis within the same column to align
        # ticks; v axes also share x-axis within the same column.
        # We also share Delta-q and v x-axes via the bottom v axis as
        # the master per column (we'll do that with set_xlim below).
        axQs[run_idx] = fig.add_subplot(outer[qr, col])
        axVs[run_idx] = fig.add_subplot(outer[vr, col])

    legend_handles, legend_labels = None, None

    for run_idx, qr, vr, col in panel_map:
        run = runs[run_idx]
        dQ = np.asarray(run["dQ"], dtype=np.float32)
        V = np.asarray(run["V"], dtype=np.float32)
        dc_buses = list(run["dc_buses"])

        Tq = dQ.shape[0]
        Tv = V.shape[0]
        t_q = dt * np.arange(1, Tq + 1)
        t_v = dt * np.arange(Tv)

        axQ = axQs[run_idx]
        axV = axVs[run_idx]

        def line_style(i, dcs=dc_buses):
            if i in dcs:
                return dict(lw=DC_LW, alpha=1.0, zorder=3)
            return dict(dashes=(20, 10), lw=NON_DC_LW, alpha=1.0, zorder=1)

        def draw_curves(ax, t, data, add_labels, dcs=dc_buses):
            non_dc = [i for i in plot_buses if i not in dcs]
            dc_list = [i for i in plot_buses if i in dcs]
            for i in non_dc:
                ax.plot(t, data[:, i], color=bus_colors[i],
                        label=f"Bus {i+2}" if add_labels else None,
                        **line_style(i, dcs))
            for i in dc_list:
                ax.plot(t, data[:, i], color=bus_colors[i],
                        label=f"Bus {i+2}" if add_labels else None,
                        **line_style(i, dcs))

        # Delta-q panel sits at the bottom of each scenario block.
        # Only the very last row (two-DC Delta q, qr == 6) carries the
        # "Time (s)" label and tick labels; the single-DC Delta-q row
        # (qr == 3) suppresses both so the single-DC block visually
        # closes without orphan tick numbers.
        draw_curves(axQ, t_q, dQ, add_labels=False)
        axQ.set_ylim(*dQ_ylim)
        axQ.set_yticks(dQ_yticks)
        if qr == 6:
            axQ.set_xlabel("Time (s)", fontsize=LABEL_FS)
        else:
            axQ.tick_params(labelbottom=False)
        style_axis(axQ)

        # v panel with shaded band.
        axV.axhspan(vmin, vmax, color="0.85", alpha=0.2, zorder=0)
        axV.axhline(vmax, color="0.6", ls="--", lw=BAND_LW, zorder=1)
        axV.axhline(vmin, color="0.6", ls="--", lw=BAND_LW, zorder=1)
        draw_curves(axV, t_v, V, add_labels=(run_idx == 0))
        axV.set_ylim(*V_ylim)
        axV.set_yticks(V_yticks)
        axV.yaxis.set_major_formatter(FuncFormatter(plus_one_formatter))
        style_axis(axV)

        # v panels sit at the top of each scenario block; the Delta-q
        # row below carries the time axis, so always suppress labelbottom
        # here.
        axV.tick_params(labelbottom=False)

        if run_idx == 0:
            legend_handles, legend_labels = axV.get_legend_handles_labels()
            order = np.argsort([int(lbl.split()[-1]) for lbl in legend_labels])
            legend_handles = [legend_handles[i] for i in order]
            legend_labels = [legend_labels[i] for i in order]
            n_h = len(legend_handles)
            nrows = -(-n_h // legend_ncol)
            cm_to_rm = []
            for c in range(legend_ncol):
                for r in range(nrows):
                    src = r * legend_ncol + c
                    if src < n_h:
                        cm_to_rm.append(src)
            legend_handles = [legend_handles[i] for i in cm_to_rm]
            legend_labels = [legend_labels[i] for i in cm_to_rm]

    # Row labels (y-axis); only left column carries them.  Two-line
    # form keeps the labels narrow enough to leave room for tick labels
    # ("0.01" / "1.05") without colliding across rows.
    axQs[0].set_ylabel("Single DC\n" + r"$\Delta q$ (p.u.)", fontsize=LABEL_FS)
    axQs[2].set_ylabel("Two DC\n"    + r"$\Delta q$ (p.u.)", fontsize=LABEL_FS)
    axVs[0].set_ylabel("Single DC\n" + r"$v$ (p.u.)",        fontsize=LABEL_FS)
    axVs[2].set_ylabel("Two DC\n"    + r"$v$ (p.u.)",        fontsize=LABEL_FS)
    # Right column: suppress y-tick labels (axis ticks still drawn).
    for run_idx in (1, 3):
        axQs[run_idx].tick_params(labelleft=False)
        axVs[run_idx].tick_params(labelleft=False)

    # Column headers are rendered in the dedicated header strip
    # (`header_axL` / `header_axR`) above.

    if legend_handles:
        # Anchor the legend to the upper portion of legend_ax so its
        # bottom border sits well above the column-header row below.
        legend_ax.legend(
            legend_handles, legend_labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.0),
            ncol=legend_ncol,
            fontsize=LEGEND_FS, frameon=True,
            edgecolor="0.75", facecolor="white",
            handlelength=1.5, handletextpad=0.4, columnspacing=1.6,
            borderpad=0.5, labelspacing=0.30,
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    plt.close(fig)
    plt.rcParams.update(_saved_rc)


def plot_publication_scenario_2col(
    runs,
    *,
    bus_colors,
    plot_buses,
    save_path,
    dt=0.1,
    vmin=-0.05,
    vmax=0.05,
    legend_ncol=4,
    figsize=(7.0, 2.8),
):
    """Per-scenario double-column figure: two runs (fixed-reference and
    switching-reference) side-by-side, with v above Delta q in each
    column.

    Layout (2 rows of panels x 2 columns of controllers):

        +-------------------- shared bus legend ---------------------+
        |  Fixed-reference            |   Switching-reference        |
        +-----------------------------+------------------------------+
        |  v,        fixed            |  v,        switching         |
        |  Delta q,  fixed            |  Delta q,  switching         |
        +-------------------------------------------------------------+

    ``runs`` is a sequence of exactly two dicts (fixed first, switching
    second), each carrying ``dQ`` / ``V`` / ``dc_buses``.
    """
    if len(runs) != 2:
        raise ValueError(f"Expected 2 runs (fixed, switching), got {len(runs)}")

    rc_overrides = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
        "font.family": "serif",
    }
    _saved_rc = {k: plt.rcParams[k] for k in rc_overrides}
    plt.rcParams.update(rc_overrides)

    TITLE_FS, LABEL_FS, TICK_FS = 10, 9, 7
    LEGEND_FS = 8
    DC_LW = 0.8
    BAND_LW = 0.6

    fig = plt.figure(figsize=figsize, dpi=300)
    # v gets a noticeably taller PLOT area than Delta q because the
    # voltage trajectory is the main regulation outcome that the
    # reader inspects, while Delta q (reactive effort) is secondary.
    # The Delta-q row also pays for the x-tick labels and the
    # "Time (s)" xlabel, which further compresses its plot height.
    outer = fig.add_gridspec(
        4, 2,
        height_ratios=[0.55, 0.22, 1.20, 0.90],
        hspace=0.18, wspace=0.10,
        left=0.12 if figsize[0] < 5 else 0.06,
        right=0.985, top=0.98, bottom=0.11,
    )

    legend_ax = fig.add_subplot(outer[0, :]); legend_ax.axis("off")
    header_axL = fig.add_subplot(outer[1, 0]); header_axL.axis("off")
    header_axR = fig.add_subplot(outer[1, 1]); header_axR.axis("off")
    # Column-header text is positioned AFTER the panels are drawn so we
    # can center it over the actual plot-area x-center (rather than the
    # gridspec column center, which includes the y-tick-label space and
    # would shift the header slightly off-center over the panel).

    def plus_one_formatter(x, pos):
        return f"{x + 1:.2f}"

    def style_axis(ax):
        ax.tick_params(labelsize=TICK_FS, direction="in", length=2.5, width=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.8)

    axQs = {}
    axVs = {}
    for col in (0, 1):
        axVs[col] = fig.add_subplot(outer[2, col])
        axQs[col] = fig.add_subplot(outer[3, col])

    # Non-DC traces are styled identically across all panels so the
    # shared legend reads the same regardless of DC count.
    NON_DC_STYLE = dict(lw=0.60, alpha=0.65, zorder=1)  # solid

    legend_handles, legend_labels = None, None
    for col, run in enumerate(runs):
        dQ = np.asarray(run["dQ"], dtype=np.float32)
        V  = np.asarray(run["V"],  dtype=np.float32)
        dc_buses = list(run["dc_buses"])

        Tq = dQ.shape[0]; Tv = V.shape[0]
        t_q = dt * np.arange(1, Tq + 1)
        t_v = dt * np.arange(Tv)

        axQ = axQs[col]
        axV = axVs[col]

        # Two-DC distinction: keep the first DC trace at full width and
        # put subsequent DC traces on a thinner solid line. Earlier iterations
        # added a dash on the secondary DC; the dash helped v slightly but
        # fragmented Delta q's short vertical impulses into "dotted bars",
        # so it hurt overall. The lineweight differential alone is enough at
        # this trace density.
        _DC_STYLES = [
            dict(lw=DC_LW,        alpha=0.90, zorder=3),
            dict(lw=DC_LW * 0.70, alpha=1.00, zorder=3),
            dict(lw=DC_LW * 0.55, alpha=1.00, zorder=3),
        ]
        def line_style(i, dcs=dc_buses, _nondc=NON_DC_STYLE):
            if i in dcs:
                return dict(_DC_STYLES[dcs.index(i) % len(_DC_STYLES)])
            return dict(_nondc)

        def draw_curves(ax, t, data, add_labels, dcs=dc_buses):
            non_dc = [i for i in plot_buses if i not in dcs]
            dc_list = [i for i in plot_buses if i in dcs]
            for i in non_dc:
                ax.plot(t, data[:, i], color=bus_colors[i],
                        label=f"Bus {i+2}" if add_labels else None,
                        **line_style(i, dcs))
            for i in dc_list:
                ax.plot(t, data[:, i], color=bus_colors[i],
                        label=f"Bus {i+2}" if add_labels else None,
                        **line_style(i, dcs))

        # v panel (top of each column block).
        axV.axhspan(vmin, vmax, color="0.85", alpha=0.2, zorder=0)
        axV.axhline(vmax, color="0.6", ls="--", lw=BAND_LW, zorder=1)
        axV.axhline(vmin, color="0.6", ls="--", lw=BAND_LW, zorder=1)
        draw_curves(axV, t_v, V, add_labels=(col == 0))
        axV.set_ylim(-0.1, 0.1)
        axV.set_yticks([-0.05, 0.0, 0.05])
        axV.yaxis.set_major_formatter(FuncFormatter(plus_one_formatter))
        axV.tick_params(labelbottom=False)
        style_axis(axV)

        # Delta q panel (bottom).
        draw_curves(axQ, t_q, dQ, add_labels=False)
        axQ.set_ylim(-0.011, 0.011)
        axQ.set_yticks([-0.01, 0.0, 0.01])
        axQ.set_xlabel("Time (s)", fontsize=LABEL_FS)
        style_axis(axQ)

        if col == 0:
            # Build legend handles fresh with full opacity, mirroring
            # the per-DC-bus distinct line styles used in the panels
            # (solid for the first DC bus, dashed for the second, etc.).
            legend_handles = []
            legend_labels  = []
            for i in plot_buses:
                if i in dc_buses:
                    style = dict(_DC_STYLES[dc_buses.index(i) % len(_DC_STYLES)])
                    style.pop("zorder", None)
                    style["alpha"] = 1.0
                    handle = Line2D([0], [0], color=bus_colors[i], **style)
                else:
                    handle = Line2D([0], [0], color=bus_colors[i],
                                    lw=NON_DC_STYLE["lw"], alpha=1.0)
                legend_handles.append(handle)
                legend_labels.append(f"Bus {i+2}")
            order = np.argsort([int(lbl.split()[-1]) for lbl in legend_labels])
            legend_handles = [legend_handles[i] for i in order]
            legend_labels  = [legend_labels[i]  for i in order]
            n_h = len(legend_handles)
            nrows = -(-n_h // legend_ncol)
            cm_to_rm = []
            for c in range(legend_ncol):
                for r in range(nrows):
                    src = r * legend_ncol + c
                    if src < n_h:
                        cm_to_rm.append(src)
            legend_handles = [legend_handles[i] for i in cm_to_rm]
            legend_labels  = [legend_labels[i]  for i in cm_to_rm]

    axVs[0].set_ylabel(r"$v$ (p.u.)",         fontsize=LABEL_FS)
    axQs[0].set_ylabel(r"$\Delta q$ (p.u.)",  fontsize=LABEL_FS)
    for col in (1,):
        axQs[col].tick_params(labelleft=False)
        axVs[col].tick_params(labelleft=False)

    # Align the y-labels of v and Delta q at the same x position
    # (otherwise the wider "-0.01" tick label pushes the Delta-q
    # ylabel further left than the v ylabel).
    fig.align_ylabels([axVs[0], axQs[0]])

    if legend_handles:
        legend_ax.legend(
            legend_handles, legend_labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.0),
            ncol=legend_ncol,
            fontsize=LEGEND_FS, frameon=True,
            edgecolor="0.75", facecolor="white",
            handlelength=1.5, handletextpad=0.4, columnspacing=1.6,
            borderpad=0.5, labelspacing=0.30,
        )

    # Position column headers centered on the actual plot-area x-center
    # (computed after layout), anchored at the vertical center of the
    # header row so they sit safely below the legend and above the v
    # panel without overlapping either.
    fig.canvas.draw()
    for col, header_text in [(0, "Fixed-reference"), (1, "Switching-reference")]:
        bbox_plot = axVs[col].get_position()
        x_center = 0.5 * (bbox_plot.x0 + bbox_plot.x1)
        bbox_hdr = header_axL.get_position() if col == 0 else header_axR.get_position()
        y_center = 0.5 * (bbox_hdr.y0 + bbox_hdr.y1)
        fig.text(x_center, y_center, header_text,
                 ha="center", va="center", fontsize=TITLE_FS)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.rcParams.update(_saved_rc)


def plot_publication_loadsmooth_sc(
    t_pwr, p_pwr,
    dQ, V,
    *,
    bus_colors,
    plot_buses,
    dc_buses,
    save_path,
    dt=0.1,
    vmin=-0.05,
    vmax=0.05,
    smoothing_onset=None,
    legend_ncol=4,
):
    """Single-column figure for the load-smoothing scenario:
    p_dc, Delta q, and v stacked vertically with a shared time axis and
    a compact framed bus legend at the top.

    Column-width layout: figsize=(3.5, ~3.7), CMR serif via usetex,
    ~9pt fonts.
    """
    rc_overrides = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
        "font.family": "serif",
    }
    _saved_rc = {k: plt.rcParams[k] for k in rc_overrides}
    plt.rcParams.update(rc_overrides)

    LABEL_FS, TICK_FS, LEGEND_FS = 10, 8, 8
    DC_LW, NON_DC_LW = 1.0, 0.5
    BAND_LW = 0.6

    fig = plt.figure(figsize=(3.5, 3.7), dpi=300)
    # Panel order top-to-bottom: legend, p_dc, v, Delta q.
    # v gets noticeably more vertical PLOT area than Delta q because
    # voltage compliance is the main regulation outcome the reader
    # inspects (Delta q is the secondary "control cost").
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[0.30, 1.0, 1.20, 0.90],
        hspace=0.18,
        left=0.13, right=0.985, top=0.99, bottom=0.10,
    )
    legend_ax = fig.add_subplot(gs[0, 0])
    legend_ax.axis("off")
    axP = fig.add_subplot(gs[1, 0])
    axV = fig.add_subplot(gs[2, 0], sharex=axP)
    axQ = fig.add_subplot(gs[3, 0], sharex=axP)

    def style_axis(ax):
        ax.tick_params(labelsize=TICK_FS, direction="in", length=2.5, width=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.8)

    # Power panel.
    axP.plot(np.asarray(t_pwr), np.asarray(p_pwr),
             color="0.18", lw=0.85)
    axP.set_ylabel(r"$L_{\mathrm{dc}}$ (p.u.)", fontsize=LABEL_FS)
    axP.tick_params(labelbottom=False)
    # Show only the 1.0 reference tick on the P_dc axis; sub-1.0 grid
    # values are not informative once the trace is biased into a
    # realistic [idle, 1.0] band. Label as "1.0" (not the matplotlib
    # default "1") to match the voltage panels.
    axP.set_yticks([1.0])
    axP.set_yticklabels(["1.0"])
    # Y-axis starts at 0 (full per-unit reference frame); top margin
    # set just above the data peak.
    _p_arr = np.asarray(p_pwr)
    _p_max = float(_p_arr.max())
    axP.set_ylim(0.0, _p_max + 0.05)
    style_axis(axP)

    dQ = np.asarray(dQ, dtype=np.float32)
    V = np.asarray(V, dtype=np.float32)
    Tq = dQ.shape[0]
    Tv = V.shape[0]
    t_q = dt * np.arange(1, Tq + 1)
    t_v = dt * np.arange(Tv)

    dc_list = list(dc_buses)

    def line_style(i):
        if i in dc_list:
            return dict(lw=DC_LW, alpha=0.80, zorder=3)
        return dict(dashes=(20, 10), lw=NON_DC_LW, alpha=0.40, zorder=1)

    def draw_curves(ax, t, data, add_labels):
        non_dc = [i for i in plot_buses if i not in dc_list]
        dc = [i for i in plot_buses if i in dc_list]
        for i in non_dc:
            ax.plot(t, data[:, i], color=bus_colors[i],
                    label=f"Bus {i+2}" if add_labels else None,
                    **line_style(i))
        for i in dc:
            ax.plot(t, data[:, i], color=bus_colors[i],
                    label=f"Bus {i+2}" if add_labels else None,
                    **line_style(i))

    # v panel (middle, sits between p_dc and Delta q).
    axV.axhspan(vmin, vmax, color="0.85", alpha=0.2, zorder=0)
    axV.axhline(vmax, color="0.6", ls="--", lw=BAND_LW, zorder=1)
    axV.axhline(vmin, color="0.6", ls="--", lw=BAND_LW, zorder=1)
    draw_curves(axV, t_v, V, add_labels=True)
    axV.set_ylabel(r"$v$ (p.u.)", fontsize=LABEL_FS)
    axV.set_ylim(-0.1, 0.1)
    axV.set_yticks([-0.05, 0.0, 0.05])
    axV.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x+1:.2f}"))
    axV.tick_params(labelbottom=False)
    style_axis(axV)

    # Delta q panel (bottom, carries the "Time (s)" xlabel).
    draw_curves(axQ, t_q, dQ, add_labels=False)
    axQ.set_ylabel(r"$\Delta q$ (p.u.)", fontsize=LABEL_FS)
    axQ.set_ylim(-0.011, 0.011)
    axQ.set_yticks([-0.01, 0.0, 0.01])
    axQ.set_xlabel("Time (s)", fontsize=LABEL_FS)
    style_axis(axQ)

    # Build legend handles fresh with full opacity rather than reusing
    # the plotted-line handles (whose alpha is 0.4 / 0.8 for visual
    # blending in the trace panels).
    handles = []
    labels  = []
    for i in plot_buses:
        if i in dc_list:
            h = Line2D([0], [0], color=bus_colors[i], lw=DC_LW, alpha=1.0)
        else:
            h = Line2D([0], [0], color=bus_colors[i], lw=NON_DC_LW,
                       dashes=(20, 10), alpha=1.0)
        handles.append(h)
        labels.append(f"Bus {i+2}")
    order = np.argsort([int(lbl.split()[-1]) for lbl in labels])
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    n = len(handles)
    nrows = -(-n // legend_ncol)
    cm_to_rm = []
    for c in range(legend_ncol):
        for r in range(nrows):
            src = r * legend_ncol + c
            if src < n:
                cm_to_rm.append(src)
    handles = [handles[i] for i in cm_to_rm]
    labels = [labels[i] for i in cm_to_rm]
    legend_ax.legend(
        handles, labels,
        loc="center", ncol=legend_ncol,
        fontsize=LEGEND_FS, frameon=True,
        edgecolor="0.75", facecolor="white",
        handlelength=1.5, handletextpad=0.4, columnspacing=1.0,
        borderpad=0.3, labelspacing=0.25,
    )

    # Draw a single dotted vertical line at the smoothing onset that
    # spans all three subplots (continuous through the inter-panel
    # gaps), rather than three separate per-axis lines that stop at
    # each subplot boundary.
    if smoothing_onset is not None:
        fig.canvas.draw()
        x_disp = axP.transData.transform((smoothing_onset, 0))[0]
        x_fig = fig.transFigure.inverted().transform((x_disp, 0))[0]
        y_top = axP.get_position().y1
        y_bot = axQ.get_position().y0
        fig.add_artist(Line2D(
            [x_fig, x_fig], [y_bot, y_top],
            transform=fig.transFigure,
            color="0.55", ls=":", lw=0.7, zorder=10,
        ))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.rcParams.update(_saved_rc)


def plot_Q_from_dQ(dQ, Q_init, dc_bus, bus_aux, dt, *, bus_colors):
    """
    Plot deltaQ(t) and the resulting Q(t) trajectory implied by
    Q(t) = Q_init + sum_{tau<=t} dQ(tau).

    Differences from notebook:
      - ``Q_init`` is now an explicit parameter (was ``env.Q0``).
      - ``bus_colors`` is required (was a global).
    """
    dQ = np.asarray(dQ, dtype=np.float32)
    Q_init = np.asarray(Q_init, dtype=np.float32).reshape(-1)

    t_dQ = dt * np.arange(dQ.shape[0])
    Q_traj = np.vstack([Q_init, Q_init + np.cumsum(dQ, axis=0)]).astype(np.float32)
    t_Q = dt * np.arange(Q_traj.shape[0])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4), dpi=100)

    for i in range(dQ.shape[1]):
        ax0.plot(t_dQ, dQ[:, i], color=bus_colors[i], lw=0.8, alpha=0.8)
    for i in bus_aux:
        ax0.plot(t_dQ, dQ[:, i], color=bus_colors[i], lw=1.8, alpha=1.0, zorder=3)
    dc_line, = ax0.plot(t_dQ, dQ[:, dc_bus], color=bus_colors[dc_bus],
                        lw=1.2, alpha=0.6, zorder=4)
    dc_line.set_dashes([10, 6])
    ax0.set(xlabel="time (s)", title=r"$\Delta Q$")

    for i in range(Q_traj.shape[1]):
        ax1.plot(t_Q, Q_traj[:, i], color=bus_colors[i], lw=1.0, ls=":", alpha=1.0)
    for i in bus_aux:
        ax1.plot(t_Q, Q_traj[:, i], color=bus_colors[i], lw=3.0, ls=":", zorder=3)
    ax1.plot(t_Q, Q_traj[:, dc_bus], color=bus_colors[dc_bus], lw=3.0, alpha=0.6, zorder=4)
    ax1.set(xlabel="time (s)", title=r"$Q$")

    plt.tight_layout()
    plt.show()


def plot_power_preprocessing(time_raw, pwr_raw, time_new, pwr_new, dP, dt_new):
    """Plot raw power, interpolated grid, and first difference. (Cell 42.)"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 4), sharex=True)

    axes[0].plot(time_raw, pwr_raw, lw=1.0, alpha=0.6, label="raw power")
    axes[0].plot(time_new, pwr_new, lw=1.8, label=f"interpolated (dt={dt_new}s)")
    axes[0].set_ylabel("Power")
    axes[0].grid(alpha=0.3)

    dP_right = np.concatenate(([np.nan], dP))
    axes[1].plot(time_new, dP_right, lw=1.5)
    axes[1].set_ylabel("dP")
    axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()
