"""Interactive PySide viewer for 500-step forecast falloff results.

Usage:
  python scripts/falloff_viewer.py
  python scripts/falloff_viewer.py --results results/ram_improve/falloff_500
  python scripts/falloff_viewer.py --mode lab
  python scripts/falloff_viewer.py --mode live_lab
  python scripts/falloff_viewer.py --mode ary_compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEFAULT_RESULTS = ROOT / "results" / "ram_improve" / "falloff_500"

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except ImportError as exc:
    print("Install viewer deps: pip install PySide6 pyqtgraph", file=sys.stderr)
    raise SystemExit(1) from exc

from src.aryan.feature_schema242 import FEATURE_BLOCKS_242, FEATURE_COLS_242  # noqa: E402

MODEL_COLORS = {
    "ary_base": "#9aa0a6",
    "ary_V10": "#e67e22",
    "timesfm": "#3498db",
    "actual": "#f0f6fc",
    "context": "#6e7681",
}
FORECAST_MODELS = {
    "ary_base": "ARY base",
    "ary_V10": "ARY V10",
    "timesfm": "TimesFM",
}
SERIES_LABELS = {
    **FORECAST_MODELS,
    "actual": "Ground truth",
    "context": "Context (500 steps)",
}
BLOCK_FOR_FEATURE: dict[str, str] = {}
for block, names in FEATURE_BLOCKS_242.items():
    for name in names:
        BLOCK_FOR_FEATURE[name] = block


def feature_title(fidx: int, name: str | None = None) -> str:
    name = name or FEATURE_COLS_242[fidx]
    block = BLOCK_FOR_FEATURE.get(name, "?")
    return f"feat[{fidx:03d}] · {name} · block {block}"


class CrosshairReadout(QtWidgets.QFrame):
    """Hover readout — shows nearest-step values with more precision when zoomed."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 4px; }"
        )
        self.label = QtWidgets.QLabel("Hover a plot to inspect step-level values")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self.label)

    def set_html(self, html: str) -> None:
        self.label.setText(html)


class DetailPlotWidget(pg.PlotWidget):
    """Plot with crosshair, zoom-aware markers, and no downsampling."""

    def __init__(self, readout: CrosshairReadout, plot_kind: str, **kwargs):
        super().__init__(**kwargs)
        self.readout = readout
        self.plot_kind = plot_kind  # "error" | "timeline"
        self._series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._scatter: dict[str, pg.ScatterPlotItem] = {}
        self._vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ffffff55", width=1))
        self._hline = pg.InfiniteLine(angle=0, pen=pg.mkPen("#ffffff33", width=1))
        self.addItem(self._vline, ignoreBounds=True)
        self.addItem(self._hline, ignoreBounds=True)
        self._vline.hide()
        self._hline.hide()
        self.getViewBox().sigRangeChanged.connect(self._update_marker_density)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.setMouseEnabled(x=True, y=True)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.getAxis("bottom").enableAutoSIPrefix(False)
        self.getAxis("left").enableAutoSIPrefix(False)

    def clear_plot(self) -> None:
        """Clear data items but keep crosshair infrastructure."""
        self._series.clear()
        for item in self._scatter.values():
            self.removeItem(item)
        self._scatter.clear()
        plot_item = self.getPlotItem()
        keep = {self._vline, self._hline}
        for item in list(plot_item.items):
            if item not in keep:
                plot_item.removeItem(item)
        self._vline.hide()
        self._hline.hide()
        if plot_item.legend is None:
            self.addLegend(offset=(10, 10))

    def set_series(self, series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self._series = series
        for item in self._scatter.values():
            self.removeItem(item)
        self._scatter.clear()
        for key, (x, y) in series.items():
            scatter = pg.ScatterPlotItem(
                size=6,
                pen=pg.mkPen(MODEL_COLORS.get(key, "#ffffff"), width=1),
                brush=pg.mkBrush(MODEL_COLORS.get(key, "#ffffff")),
                symbol="o",
            )
            scatter.setZValue(10)
            scatter.hide()
            self.addItem(scatter)
            self._scatter[key] = scatter
        self._update_marker_density()

    def _x_span(self) -> float:
        x_range = self.getViewBox().viewRange()[0]
        return float(x_range[1] - x_range[0])

    def _update_marker_density(self) -> None:
        span = self._x_span()
        if span <= 0:
            return
        # More zoom → show individual step markers (never downsample lines).
        show_markers = span <= 120
        show_labels = span <= 35
        for key, (x, y) in self._series.items():
            scatter = self._scatter.get(key)
            if scatter is None or len(x) == 0:
                continue
            if not show_markers:
                scatter.hide()
                continue
            x_range = self.getViewBox().viewRange()[0]
            mask = (x >= x_range[0]) & (x <= x_range[1])
            xv, yv = x[mask], y[mask]
            if len(xv) == 0:
                scatter.hide()
                continue
            # When very zoomed, show every point; mid zoom thin to ~1 marker per pixel column.
            if span > 35:
                stride = max(1, int(len(xv) / max(span, 1)))
                xv, yv = xv[::stride], yv[::stride]
            size = 9 if show_labels else max(4, min(8, int(120 / max(span, 1))))
            scatter.setData(x=xv, y=yv, size=size)
            scatter.show()

        # Finer tick spacing when zoomed in.
        axis = self.getAxis("bottom")
        if span <= 20:
            axis.setTickSpacing(major=1, minor=0.25)
        elif span <= 60:
            axis.setTickSpacing(major=5, minor=1)
        elif span <= 150:
            axis.setTickSpacing(major=10, minor=2)
        else:
            axis.setTickSpacing(major=50, minor=10)

    def _nearest_index(self, x_arr: np.ndarray, x_val: float) -> int:
        return int(np.clip(np.searchsorted(x_arr, x_val), 0, len(x_arr) - 1))

    def _on_mouse_moved(self, pos) -> None:
        if not self._series or not self.sceneBoundingRect().contains(pos):
            self._vline.hide()
            self._hline.hide()
            return
        mouse = self.getViewBox().mapSceneToView(pos)
        mx, my = float(mouse.x()), float(mouse.y())
        self._vline.setPos(mx)
        self._hline.setPos(my)
        self._vline.show()
        self._hline.show()

        # Prefer "actual" / first series for index lookup.
        ref_key = "actual" if "actual" in self._series else next(iter(self._series))
        x_ref, _ = self._series[ref_key]
        idx = self._nearest_index(x_ref, mx)
        step_x = float(x_ref[idx])

        lines = [f"<b>{self.plot_kind.title()} · x={step_x:.0f}</b>"]
        prec = 4 if self._x_span() <= 30 else (2 if self._x_span() <= 100 else 1)
        for key, (x_arr, y_arr) in self._series.items():
            j = self._nearest_index(x_arr, step_x)
            label = SERIES_LABELS.get(key, key)
            color = MODEL_COLORS.get(key, "#e6edf3")
            lines.append(
                f'<span style="color:{color}"><b>{label}</b></span>: '
                f"{float(y_arr[j]):.{prec}f}"
            )
        if self.plot_kind == "error" and "actual" in self._series:
            pass  # error plot has no actual series
        self.readout.set_html("<br>".join(lines))


class NumericTableItem(QtWidgets.QTableWidgetItem):
    """Sort by numeric UserRole, not display string."""

    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:
        a = self.data(QtCore.Qt.ItemDataRole.UserRole)
        b = other.data(QtCore.Qt.ItemDataRole.UserRole)
        if a is not None and b is not None:
            return int(a) < int(b)
        return super().__lt__(other)


class FalloffViewer(QtWidgets.QMainWindow):
    def __init__(self, results_dir: Path):
        super().__init__()
        self.results_dir = results_dir
        self._payloads: dict[int, dict] = {}
        self._summary: dict = {}
        self._current_fidx: int | None = None

        self.setWindowTitle("Forecast Falloff Viewer — 500 ctx → 500 forecast")
        self.resize(1380, 900)
        pg.setConfigOptions(antialias=True, background="#161b22", foreground="#e6edf3")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # --- left: feature table ---
        left = QtWidgets.QVBoxLayout()
        search_row = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Filter by index, name, or block (e.g. A_meta, num_flows)…")
        self.search.textChanged.connect(self._filter_table)
        search_row.addWidget(self.search)
        left.addLayout(search_row)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Idx", "Feature name", "ARY↓", "TFM↓", "Δ", "Winner"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setStyleSheet(
            "QTableWidget { color: #e6edf3; gridline-color: #30363d; }"
            "QHeaderView::section { background: #21262d; color: #e6edf3; }"
        )
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.itemDoubleClicked.connect(lambda _: self._reset_plot_views())
        left.addWidget(self.table)

        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        left.addWidget(self.summary_label)

        left_w = QtWidgets.QWidget()
        left_w.setLayout(left)
        left_w.setMinimumWidth(400)
        layout.addWidget(left_w)

        # --- right: plots + controls ---
        right = QtWidgets.QVBoxLayout()

        self.feature_header = QtWidgets.QLabel("Select a feature")
        self.feature_header.setStyleSheet("font-size: 15px; font-weight: 700; color: #e6edf3;")
        self.feature_header.setWordWrap(True)
        right.addWidget(self.feature_header)

        ctrl = QtWidgets.QHBoxLayout()
        self.model_checks: dict[str, QtWidgets.QCheckBox] = {}
        for key, label in FORECAST_MODELS.items():
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_plots)
            color = MODEL_COLORS[key]
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: 600; }}")
            self.model_checks[key] = cb
            ctrl.addWidget(cb)
        self.show_actual = QtWidgets.QCheckBox("Ground truth")
        self.show_actual.setChecked(True)
        self.show_actual.stateChanged.connect(self._refresh_plots)
        self.show_actual.setStyleSheet(f"QCheckBox {{ color: {MODEL_COLORS['actual']}; font-weight: 600; }}")
        ctrl.addWidget(self.show_actual)
        self.show_context = QtWidgets.QCheckBox("Context window")
        self.show_context.setChecked(True)
        self.show_context.stateChanged.connect(self._refresh_plots)
        ctrl.addWidget(self.show_context)
        self.show_falloff = QtWidgets.QCheckBox("Falloff markers")
        self.show_falloff.setChecked(True)
        self.show_falloff.stateChanged.connect(self._refresh_plots)
        ctrl.addWidget(self.show_falloff)
        reset_btn = QtWidgets.QPushButton("Reset zoom")
        reset_btn.clicked.connect(self._reset_plot_views)
        ctrl.addWidget(reset_btn)
        ctrl.addStretch()
        right.addLayout(ctrl)

        self.readout = CrosshairReadout()
        right.addWidget(self.readout)

        self.error_plot = DetailPlotWidget(self.readout, "error", title="Per-step |error|")
        self.error_plot.setLabel("bottom", "Forecast step (1 = first step ahead)")
        self.error_plot.setLabel("left", "|error|")
        right.addWidget(self.error_plot, stretch=1)

        self.timeline_plot = DetailPlotWidget(self.readout, "timeline", title="Timeline")
        self.timeline_plot.setLabel("bottom", "Timeline step")
        self.timeline_plot.setLabel("left", "Feature value")
        right.addWidget(self.timeline_plot, stretch=1)

        hint = QtWidgets.QLabel(
            "Zoom: scroll or drag · Pan: middle-click drag · "
            "Markers appear when zoomed in · Double-click row to reset view"
        )
        hint.setStyleSheet("color: #8b949e; font-size: 11px;")
        right.addWidget(hint)

        self.detail_label = QtWidgets.QLabel("")
        self.detail_label.setWordWrap(True)
        right.addWidget(self.detail_label)

        right_w = QtWidgets.QWidget()
        right_w.setLayout(right)
        layout.addWidget(right_w, stretch=1)

        self._load_data()
        self.table.setSortingEnabled(True)
        self.table.sortItems(0, QtCore.Qt.SortOrder.AscendingOrder)
        if self.table.rowCount():
            self.table.selectRow(0)
        self._refresh_plots()

    def _load_data(self) -> None:
        summary_path = self.results_dir / "run_summary.json"
        if summary_path.exists():
            self._summary = json.loads(summary_path.read_text())
            rows = self._summary.get("features", [])
        else:
            rows = []
            for path in sorted(self.results_dir.glob("feat*/falloff.json")):
                p = json.loads(path.read_text())
                m = p["models"]
                a, t = m["ary_base"]["falloff_step"], m["timesfm"]["falloff_step"]
                delta = (a - t) if (a and t) else None
                rows.append(
                    {
                        "feature_idx": p["feature_idx"],
                        "feature_name": p["feature_name"],
                        "ary_falloff": a,
                        "timesfm_falloff": t,
                        "falloff_delta_ary_minus_tfm": delta,
                    }
                )

        # Load payloads directly from disk (robust to summary/path mismatches).
        for path in sorted(self.results_dir.glob("feat*/falloff.json")):
            payload = json.loads(path.read_text())
            self._payloads[int(payload["feature_idx"])] = payload

        self.table.setRowCount(len(rows))
        fg = QtGui.QColor("#e6edf3")
        for r, row in enumerate(rows):
            fidx = row["feature_idx"]
            fname = row["feature_name"]
            block = BLOCK_FOR_FEATURE.get(fname, "")
            a, t = row.get("ary_falloff"), row.get("timesfm_falloff")
            delta = row.get("falloff_delta_ary_minus_tfm")
            if delta is None and a and t:
                delta = a - t
            if a and t:
                winner = "ARY" if a > t else ("TimesFM" if t > a else "tie")
            elif a and not t:
                winner = "ARY"
            elif t and not a:
                winner = "TimesFM"
            else:
                winner = "—"

            display_name = f"{fname}  ({block})" if block else fname
            items = [
                f"{fidx:03d}",
                display_name,
                str(a) if a else "—",
                str(t) if t else "—",
                str(delta) if delta is not None else "—",
                winner,
            ]
            for c, text in enumerate(items):
                item = NumericTableItem(text) if c == 0 else QtWidgets.QTableWidgetItem(text)
                item.setForeground(fg)
                if c == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, fidx)
                if c == 1:
                    item.setToolTip(feature_title(fidx, fname))
                if winner == "ARY" and c == 5:
                    item.setForeground(QtGui.QColor("#3fb950"))
                elif winner == "TimesFM" and c == 5:
                    item.setForeground(QtGui.QColor("#58a6ff"))
                self.table.setItem(r, c, item)

        self.table.setColumnWidth(0, 42)
        self.table.setColumnWidth(2, 52)
        self.table.setColumnWidth(3, 52)
        self.table.setColumnWidth(4, 36)
        self.table.setColumnWidth(5, 64)

        wins = self._summary.get("falloff_wins", {})
        if wins:
            self.summary_label.setText(
                f"Results: {self.results_dir}\n"
                f"Falloff wins (holds longer): ARY {wins.get('ary', 0)} · "
                f"TimesFM {wins.get('timesfm', 0)} · tie {wins.get('tie', 0)}"
            )
        else:
            self.summary_label.setText(f"Results: {self.results_dir} · {len(rows)} features")

    def _filter_table(self, text: str) -> None:
        text = text.lower()
        for r in range(self.table.rowCount()):
            idx = self.table.item(r, 0).text().lower()
            name = self.table.item(r, 1).text().lower()
            show = not text or text in name or text in idx
            self.table.setRowHidden(r, not show)

    def _selected_fidx(self) -> int | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        if item is None:
            return None
        val = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return int(val) if val is not None else None

    def _on_select(self) -> None:
        self._refresh_plots()

    def _reset_plot_views(self) -> None:
        self.error_plot.autoRange()
        self.timeline_plot.autoRange()

    def _plot_line(
        self,
        plot: DetailPlotWidget,
        x: np.ndarray,
        y: np.ndarray,
        key: str,
        width: float = 2.0,
        z: int = 1,
    ) -> pg.PlotDataItem:
        color = MODEL_COLORS.get(key, "#e6edf3")
        label = SERIES_LABELS.get(key, key)
        pen = pg.mkPen(color, width=width)
        curve = plot.plot(x, y, pen=pen, name=label)
        curve.setDownsampling(auto=False)
        curve.setClipToView(True)
        curve.setZValue(z)
        return curve

    def _refresh_plots(self) -> None:
        try:
            self._refresh_plots_inner()
        except Exception as exc:
            self.feature_header.setText("Plot error")
            self.detail_label.setText(f"<span style='color:#f85149'>Error rendering plots: {exc}</span>")
            import traceback
            traceback.print_exc()

    def _refresh_plots_inner(self) -> None:
        self.error_plot.clear_plot()
        self.timeline_plot.clear_plot()

        fidx = self._selected_fidx()
        if fidx is None or fidx not in self._payloads:
            self.feature_header.setText("Select a feature")
            self.detail_label.setText(
                f"No data loaded for feature index {fidx}."
                if fidx is not None
                else "Select a feature from the table."
            )
            return

        self._current_fidx = fidx
        p = self._payloads[fidx]
        fname = p["feature_name"]
        title = feature_title(fidx, fname)
        self.setWindowTitle(f"Falloff Viewer — {title}")
        self.feature_header.setText(title)

        t_end = p["anchor_t_end"]
        horizon = p["horizon"]
        context_len = p["context"]
        models = p["models"]
        actual = np.array(p["actual_series"])
        context = np.array(p.get("context_series", []))

        steps = np.arange(1, horizon + 1, dtype=float)
        fc_x = np.arange(t_end + 1, t_end + 1 + horizon, dtype=float)

        err_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key in FORECAST_MODELS:
            if not self.model_checks[key].isChecked():
                continue
            err = np.array(models[key]["per_step_abs_err"], dtype=float)
            self._plot_line(self.error_plot, steps, err, key)
            err_series[key] = (steps, err)
            if self.show_falloff.isChecked():
                fo = models[key]["falloff_step"]
                if fo:
                    line = pg.InfiniteLine(
                        pos=fo,
                        angle=90,
                        pen=pg.mkPen(MODEL_COLORS[key], style=QtCore.Qt.PenStyle.DotLine, width=1.5),
                    )
                    self.error_plot.addItem(line)

        self.error_plot.setTitle(f"|error| per forecast step — {fname}")
        self.error_plot.set_series(err_series)

        tl_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self.show_context.isChecked() and len(context):
            ctx_x = np.arange(t_end - context_len + 1, t_end + 1, dtype=float)
            self._plot_line(self.timeline_plot, ctx_x, context, "context", width=1.0, z=0)
            tl_series["context"] = (ctx_x, context)
        if self.show_actual.isChecked():
            self._plot_line(self.timeline_plot, fc_x, actual, "actual", width=3.0, z=5)
            tl_series["actual"] = (fc_x, actual)

        vline = pg.InfiniteLine(
            pos=t_end,
            angle=90,
            pen=pg.mkPen("#ffffff", style=QtCore.Qt.PenStyle.DashLine, width=1),
        )
        self.timeline_plot.addItem(vline)

        for key in FORECAST_MODELS:
            if not self.model_checks[key].isChecked():
                continue
            pred = np.array(models[key]["forecast"], dtype=float)
            self._plot_line(self.timeline_plot, fc_x, pred, key)
            tl_series[key] = (fc_x, pred)
            if self.show_falloff.isChecked():
                fo = models[key]["falloff_step"]
                if fo:
                    self.timeline_plot.addItem(
                        pg.InfiniteLine(
                            pos=t_end + fo,
                            angle=90,
                            pen=pg.mkPen(MODEL_COLORS[key], style=QtCore.Qt.PenStyle.DotLine, width=1.5),
                        )
                    )

        self.timeline_plot.setTitle(f"Timeline — {fname}  (gray=context, black=actual)")
        self.timeline_plot.set_series(tl_series)

        a = models["ary_base"]["falloff_step"]
        t = models["timesfm"]["falloff_step"]
        delta = (a - t) if (a and t) else None
        mae_a = models["ary_base"]["mean_abs_err"]
        mae_t = models["timesfm"]["mean_abs_err"]
        winner = ""
        if a and t:
            if a > t:
                winner = f"Falloff: ARY holds {a - t} step(s) longer than TimesFM"
            elif t > a:
                winner = f"Falloff: TimesFM holds {t - a} step(s) longer than ARY"
            else:
                winner = "Falloff: both models cross threshold at the same step"
        mae_winner = "ARY" if mae_a < mae_t else ("TimesFM" if mae_t < mae_a else "tie")
        self.detail_label.setText(
            f"Block <b>{BLOCK_FOR_FEATURE.get(fname, '?')}</b> · anchor timeline step {t_end} · "
            f"{p.get('attack_steps_in_future', '?')} attack steps in forecast window<br>"
            f"Falloff step — ARY: {a or '—'} · TimesFM: {t or '—'} · Δ(ary−tfm): "
            f"{delta if delta is not None else '—'} · {winner}<br>"
            f"Mean |error| — ARY: {mae_a:.3g} · TimesFM: {mae_t:.3g} · lower-MSE winner: <b>{mae_winner}</b>"
        )
        self.error_plot.enableAutoRange()
        self.timeline_plot.enableAutoRange()


LAB_RESULTS = ROOT / "results" / "ram_improve" / "lab_attack"
LAB_VARIANTS = {
    "ary_frozen": ("ARY frozen", "#9aa0a6"),
    "ary_v8_ram": ("ARY V8 RAM", "#e67e22"),
    "ary_v10_ram": ("ARY V10 RAM", "#f39c12"),
    "timesfm_ary_cls": ("TimesFM + ARY cls", "#3498db"),
}


class LabAttackViewer(QtWidgets.QMainWindow):
    """Interactive viewer for lab attack-detection eval (TimesFM+ARY vs RAM)."""

    def __init__(self, results_dir: Path):
        super().__init__()
        self.results_dir = results_dir
        self._datasets: dict[str, dict] = {}
        self._current: dict | None = None

        self.setWindowTitle("Lab Attack Eval — TimesFM+ARY vs RAM")
        self.resize(1380, 900)
        pg.setConfigOptions(antialias=True, background="#161b22", foreground="#e6edf3")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_change)
        left.addWidget(QtWidgets.QLabel("Dataset"))
        left.addWidget(self.dataset_combo)

        self.variant_table = QtWidgets.QTableWidget(0, 6)
        self.variant_table.setHorizontalHeaderLabels(
            ["Variant", "Bin F1", "Prec", "Rec", "Mitre F1", "Dyn MSE"]
        )
        self.variant_table.horizontalHeader().setStretchLastSection(True)
        self.variant_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.variant_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.variant_table.itemSelectionChanged.connect(self._refresh_plots)
        left.addWidget(self.variant_table)

        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        left.addWidget(self.summary_label)

        left_w = QtWidgets.QWidget()
        left_w.setLayout(left)
        left_w.setMinimumWidth(380)
        layout.addWidget(left_w)

        right = QtWidgets.QVBoxLayout()
        self.header = QtWidgets.QLabel("Lab attack detection")
        self.header.setStyleSheet("font-size: 15px; font-weight: 700;")
        right.addWidget(self.header)

        ctrl = QtWidgets.QHBoxLayout()
        self.show_threshold = QtWidgets.QCheckBox("Decision threshold (0.5)")
        self.show_threshold.setChecked(True)
        self.show_threshold.stateChanged.connect(self._refresh_plots)
        ctrl.addWidget(self.show_threshold)
        self.show_truth = QtWidgets.QCheckBox("Ground-truth attacks")
        self.show_truth.setChecked(True)
        self.show_truth.stateChanged.connect(self._refresh_plots)
        ctrl.addWidget(self.show_truth)
        reset_btn = QtWidgets.QPushButton("Reset zoom")
        reset_btn.clicked.connect(self._reset_views)
        ctrl.addWidget(reset_btn)
        ctrl.addStretch()
        right.addLayout(ctrl)

        self.readout = CrosshairReadout()
        right.addWidget(self.readout)

        self.prob_plot = DetailPlotWidget(self.readout, "P(attack)")
        self.prob_plot.setLabel("bottom", "Timeline step")
        self.prob_plot.setLabel("left", "P(attack)")
        right.addWidget(self.prob_plot, stretch=1)

        self.dyn_plot = DetailPlotWidget(self.readout, "Dynamics MSE")
        self.dyn_plot.setLabel("bottom", "Timeline step")
        self.dyn_plot.setLabel("left", "1-step MSE")
        right.addWidget(self.dyn_plot, stretch=1)

        self.detail_label = QtWidgets.QLabel("")
        self.detail_label.setWordWrap(True)
        right.addWidget(self.detail_label)

        right_w = QtWidgets.QWidget()
        right_w.setLayout(right)
        layout.addWidget(right_w, stretch=1)

        self._load_data()

    def _load_data(self) -> None:
        for path in sorted(self.results_dir.glob("*/eval.json")):
            payload = json.loads(path.read_text())
            self._datasets[payload["dataset"]] = payload
        self.dataset_combo.clear()
        for name in self._datasets:
            self.dataset_combo.addItem(name)
        if not self._datasets:
            self.summary_label.setText(f"No eval.json under {self.results_dir}")

    def _on_dataset_change(self, name: str) -> None:
        if name not in self._datasets:
            return
        self._current = self._datasets[name]
        p = self._current
        self.header.setText(
            f"{name} test · {p['n_windows']} windows · attack rate {p['attack_rate']:.0%}"
        )
        rows = p.get("summary_table", [])
        self.variant_table.setRowCount(len(rows))
        fg = QtGui.QColor("#e6edf3")
        for r, row in enumerate(rows):
            vname = row["variant"]
            label = LAB_VARIANTS.get(vname, (vname, "#ccc"))[0]
            vals = [
                label,
                f"{row.get('binary_f1', 0):.3f}",
                f"{row.get('binary_precision', 0):.3f}",
                f"{row.get('binary_recall', 0):.3f}",
                f"{row.get('mitre_f1_macro', 0):.3f}",
                f"{row.get('dynamics_mse', 0):.1f}",
            ]
            for c, text in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(text)
                item.setForeground(fg)
                if c == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, vname)
                    color = LAB_VARIANTS.get(vname, (vname, "#ccc"))[1]
                    item.setForeground(QtGui.QColor(color))
                self.variant_table.setItem(r, c, item)
        self.variant_table.selectAll()
        self.summary_label.setText(
            f"TimesFM+ARY cls uses ARY classifier on real windows; dynamics MSE from TimesFM 1-step.\n"
            f"RAM variants: V8=frozen+mem, V10=TTT+mem (classify-blend)."
        )
        self._refresh_plots()

    def _selected_variants(self) -> list[str]:
        rows = self.variant_table.selectionModel().selectedRows()
        out = []
        for row in rows:
            item = self.variant_table.item(row.row(), 0)
            if item:
                v = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if v:
                    out.append(str(v))
        return out

    def _reset_views(self) -> None:
        self.prob_plot.autoRange()
        self.dyn_plot.autoRange()

    def _refresh_plots(self) -> None:
        try:
            self._refresh_plots_inner()
        except Exception as exc:
            self.detail_label.setText(f"<span style='color:#f85149'>{exc}</span>")
            import traceback
            traceback.print_exc()

    def _refresh_plots_inner(self) -> None:
        self.prob_plot.clear_plot()
        self.dyn_plot.clear_plot()
        if not self._current:
            return

        variants = self._selected_variants()
        if not variants:
            return

        prob_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        dyn_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        ref_steps = None
        true_bin = None

        for vname in variants:
            vdata = self._current["variants"].get(vname)
            if not vdata:
                continue
            tr = vdata["trace"]
            steps = np.array(tr["step"], dtype=float)
            p_attack = np.array(tr["p_attack"], dtype=float)
            ref_steps = steps
            true_bin = np.array(tr["true_binary"], dtype=int)
            key = vname
            self._plot_line(self.prob_plot, steps, p_attack, key)
            prob_series[key] = (steps, p_attack)

            dyn = np.array(tr.get("dynamics_mse_step", []), dtype=float)
            if len(dyn) == len(steps):
                self._plot_line(self.dyn_plot, steps, dyn, key)
                dyn_series[key] = (steps, dyn)

        if "timesfm_ary_cls" in variants:
            tfm = self._current["variants"].get("timesfm_ary_cls", {}).get("timesfm", {})
            tfm_mse = tfm.get("per_step_mse", [])
            if tfm_mse and ref_steps is not None:
                # TimesFM MSE series may be shorter (starts later due to context)
                offset = len(ref_steps) - len(tfm_mse)
                tfm_steps = ref_steps[offset:] if offset >= 0 else ref_steps[: len(tfm_mse)]
                tfm_y = np.array(tfm_mse[-len(tfm_steps) :], dtype=float)
                self._plot_line(self.dyn_plot, tfm_steps, tfm_y, "timesfm_ary_cls")
                dyn_series["timesfm_dynamics"] = (tfm_steps, tfm_y)

        if self.show_truth.isChecked() and ref_steps is not None and true_bin is not None:
            self._plot_line(self.prob_plot, ref_steps, true_bin.astype(float), "actual")
            prob_series["actual"] = (ref_steps, true_bin.astype(float))

        if self.show_threshold.isChecked():
            if ref_steps is not None and len(ref_steps):
                lo, hi = float(ref_steps[0]), float(ref_steps[-1])
                self.prob_plot.addItem(
                    pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen("#ffffff55", width=1, style=QtCore.Qt.PenStyle.DashLine))
                )

        self.prob_plot.setTitle("P(attack) — causal one-step (real revealed context)")
        self.dyn_plot.setTitle("1-step dynamics MSE (ARY pred_state vs TimesFM forecast)")
        self.prob_plot.set_series(prob_series)
        self.dyn_plot.set_series(dyn_series)

        lines = []
        for vname in variants:
            m = self._current["variants"][vname]["metrics"]
            label = LAB_VARIANTS.get(vname, (vname, ""))[0]
            lines.append(
                f"<b>{label}</b>: F1={m['binary_f1']:.3f} prec={m['binary_precision']:.3f} "
                f"rec={m['binary_recall']:.3f} mitre={m['mitre_f1_macro']:.3f} dynMSE={m['dynamics_mse']:.1f}"
            )
        self.detail_label.setText("<br>".join(lines))
        self._reset_views()

    def _plot_line(self, plot, x, y, key, width=2.0):
        color = LAB_VARIANTS.get(key, (key, MODEL_COLORS.get(key, "#e6edf3")))[1]
        if key == "actual":
            color = MODEL_COLORS["actual"]
        label = LAB_VARIANTS.get(key, (SERIES_LABELS.get(key, key), ""))[0]
        if key == "actual":
            label = "Ground truth (attack)"
        pen = pg.mkPen(color, width=width)
        curve = plot.plot(x, y, pen=pen, name=label)
        curve.setDownsampling(auto=False)
        return curve


LIVE_LAB_RESULTS = ROOT / "results" / "ram_improve" / "live_lab"
ARY_COMPARE_RESULTS = ROOT / "results" / "lab_ary_comparison" / "eval"

ARY_COMPARE_SYSTEMS: dict[str, tuple[str, str]] = {
    "ary30_base": ("ARY.01 30s base", "#58a6ff"),
    "ary30_ramx": ("ARY.01 30s + RAMX", "#1f6feb"),
    "ary5_base": ("ARY 5s base", "#3fb950"),
    "ary5_ramx": ("ARY 5s + RAMX", "#238636"),
    "actual": ("Ground truth (attack)", "#f85149"),
}


class AryCompareViewer(QtWidgets.QMainWindow):
    """Interactive viewer for ARY 30s vs 5s base + RAMX lab comparisons."""

    def __init__(self, results_dir: Path):
        super().__init__()
        self.results_dir = results_dir
        self._scenarios: dict[str, dict] = {}
        self._current: dict | None = None

        self.setWindowTitle("ARY Lab Compare — 30s vs 5s base + RAMX")
        self.resize(1500, 920)
        pg.setConfigOptions(antialias=True, background="#161b22", foreground="#e6edf3")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Scenario"))
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.currentTextChanged.connect(self._on_scenario_change)
        left.addWidget(self.scenario_combo)

        self.metrics_table = QtWidgets.QTableWidget(0, 5)
        self.metrics_table.setHorizontalHeaderLabels(["System", "F1", "Prec", "Rec", "TTD"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.metrics_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.metrics_table.itemSelectionChanged.connect(self._refresh_plots)
        left.addWidget(self.metrics_table)

        self.meta_label = QtWidgets.QLabel("")
        self.meta_label.setWordWrap(True)
        left.addWidget(self.meta_label)

        left.addWidget(QtWidgets.QLabel("Systems shown"))
        self.system_checks: dict[str, QtWidgets.QCheckBox] = {}
        for sid, (label, color) in ARY_COMPARE_SYSTEMS.items():
            if sid == "actual":
                continue
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_plots)
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: 600; }}")
            self.system_checks[sid] = cb
            left.addWidget(cb)

        self.show_truth = QtWidgets.QCheckBox("Ground-truth attacks")
        self.show_truth.setChecked(True)
        self.show_truth.stateChanged.connect(self._refresh_plots)
        left.addWidget(self.show_truth)

        self.show_threshold = QtWidgets.QCheckBox("Decision threshold (0.5)")
        self.show_threshold.setChecked(True)
        self.show_threshold.stateChanged.connect(self._refresh_plots)
        left.addWidget(self.show_threshold)

        left_w = QtWidgets.QWidget()
        left_w.setLayout(left)
        left_w.setMinimumWidth(400)
        layout.addWidget(left_w)

        right = QtWidgets.QVBoxLayout()
        self.header = QtWidgets.QLabel("ARY lab comparison")
        self.header.setStyleSheet("font-size: 15px; font-weight: 700;")
        right.addWidget(self.header)

        self.readout = CrosshairReadout()
        right.addWidget(self.readout)

        self.truth_plot = DetailPlotWidget(self.readout, "Ground truth")
        self.truth_plot.setLabel("bottom", "Window step")
        self.truth_plot.setLabel("left", "Attack label")
        self.truth_plot.setYRange(0, 1, padding=0.05)
        right.addWidget(self.truth_plot, stretch=1)

        self.prob_plot_top = DetailPlotWidget(self.readout, "P(attack) 30s")
        self.prob_plot_top.setLabel("bottom", "Window step")
        self.prob_plot_top.setLabel("left", "P(attack)")
        self.prob_plot_top.setYRange(0, 1, padding=0.05)
        right.addWidget(self.prob_plot_top, stretch=2)

        self.prob_plot_bottom = DetailPlotWidget(self.readout, "P(attack) 5s")
        self.prob_plot_bottom.setLabel("bottom", "Window step")
        self.prob_plot_bottom.setLabel("left", "P(attack)")
        self.prob_plot_bottom.setYRange(0, 1, padding=0.05)
        right.addWidget(self.prob_plot_bottom, stretch=2)

        reset_btn = QtWidgets.QPushButton("Reset zoom")
        reset_btn.clicked.connect(self._reset_views)
        right.addWidget(reset_btn)

        self.detail_label = QtWidgets.QLabel("")
        self.detail_label.setWordWrap(True)
        right.addWidget(self.detail_label)

        right_w = QtWidgets.QWidget()
        right_w.setLayout(right)
        layout.addWidget(right_w, stretch=1)

        self._load_data()

    def _load_data(self) -> None:
        for path in sorted(self.results_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            key = payload.get("scenario_id") or path.stem
            self._scenarios[key] = payload
        self.scenario_combo.clear()
        for key in sorted(self._scenarios):
            p = self._scenarios[key]
            kind = p.get("kind", "?")
            title = p.get("title", key)
            self.scenario_combo.addItem(f"[{kind}] {title}", key)
        if not self._scenarios:
            self.meta_label.setText(
                f"No eval JSON under {self.results_dir}.\n"
                f"Run: python scripts/plot_lab_ary_comparison.py"
            )

    def _on_scenario_change(self, _text: str) -> None:
        key = self.scenario_combo.currentData()
        if not key or key not in self._scenarios:
            return
        self._current = self._scenarios[key]
        p = self._current
        cls = p.get("class_id", "?")
        self.header.setText(p.get("title", key))
        if p.get("kind") == "live_round":
            self.meta_label.setText(
                f"Live zero-day round · class={cls} · "
                f"{p.get('window_sec', '?')}s windows · attack starts @ step {p.get('attack_start', '?')}"
            )
        else:
            self.meta_label.setText(
                f"Lab PCAP · class={cls} · 30s + 5s ingest · "
                f"warm-up then attack PCAP windows"
            )
        self._populate_metrics_table()
        self._refresh_plots()

    def _populate_metrics_table(self) -> None:
        self.metrics_table.setRowCount(0)
        if not self._current:
            return
        rows: list[tuple[str, dict]] = []
        if self._current.get("kind") == "live_round":
            for sid, vdata in self._current.get("variants", {}).items():
                rows.append((sid, vdata.get("metrics", {})))
        else:
            for panel_key in ("30s", "5s"):
                panel = self._current.get("panels", {}).get(panel_key, {})
                for sid, vdata in panel.get("variants", {}).items():
                    rows.append((sid, vdata.get("metrics", {})))
        self.metrics_table.setRowCount(len(rows))
        fg = QtGui.QColor("#e6edf3")
        for r, (sid, m) in enumerate(rows):
            label, color = ARY_COMPARE_SYSTEMS.get(sid, (sid, "#ccc"))
            ttd = m.get("ttd")
            vals = [label, f"{m.get('f1', 0):.3f}", f"{m.get('precision', 0):.3f}",
                    f"{m.get('recall', 0):.3f}", str(ttd) if ttd is not None else "—"]
            for c, text in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(text)
                item.setForeground(QtGui.QColor(color) if c == 0 else fg)
                if c == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, sid)
                self.metrics_table.setItem(r, c, item)
        self.metrics_table.selectAll()

    def _selected_systems(self) -> list[str]:
        return [sid for sid, cb in self.system_checks.items() if cb.isChecked()]

    def _reset_views(self) -> None:
        self.truth_plot.autoRange()
        self.prob_plot_top.autoRange()
        self.prob_plot_bottom.autoRange()

    def _refresh_plots(self) -> None:
        try:
            self._refresh_plots_inner()
        except Exception as exc:
            self.detail_label.setText(f"<span style='color:#f85149'>{exc}</span>")
            import traceback
            traceback.print_exc()

    def _shade_attack(self, plot: DetailPlotWidget, attack_start: int, n: int) -> None:
        region = pg.LinearRegionItem(
            values=[attack_start, max(attack_start, n - 1)],
            movable=False,
            brush=pg.mkBrush("#f8514928"),
            pen=pg.mkPen(None),
        )
        plot.addItem(region)

    def _plot_panel(
        self,
        plot: DetailPlotWidget,
        variants: dict,
        systems: list[str],
        title: str,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        plot.clear_plot()
        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        ref_steps = None
        true_bin = None
        attack_start = 0
        for sid in systems:
            vdata = variants.get(sid)
            if not vdata:
                continue
            tr = vdata["trace"]
            steps = np.array(tr["step"], dtype=float)
            p_attack = np.array(tr["p_attack"], dtype=float)
            ref_steps = steps
            true_bin = np.array(tr["true_binary"], dtype=int)
            attack_start = int(vdata.get("attack_start", 0))
            self._plot_line(plot, steps, p_attack, sid)
            series[sid] = (steps, p_attack)
        if ref_steps is not None and len(ref_steps):
            self._shade_attack(plot, attack_start, len(ref_steps))
        if self.show_threshold.isChecked() and ref_steps is not None and len(ref_steps):
            plot.addItem(
                pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen("#ffffff55", width=1, style=QtCore.Qt.PenStyle.DashLine))
            )
        plot.setTitle(title)
        plot.set_series(series)
        if true_bin is not None:
            series["_true_bin"] = (ref_steps, true_bin.astype(float))
        return series

    def _refresh_plots_inner(self) -> None:
        self.truth_plot.clear_plot()
        self.prob_plot_top.clear_plot()
        self.prob_plot_bottom.clear_plot()
        if not self._current:
            return

        systems = self._selected_systems()
        if not systems:
            return

        kind = self._current.get("kind")
        truth_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        if kind == "live_round":
            variants = self._current.get("variants", {})
            attack_start = int(self._current.get("attack_start", 0))
            any_v = next(iter(variants.values()), {})
            tr = any_v.get("trace", {})
            steps = np.array(tr.get("step", []), dtype=float)
            true_bin = np.array(tr.get("true_binary", []), dtype=int)
            if len(steps) and self.show_truth.isChecked():
                self._plot_line(self.truth_plot, steps, true_bin.astype(float), "actual")
                self._shade_attack(self.truth_plot, attack_start, len(steps))
                truth_series["actual"] = (steps, true_bin.astype(float))
            self.truth_plot.setTitle("Ground truth — live round (15s windows)")
            self.truth_plot.set_series(truth_series)

            live_systems = [s for s in systems if s in variants]
            top_series = self._plot_panel(
                self.prob_plot_top,
                variants,
                live_systems,
                "P(attack) — all variants on live round trace (15s windows)",
            )
            self.prob_plot_bottom.hide()
            self.prob_plot_top.show()
            self.prob_plot_top.set_series(top_series)
        else:
            panels = self._current.get("panels", {})
            p30 = panels.get("30s", {})
            p5 = panels.get("5s", {})
            v30 = p30.get("variants", {})
            v5 = p5.get("variants", {})
            start30 = int(p30.get("attack_start", 0))
            start5 = int(p5.get("attack_start", 0))

            tr30 = next(iter(v30.values()), {}).get("trace", {})
            steps30 = np.array(tr30.get("step", []), dtype=float)
            true30 = np.array(tr30.get("true_binary", []), dtype=int)
            if len(steps30) and self.show_truth.isChecked():
                self._plot_line(self.truth_plot, steps30, true30.astype(float), "actual")
                self._shade_attack(self.truth_plot, start30, len(steps30))
                truth_series["actual"] = (steps30, true30.astype(float))
            self.truth_plot.setTitle("Ground truth — 30s windows (CIC warm-up → attack PCAP)")
            self.truth_plot.set_series(truth_series)

            s30 = [s for s in systems if s.startswith("ary30")]
            s5 = [s for s in systems if s.startswith("ary5")]
            top_series = self._plot_panel(
                self.prob_plot_top, v30, s30,
                "P(attack) — ARY.01 30s base + RAMX",
            )
            bot_series = self._plot_panel(
                self.prob_plot_bottom, v5, s5,
                "P(attack) — ARY 5s base + RAMX",
            )
            self.prob_plot_bottom.show()
            self.prob_plot_top.set_series(top_series)
            self.prob_plot_bottom.set_series(bot_series)

        lines = []
        if kind == "live_round":
            for sid in systems:
                m = self._current["variants"].get(sid, {}).get("metrics", {})
                if not m:
                    continue
                label = ARY_COMPARE_SYSTEMS.get(sid, (sid, ""))[0]
                lines.append(
                    f"<b>{label}</b>: F1={m.get('f1', 0):.3f} "
                    f"maxP={m.get('max_p_attack', 0):.3f} TTD={m.get('ttd', '—')} "
                    f"det={m.get('detected', False)} harm={m.get('harm', 0)}"
                )
        else:
            for sid, m in self._current.get("scores", {}).items():
                label = ARY_COMPARE_SYSTEMS.get(sid, (sid, ""))[0]
                lines.append(
                    f"<b>{label}</b>: F1={m.get('f1', 0):.3f} "
                    f"maxP={m.get('max_p_attack', 0):.3f} TTD={m.get('ttd', '—')} "
                    f"det={m.get('detected', False)}"
                )
        self.detail_label.setText("<br>".join(lines) if lines else "")
        self._reset_views()

    def _plot_line(self, plot, x, y, key, width=2.0):
        color = ARY_COMPARE_SYSTEMS.get(key, (key, MODEL_COLORS.get(key, "#e6edf3")))[1]
        if key == "actual":
            color = ARY_COMPARE_SYSTEMS["actual"][1]
        label = ARY_COMPARE_SYSTEMS.get(key, (key, key))[0]
        pen = pg.mkPen(color, width=width)
        curve = plot.plot(x, y, pen=pen, name=label)
        curve.setDownsampling(auto=False)
        return curve


LIVE_LAB_SYSTEMS: dict[str, tuple[str, str]] = {
    "ary_base": ("ARY base", "#9aa0a6"),
    "ary_v8": ("ARY V8 RAM", "#27ae60"),
    "ary_v10": ("ARY V10 RAM", "#e67e22"),
    "timesfm_ary_cls": ("TimesFM+ARY cls", "#3498db"),
    "fmary_ram_ungated": ("TimesFM+ARY RAM (ungated)", "#e74c3c"),
    "fmary_gated_ram": ("TimesFM+ARY RAM (gated)", "#c0392b"),
}

PHASE_COLORS = {
    "warmup": "#30363d",
    "attack": "#f85149",
    "recon": "#d29922",
    "exploit": "#f85149",
    "post_exploit": "#a371f7",
    "cooldown": "#1f6feb",
}


def _fine_phase_segments(trace: list[dict], events: list[dict]) -> list[tuple[str, int, int]]:
    """Refine each window's coarse phase ("warmup"/"attack"/"cooldown") into
    recon/exploit/post_exploit sub-stages where the objective bot's own
    structured event log (which has real timestamps for those 3 stages)
    covers a window's capture time. Windows during attack before any staged
    event has fired yet (e.g. raw recon-bot noise, which isn't logged with
    structured events) stay labeled "attack" generically."""
    ev_sorted = sorted(events, key=lambda e: e["ts"])
    segments: list[tuple[str, int, int]] = []
    cur_label, seg_start = None, 0
    for i, w in enumerate(trace):
        if w["phase"] in ("warmup", "cooldown"):
            label = w["phase"]
        else:
            label = "attack"
            for e in ev_sorted:
                if e["ts"] <= w["t_end"]:
                    label = e["stage"]
                else:
                    break
        if label != cur_label:
            if cur_label is not None:
                segments.append((cur_label, seg_start, i))
            cur_label, seg_start = label, i
    if cur_label is not None:
        segments.append((cur_label, seg_start, len(trace)))
    return segments


class LiveLabViewer(QtWidgets.QMainWindow):
    """Interactive viewer for the live adversarial attack lab (plan section 6):
    objective x evasion-version picker, a shaded-stage P(attack) timeline for
    all 6 systems with first-detection markers, a forecast panel (pick a
    window, overlay its stored N-step-ahead forecast per system against what
    actually happened next), and an aggregated leaderboard."""

    def __init__(self, results_dir: Path):
        super().__init__()
        self.results_dir = results_dir
        self._rounds: dict[tuple[str, str], dict] = {}
        self._current: dict | None = None

        self.setWindowTitle("Live Adversarial Attack Lab")
        self.resize(1500, 940)
        pg.setConfigOptions(antialias=True, background="#161b22", foreground="#e6edf3")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # --- left: round picker + leaderboard ---
        left = QtWidgets.QVBoxLayout()

        picker_row = QtWidgets.QHBoxLayout()
        self.objective_combo = QtWidgets.QComboBox()
        self.objective_combo.currentTextChanged.connect(self._on_objective_change)
        self.evasion_combo = QtWidgets.QComboBox()
        self.evasion_combo.currentTextChanged.connect(self._on_round_change)
        picker_row.addWidget(QtWidgets.QLabel("Objective"))
        picker_row.addWidget(self.objective_combo)
        picker_row.addWidget(QtWidgets.QLabel("Evasion"))
        picker_row.addWidget(self.evasion_combo)
        left.addLayout(picker_row)

        self.round_label = QtWidgets.QLabel("")
        self.round_label.setWordWrap(True)
        left.addWidget(self.round_label)

        left.addWidget(QtWidgets.QLabel("Leaderboard (aggregated across all rounds)"))
        self.leaderboard_table = QtWidgets.QTableWidget(0, 7)
        self.leaderboard_table.setHorizontalHeaderLabels(
            ["System", "Wins", "Losses", "Det.rate", "Mean F1", "Mean TTD(s)", "Benign harm"]
        )
        self.leaderboard_table.horizontalHeader().setStretchLastSection(True)
        self.leaderboard_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        left.addWidget(self.leaderboard_table)

        self.winner_label = QtWidgets.QLabel("")
        self.winner_label.setWordWrap(True)
        self.winner_label.setStyleSheet("font-size: 13px; font-weight: 700; color: #3fb950;")
        left.addWidget(self.winner_label)

        left.addWidget(QtWidgets.QLabel("Systems shown"))
        self.system_checks: dict[str, QtWidgets.QCheckBox] = {}
        for sid, (label, color) in LIVE_LAB_SYSTEMS.items():
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_timeline)
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: 600; }}")
            self.system_checks[sid] = cb
            left.addWidget(cb)

        left_w = QtWidgets.QWidget()
        left_w.setLayout(left)
        left_w.setMinimumWidth(420)
        layout.addWidget(left_w)

        # --- right: timeline + forecast panels ---
        right = QtWidgets.QVBoxLayout()

        self.header = QtWidgets.QLabel("Select an objective/evasion round")
        self.header.setStyleSheet("font-size: 15px; font-weight: 700;")
        right.addWidget(self.header)

        self.readout = CrosshairReadout()
        right.addWidget(self.readout)

        self.timeline_plot = DetailPlotWidget(self.readout, "P(attack)")
        self.timeline_plot.setLabel("bottom", "Window index")
        self.timeline_plot.setLabel("left", "P(attack)")
        self.timeline_plot.setYRange(0, 1, padding=0.05)
        right.addWidget(self.timeline_plot, stretch=2)

        fc_ctrl = QtWidgets.QHBoxLayout()
        fc_ctrl.addWidget(QtWidgets.QLabel("Forecast from window:"))
        self.forecast_window_spin = QtWidgets.QSpinBox()
        self.forecast_window_spin.valueChanged.connect(self._refresh_forecast)
        fc_ctrl.addWidget(self.forecast_window_spin)
        fc_ctrl.addWidget(QtWidgets.QLabel("Feature:"))
        self.forecast_feature_combo = QtWidgets.QComboBox()
        self.forecast_feature_combo.currentIndexChanged.connect(self._refresh_forecast)
        fc_ctrl.addWidget(self.forecast_feature_combo)
        reset_btn = QtWidgets.QPushButton("Reset zoom")
        reset_btn.clicked.connect(self._reset_views)
        fc_ctrl.addWidget(reset_btn)
        fc_ctrl.addStretch()
        right.addLayout(fc_ctrl)

        self.forecast_plot = DetailPlotWidget(self.readout, "forecast")
        self.forecast_plot.setLabel("bottom", "Window index")
        self.forecast_plot.setLabel("left", "Feature value")
        right.addWidget(self.forecast_plot, stretch=2)

        self.detail_label = QtWidgets.QLabel("")
        self.detail_label.setWordWrap(True)
        right.addWidget(self.detail_label)

        right_w = QtWidgets.QWidget()
        right_w.setLayout(right)
        layout.addWidget(right_w, stretch=1)

        self._load_data()

    # -- data loading --------------------------------------------------
    def _load_data(self) -> None:
        for path in sorted(self.results_dir.glob("*/*/round.json")):
            payload = json.loads(path.read_text())
            self._rounds[(payload["objective"], payload["evasion"])] = payload

        objectives = sorted({k[0] for k in self._rounds})
        self.objective_combo.clear()
        self.objective_combo.addItems(objectives)

        self._load_leaderboard()

        if not self._rounds:
            self.round_label.setText(
                f"No round.json under {self.results_dir}.\nRun: python scripts/live_attack_lab.py"
            )

    def _load_leaderboard(self) -> None:
        leaderboard_path = self.results_dir / "leaderboard.json"
        payload = None
        if leaderboard_path.exists():
            try:
                payload = json.loads(leaderboard_path.read_text())
            except Exception:
                payload = None
        if payload is None:
            round_files = sorted(self.results_dir.glob("*/*/round.json"))
            if round_files:
                from src.aryan.lab_scoring import aggregate_leaderboard

                payload = aggregate_leaderboard(round_files)
        if payload is None:
            return

        rows = payload.get("leaderboard", [])
        self.leaderboard_table.setRowCount(len(rows))
        fg = QtGui.QColor("#e6edf3")
        for r, row in enumerate(rows):
            sid = row["system"]
            label, color = LIVE_LAB_SYSTEMS.get(sid, (sid, "#ccc"))
            ttd = f"{row['mean_ttd_sec']:.1f}" if row.get("mean_ttd_sec") is not None else "—"
            vals = [
                label, str(row["wins"]), str(row["losses"]),
                f"{row['detection_rate']:.0%}", f"{row['mean_f1']:.3f}", ttd, str(row["total_benign_harm"]),
            ]
            for c, text in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(text)
                item.setForeground(QtGui.QColor(color) if c == 0 else fg)
                self.leaderboard_table.setItem(r, c, item)
        if rows:
            winner = LIVE_LAB_SYSTEMS.get(rows[0]["system"], (rows[0]["system"], ""))[0]
            self.winner_label.setText(
                f"Round winner (most wins, fewest losses, best mean F1): {winner}"
            )
        else:
            self.winner_label.setText("")

    def _on_objective_change(self, objective: str) -> None:
        versions = sorted(v for (o, v) in self._rounds if o == objective)
        self.evasion_combo.blockSignals(True)
        self.evasion_combo.clear()
        self.evasion_combo.addItems(versions)
        self.evasion_combo.blockSignals(False)
        self._on_round_change(versions[0] if versions else "")

    def _on_round_change(self, evasion: str) -> None:
        objective = self.objective_combo.currentText()
        key = (objective, evasion)
        if key not in self._rounds:
            self._current = None
            return
        self._current = self._rounds[key]
        p = self._current
        self.header.setText(
            f"{p['objective']} / {p['evasion']} · {p['mitre_tactic']} · {p['n_windows']} windows · "
            f"round {p.get('round_id', '?')}"
        )
        bot_ok = p.get("bot_result", {}).get("returncode") == 0
        n_events = len(p.get("events", []))
        self.round_label.setText(
            f"Objective bot exit code: {p.get('bot_result', {}).get('returncode')} "
            f"({'ok' if bot_ok else 'ERROR'}) · {n_events} logged stage events · "
            f"window={p.get('window_sec', 0):.1f}s (speed={p.get('speed', 1)}x)"
        )

        n_windows = p["n_windows"]
        self.forecast_window_spin.setRange(0, max(0, n_windows - 1))
        self.forecast_feature_combo.blockSignals(True)
        self.forecast_feature_combo.clear()
        for fidx in p.get("forecast_features", []):
            name = FEATURE_COLS_242[fidx] if 0 <= fidx < len(FEATURE_COLS_242) else f"dim_{fidx}"
            self.forecast_feature_combo.addItem(f"[{fidx}] {name}", userData=fidx)
        self.forecast_feature_combo.blockSignals(False)

        self._refresh_timeline()
        self._refresh_forecast()

    # -- timeline panel --------------------------------------------------
    def _reset_views(self) -> None:
        self.timeline_plot.autoRange()
        self.timeline_plot.setYRange(0, 1, padding=0.05)
        self.forecast_plot.autoRange()

    def _refresh_timeline(self) -> None:
        try:
            self._refresh_timeline_inner()
        except Exception as exc:
            self.detail_label.setText(f"<span style='color:#f85149'>Timeline error: {exc}</span>")
            import traceback
            traceback.print_exc()

    def _refresh_timeline_inner(self) -> None:
        self.timeline_plot.clear_plot()
        if not self._current:
            return
        trace = self._current["trace"]
        events = self._current.get("events", [])
        scores = self._current.get("scores", {})
        n = len(trace)
        x = np.arange(n, dtype=float)

        for label, a, b in _fine_phase_segments(trace, events):
            region = pg.LinearRegionItem(
                values=(a - 0.5, b - 0.5), movable=False, brush=pg.mkBrush(PHASE_COLORS.get(label, "#888888") + "40"),
            )
            region.setZValue(-10)
            for line in region.lines:
                line.setPen(pg.mkPen(None))
            self.timeline_plot.addItem(region, ignoreBounds=True)

        attack_idxs = [i for i, w in enumerate(trace) if w["true_bin"] == 1]
        attack_start_idx = attack_idxs[0] if attack_idxs else None

        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for sid, (label, color) in LIVE_LAB_SYSTEMS.items():
            if not self.system_checks[sid].isChecked():
                continue
            y = np.array([w["systems"].get(sid, {}).get("p_att", 0.0) for w in trace], dtype=float)
            curve = self.timeline_plot.plot(x, y, pen=pg.mkPen(color, width=2.0), name=label)
            curve.setDownsampling(auto=False)
            curve.setClipToView(True)
            series[sid] = (x, y)

            ttd_windows = scores.get(sid, {}).get("ttd_windows")
            if ttd_windows is not None and attack_start_idx is not None:
                det_x = attack_start_idx + ttd_windows
                self.timeline_plot.addItem(
                    pg.InfiniteLine(pos=det_x, angle=90, pen=pg.mkPen(color, style=QtCore.Qt.PenStyle.DotLine, width=1.5))
                )

        true_bin = np.array([w["true_bin"] for w in trace], dtype=float)
        self.timeline_plot.plot(
            x, true_bin, pen=pg.mkPen("#f0f6fc", width=1.0, style=QtCore.Qt.PenStyle.DashLine), name="ground truth (attack)"
        )
        series["actual"] = (x, true_bin)

        self.timeline_plot.addItem(
            pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen("#ffffff55", width=1, style=QtCore.Qt.PenStyle.DashLine))
        )
        self.timeline_plot.setTitle(
            "P(attack) per system — shaded: warmup(gray) recon(amber) exploit/attack(red) post_exploit(purple) cooldown(blue)"
        )
        self.timeline_plot.set_series(series)
        self.timeline_plot.enableAutoRange()
        self.timeline_plot.setYRange(0, 1, padding=0.05)

        lines = []
        for sid in LIVE_LAB_SYSTEMS:
            sc = scores.get(sid)
            if not sc:
                continue
            label = LIVE_LAB_SYSTEMS[sid][0]
            ttd = sc["ttd_windows"] if sc["ttd_windows"] is not None else "—"
            lost = sc["lost_to_attacker"]
            lost_txt = "LOST" if lost is True else ("held" if lost is False else "—")
            lines.append(
                f"<b>{label}</b>: F1={sc['binary_f1']:.3f} TTD={ttd}w "
                f"{lost_txt} benign_harm={sc['benign_harm_count']}"
            )
        self.detail_label.setText("<br>".join(lines))

    # -- forecast panel ---------------------------------------------------
    def _refresh_forecast(self) -> None:
        try:
            self._refresh_forecast_inner()
        except Exception as exc:
            self.forecast_plot.setTitle(f"Forecast error: {exc}")

    def _refresh_forecast_inner(self) -> None:
        self.forecast_plot.clear_plot()
        if not self._current:
            return
        trace = self._current["trace"]
        w_idx = self.forecast_window_spin.value()
        fidx = self.forecast_feature_combo.currentData()
        if fidx is None or w_idx >= len(trace):
            return

        fname = FEATURE_COLS_242[fidx] if 0 <= fidx < len(FEATURE_COLS_242) else f"dim_{fidx}"
        window = trace[w_idx]
        horizon = self._current.get("forecast_horizon", 100)

        actual_x = np.arange(w_idx, len(trace), dtype=float)
        actual_y = np.array([trace[j]["state"][fidx] for j in range(w_idx, len(trace))], dtype=float)
        self.forecast_plot.plot(
            actual_x, actual_y, pen=pg.mkPen("#f0f6fc", width=3.0), name="actual"
        )
        series = {"actual": (actual_x, actual_y)}

        any_forecast = False
        for sid, (label, color) in LIVE_LAB_SYSTEMS.items():
            fc = window["systems"].get(sid, {}).get("forecast", {})
            vals = fc.get(str(fidx))
            if vals is None:
                continue
            any_forecast = True
            fc_x = np.arange(w_idx + 1, w_idx + 1 + len(vals), dtype=float)
            curve = self.forecast_plot.plot(fc_x, np.array(vals, dtype=float), pen=pg.mkPen(color, width=2.0), name=label)
            curve.setDownsampling(auto=False)
            series[sid] = (fc_x, np.array(vals, dtype=float))

        self.forecast_plot.addItem(
            pg.InfiniteLine(pos=w_idx, angle=90, pen=pg.mkPen("#ffffff", style=QtCore.Qt.PenStyle.DashLine, width=1))
        )
        title = f"Forecast from window {w_idx} — {fname} (horizon={horizon})"
        if not any_forecast:
            title += "  [no forecast stored for this window — pick a multiple of --forecast-every]"
        self.forecast_plot.setTitle(title)
        self.forecast_plot.set_series(series)
        self.forecast_plot.enableAutoRange()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["falloff", "lab", "live_lab", "ary_compare"], default="falloff")
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    default_dirs = {
        "lab": LAB_RESULTS,
        "live_lab": LIVE_LAB_RESULTS,
        "ary_compare": ARY_COMPARE_RESULTS,
    }
    if args.results is None:
        args.results = default_dirs.get(args.mode, DEFAULT_RESULTS)

    if not args.results.exists():
        if args.mode == "lab":
            print(f"No results at {args.results}. Run: python scripts/timesfm_ary_lab_eval.py")
        elif args.mode == "live_lab":
            print(f"No results at {args.results}. Run: python scripts/live_attack_lab.py")
        elif args.mode == "ary_compare":
            print(f"No results at {args.results}. Run: python scripts/plot_lab_ary_comparison.py")
        else:
            print(f"No results at {args.results}. Run: python scripts/forecast_falloff_eval.py --all-features")
        raise SystemExit(1)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#0d1117"))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#e6edf3"))
    app.setPalette(palette)
    if args.mode == "lab":
        w = LabAttackViewer(args.results)
    elif args.mode == "live_lab":
        w = LiveLabViewer(args.results)
    elif args.mode == "ary_compare":
        w = AryCompareViewer(args.results)
    else:
        w = FalloffViewer(args.results)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
