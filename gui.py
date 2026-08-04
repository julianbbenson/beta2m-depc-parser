#!/usr/bin/env python3
"""
β2m DEPC Parser — desktop GUI.

Run from the repository root:
    python gui.py
"""

import logging
import subprocess
import sys
import threading
from pathlib import Path

# Ensure src/ is importable when launched from the project root.
sys.path.insert(0, str(Path(__file__).parent))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Logging → Tkinter bridge
# ---------------------------------------------------------------------------

class _TextHandler(logging.Handler):
    """Forwards log records to a Tkinter Text widget (thread-safe)."""

    def __init__(self, widget: tk.Text) -> None:
        super().__init__()
        self._widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record) + "\n"
        tag = ("error" if record.levelno >= logging.ERROR else
               "warn"  if record.levelno >= logging.WARNING else "info")
        self._widget.after(0, self._append, msg, tag)

    def _append(self, msg: str, tag: str) -> None:
        self._widget.configure(state="normal")
        self._widget.insert("end", msg, tag)
        self._widget.see("end")
        self._widget.configure(state="disabled")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("β2m DEPC Parser")
        self.minsize(820, 680)
        self._build_controls()
        self._build_log()
        self._build_results()
        self._attach_logger()

    # ------------------------------------------------------------------ UI --

    def _build_controls(self) -> None:
        frm = ttk.LabelFrame(self, text="Pipeline settings", padding=12)
        frm.pack(fill="x", padx=14, pady=(14, 4))
        frm.columnconfigure(1, weight=1)

        # Input folder
        ttk.Label(frm, text="Input folder (.mzML files)").grid(
            row=0, column=0, sticky="w", pady=5)
        self._input_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self._input_var).grid(
            row=0, column=1, sticky="ew", padx=8)
        ttk.Button(frm, text="Browse…",
                   command=lambda: self._pick_dir(self._input_var)).grid(
            row=0, column=2)

        # Output folder
        ttk.Label(frm, text="Output folder").grid(
            row=1, column=0, sticky="w", pady=5)
        self._output_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self._output_var).grid(
            row=1, column=1, sticky="ew", padx=8)
        ttk.Button(frm, text="Browse…",
                   command=lambda: self._pick_dir(self._output_var)).grid(
            row=1, column=2)

        # Resolution mode
        ttk.Label(frm, text="Resolution mode").grid(
            row=2, column=0, sticky="w", pady=5)
        self._mode_var = tk.StringVar(value="auto")
        mode_frm = ttk.Frame(frm)
        mode_frm.grid(row=2, column=1, sticky="w", padx=8)
        for label, val in [("Auto-detect", "auto"),
                            ("High-res (Orbitrap, ±0.02 Da)", "high_res"),
                            ("Low-res (ion trap, ±0.5 Da)", "low_res")]:
            ttk.Radiobutton(mode_frm, text=label,
                            variable=self._mode_var, value=val).pack(
                side="left", padx=(0, 14))

        # File-naming hint
        ttk.Label(frm,
                  text="Name no-Cu files with 'no_cu' and +Cu files with 'cu' in the filename.",
                  foreground="gray").grid(row=3, column=0, columnspan=3,
                                          sticky="w", pady=(2, 8))

        # Run / progress
        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=4, column=0, columnspan=3, pady=(4, 2))

        self._run_btn = ttk.Button(btn_frm, text="▶  Run Pipeline",
                                   command=self._run)
        self._run_btn.pack(side="left", padx=8)

        self._progress = ttk.Progressbar(btn_frm, mode="indeterminate", length=200)
        self._progress.pack(side="left", padx=8)

        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(btn_frm, textvariable=self._status_var,
                  foreground="gray").pack(side="left", padx=4)

    def _build_log(self) -> None:
        frm = ttk.LabelFrame(self, text="Log", padding=4)
        frm.pack(fill="both", expand=False, padx=14, pady=4)

        self._log_text = tk.Text(
            frm, height=7, state="disabled",
            font=("Courier", 9), wrap="word",
            background="#1e1e1e", foreground="#cccccc",
            selectbackground="#444", insertbackground="white")
        sb = ttk.Scrollbar(frm, command=self._log_text.yview)
        self._log_text["yscrollcommand"] = sb.set

        self._log_text.tag_configure("error", foreground="#ff6b6b")
        self._log_text.tag_configure("warn",  foreground="#ffd93d")
        self._log_text.tag_configure("info",  foreground="#aaddaa")

        self._log_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _build_results(self) -> None:
        frm = ttk.LabelFrame(self, text="Results", padding=4)
        frm.pack(fill="both", expand=True, padx=14, pady=(4, 14))

        # Export / open-folder buttons
        btn_row = ttk.Frame(frm)
        btn_row.pack(fill="x", pady=(0, 4))
        self._export_btn = ttk.Button(btn_row, text="💾  Save plots…",
                                      command=self._save_plots, state="disabled")
        self._export_btn.pack(side="left", padx=(0, 8))
        self._folder_btn = ttk.Button(btn_row, text="📂  Open output folder",
                                      command=self._open_output_folder, state="disabled")
        self._folder_btn.pack(side="left")

        self._fig, (self._ax_le, self._ax_ps) = plt.subplots(
            2, 1, figsize=(9, 4.5), dpi=100,
            gridspec_kw={"hspace": 0.55})
        self._fig.patch.set_facecolor("#f8f8f8")
        for ax in (self._ax_le, self._ax_ps):
            ax.set_facecolor("#fafafa")
            ax.tick_params(labelsize=7)
        self._ax_le.set_title("Per-residue labeling extents", fontsize=9)
        self._ax_ps.set_title("Cu(II) protection scores", fontsize=9)
        self._ax_ps.axhline(0.2, color="black", linestyle="--",
                            linewidth=0.7, label="threshold (0.20)")
        self._ax_ps.legend(fontsize=7)

        self._canvas = FigureCanvasTkAgg(self._fig, master=frm)
        self._toolbar = NavigationToolbar2Tk(self._canvas, frm, pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(fill="x")
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas.draw()

    def _attach_logger(self) -> None:
        handler = _TextHandler(self._log_text)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

    # --------------------------------------------------------- interactions --

    @staticmethod
    def _pick_dir(var: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _run(self) -> None:
        input_dir  = self._input_var.get().strip()
        output_dir = self._output_var.get().strip()

        if not input_dir:
            messagebox.showerror("Missing input", "Please select an input folder.")
            return
        if not output_dir:
            messagebox.showerror("Missing output", "Please select an output folder.")
            return
        if not Path(input_dir).is_dir():
            messagebox.showerror("Not found",
                                 f"Input folder does not exist:\n{input_dir}")
            return

        self._run_btn.configure(state="disabled")
        self._progress.start(12)
        self._status_var.set("Running…")
        self._reset_plots()

        threading.Thread(
            target=self._worker,
            args=(input_dir, output_dir, self._mode_var.get()),
            daemon=True,
        ).start()

    def _worker(self, input_dir: str, output_dir: str, mode: str) -> None:
        try:
            from src.pipeline import _run_pipeline
            _run_pipeline(
                input_dir=input_dir,
                output_dir=output_dir,
                mode=mode,
                no_cu_pattern="no_cu",
                cu_pattern="cu",
                workers=4,
            )
            self.after(0, self._on_done, output_dir)
        except Exception as exc:
            logging.getLogger("gui").error("Pipeline error: %s", exc, exc_info=True)
            self.after(0, self._on_error)

    def _on_done(self, output_dir: str) -> None:
        self._progress.stop()
        self._run_btn.configure(state="normal")
        self._status_var.set("Done.")
        self._last_output_dir = output_dir
        self._export_btn.configure(state="normal")
        self._folder_btn.configure(state="normal")
        self._plot_results(output_dir)

    def _on_error(self) -> None:
        self._progress.stop()
        self._run_btn.configure(state="normal")
        self._status_var.set("Error — see log.")

    # ---------------------------------------------------------------- plots --

    def _reset_plots(self) -> None:
        for ax in (self._ax_le, self._ax_ps):
            ax.cla()
        self._canvas.draw()

    def _plot_results(self, output_dir: str) -> None:
        out = Path(output_dir)
        colors = ["steelblue", "firebrick", "seagreen", "goldenrod"]

        # Labeling extents
        le_path = out / "labeling_extents.csv"
        ax = self._ax_le
        ax.cla()
        ax.set_facecolor("#fafafa")
        if le_path.exists():
            df = pd.read_csv(le_path, index_col=0)
            for i, col in enumerate(df.columns):
                ax.plot(df.index, df[col], "o-", ms=2.5, linewidth=0.9,
                        color=colors[i % len(colors)], label=col)
            ax.set_xlim(1, 114)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("β2m position", fontsize=8)
            ax.set_ylabel("Labeling extent", fontsize=8)
            ax.legend(fontsize=7)
        ax.set_title("Per-residue DEPC labeling extents", fontsize=9)
        ax.tick_params(labelsize=7)

        # Protection scores
        cp_path = out / "cu_protection.csv"
        ax = self._ax_ps
        ax.cla()
        ax.set_facecolor("#fafafa")
        if cp_path.exists():
            prot = pd.read_csv(cp_path, index_col=0)
            if "protection_score" in prot.columns:
                valid = prot["protection_score"].dropna()
                bar_colors = [
                    "firebrick" if prot.loc[i, "is_protected"] else "steelblue"
                    for i in valid.index
                ]
                ax.bar(range(len(valid)), valid.values,
                       color=bar_colors, width=0.7)
                ax.set_xticks(range(len(valid)))
                ax.set_xticklabels(
                    [str(int(i)) for i in valid.index],
                    rotation=90, fontsize=5)
                ax.axhline(0.2, color="black", linestyle="--",
                           linewidth=0.7, label="threshold (0.20)")
                ax.legend(fontsize=7)
        ax.set_title("Cu(II) protection scores  (red = protected)", fontsize=9)
        ax.set_ylabel("Protection score", fontsize=8)
        ax.tick_params(labelsize=7)

        self._fig.tight_layout()
        self._canvas.draw()

    def _save_plots(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("PNG", "*.png"), ("SVG", "*.svg")],
            initialfile="b2m_depc_results",
        )
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
            self._status_var.set(f"Saved → {Path(path).name}")

    def _open_output_folder(self) -> None:
        folder = getattr(self, "_last_output_dir", self._output_var.get().strip())
        if folder and Path(folder).is_dir():
            opener = ("open" if sys.platform == "darwin"
                      else "explorer" if sys.platform == "win32" else "xdg-open")
            subprocess.Popen([opener, folder])


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    App().mainloop()
