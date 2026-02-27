"""
╔══════════════════════════════════════════════════════════════════════╗
║      K-MEANS CLUSTERING — Interactive Educational Tool              ║
║      Unsupervised Machine Learning Visualizer                       ║
║      Built with Python + Tkinter (zero extra install)               ║
╠══════════════════════════════════════════════════════════════════════╣
║  HOW TO RUN:                                                         ║
║    python kmeans_edu.py                                              ║
╚══════════════════════════════════════════════════════════════════════╝

CONTROLS:
  Left-Click  on canvas  →  Add data point
  Right-Click on canvas  →  Remove nearest point
  SPACE                  →  Next algorithm step
  R                      →  Reset
  S                      →  Load sample dataset
"""

import tkinter as tk
from tkinter import font as tkfont
import math, random, time

# ─── Layout Constants ──────────────────────────────────────────────────
SCREEN_W   = 1200
SCREEN_H   = 750
CANVAS_X   = 260          # left panel width
CANVAS_W   = 620
CANVAS_H   = 620
PANEL_R_W  = 320
BAR_H      = 80

# ─── Cluster Colors (hex) — 10 distinct colors ────────────────────────
COLORS = [
    "#E05050",  # 1  red
    "#50C878",  # 2  green
    "#5090DC",  # 3  blue
    "#DCC850",  # 4  yellow
    "#B450DC",  # 5  purple
    "#3CCECE",  # 6  cyan
    "#DC8230",  # 7  orange
    "#A0DC50",  # 8  lime
    "#E05DA0",  # 9  pink
    "#50DCAA",  # 10 teal
]
COLORS_DIM = [
    "#7a2c2c",  # 1
    "#2c6e44",  # 2
    "#2c5080",  # 3
    "#6e6428",  # 4
    "#5a2870",  # 5
    "#1e6464",  # 6
    "#6e4018",  # 7
    "#506428",  # 8
    "#7a2c54",  # 9
    "#286e58",  # 10
]

# ─── Algorithm Phases ──────────────────────────────────────────────────
IDLE      = 0
INIT      = 1
ASSIGN    = 2
UPDATE    = 3
CONVERGED = 4

PHASE_NAMES = ["IDLE", "INIT", "ASSIGN", "UPDATE", "CONVERGED"]
PHASE_COLORS = ["#555566", "#DCC850", "#50C878", "#5090DC", "#E05050"]

# ─── Educational Lesson Content ───────────────────────────────────────
LESSONS = {
    IDLE: {
        "title": "What is K-Means?",
        "body": (
            "K-Means is an UNSUPERVISED\n"
            "learning algorithm.\n\n"
            "It finds hidden structure\n"
            "in data with NO labels.\n\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "HOW TO START:\n\n"
            " 1. Click canvas to\n"
            "    add data points\n\n"
            " 2. Press [STEP] or\n"
            "    [AUTO RUN]\n\n"
            " 3. Watch the algorithm\n"
            "    learn step by step!"
        ),
    },
    INIT: {
        "title": "Step 1: Initialize",
        "body": (
            "K centroids are placed\n"
            "using KMeans++ method.\n\n"
            "KMeans++ picks each\n"
            "centroid FAR from\n"
            "existing ones — this\n"
            "avoids bad starts.\n\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "WHY NOT RANDOM?\n\n"
            "Random init can place\n"
            "two centroids in the\n"
            "same cluster, causing\n"
            "poor final results.\n\n"
            "Centroids = ✕ symbols\n\n"
            "Press STEP to assign →"
        ),
    },
    ASSIGN: {
        "title": "Step 2: Assignment",
        "body": (
            "Each point finds its\n"
            "NEAREST centroid using\n"
            "Euclidean distance:\n\n"
            "  d = √(\n"
            "    (x₂-x₁)² +\n"
            "    (y₂-y₁)²\n"
            "  )\n\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "This is the\n"
            "EXPECTATION step\n"
            "in the EM algorithm.\n\n"
            "Points are colored\n"
            "by their cluster.\n\n"
            "Lines show which\n"
            "centroid owns each pt."
        ),
    },
    UPDATE: {
        "title": "Step 3: Update",
        "body": (
            "Each centroid MOVES\n"
            "to the mean (average)\n"
            "of its cluster:\n\n"
            "  cx = Σx / n\n"
            "  cy = Σy / n\n\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "This MINIMIZES the\n"
            "Within-Cluster Sum\n"
            "of Squares (WCSS):\n\n"
            "  WCSS = Σ dist(\n"
            "    point, centroid)²\n\n"
            "Arrows show centroid\n"
            "movement direction."
        ),
    },
    CONVERGED: {
        "title": "✓ Converged!",
        "body": (
            "No points changed\n"
            "cluster — the algorithm\n"
            "has CONVERGED.\n\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "IMPORTANT FACTS:\n\n"
            "✓ K-Means always\n"
            "  converges\n\n"
            "✗ NOT guaranteed to\n"
            "  find global optimum\n\n"
            "✗ Result depends on\n"
            "  initial centroids\n\n"
            "TIP: Run multiple\n"
            "times & pick lowest\n"
            "WCSS result!"
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ══════════════════════════════════════════════════════════════════════
class KMeansModel:
    def __init__(self):
        self.points    = []     # list of [x, y, cluster_idx]
        self.centroids = []     # list of [x, y, prev_x, prev_y, count]
        self.K         = 3
        self.phase     = IDLE
        self.iteration = 0
        self.changes   = 0
        self.cluster_names = {}  # k -> user-defined name string

    def reset(self):
        self.points        = []
        self.centroids     = []
        self.phase         = IDLE
        self.iteration     = 0
        self.changes       = 0
        self.cluster_names = {}

    def set_k(self, k):
        self.K = max(1, min(10, k))
        # Always fully reset so the new K takes effect cleanly
        self.phase         = IDLE
        self.iteration     = 0
        self.centroids     = []
        self.changes       = 0
        self.cluster_names = {}
        for p in self.points:
            p[2] = -1

    def _dist(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    # ── KMeans++ initialization ────────────────────────────────────────
    def _init_centroids(self):
        if len(self.points) < self.K:
            return
        pts = self.points
        first = random.choice(pts)
        self.centroids = [[first[0], first[1], first[0], first[1], 0]]

        for _ in range(self.K - 1):
            dists = []
            for p in pts:
                md = min(self._dist(p[0],p[1],c[0],c[1]) for c in self.centroids)
                dists.append(md ** 2)
            total = sum(dists)

            chosen = None
            if total == 0:
                # All points sit exactly on existing centroids — pick farthest unique point
                candidates = [p for p in pts if not any(
                    self._dist(p[0],p[1],c[0],c[1]) < 1e-9 for c in self.centroids)]
                chosen = random.choice(candidates) if candidates else pts[0]
            else:
                r = random.random() * total
                for i, p in enumerate(pts):
                    r -= dists[i]
                    if r <= 0:
                        chosen = p
                        break
                if chosen is None:        # floating-point edge case
                    chosen = max(zip(pts, dists), key=lambda x: x[1])[0]

            self.centroids.append([chosen[0], chosen[1], chosen[0], chosen[1], 0])

        for p in self.points:
            p[2] = -1

    # ── Assignment step ────────────────────────────────────────────────
    def _assign(self):
        changes = 0
        for p in self.points:
            best, bd = 0, float('inf')
            for k, c in enumerate(self.centroids):
                d = self._dist(p[0], p[1], c[0], c[1])
                if d < bd:
                    bd, best = d, k
            if p[2] != best:
                p[2] = best
                changes += 1
        for k, c in enumerate(self.centroids):
            c[4] = sum(1 for p in self.points if p[2] == k)
        return changes

    # ── Update step ────────────────────────────────────────────────────
    def _update(self):
        for k, c in enumerate(self.centroids):
            cluster_pts = [p for p in self.points if p[2] == k]
            c[2], c[3] = c[0], c[1]   # save prev position
            if cluster_pts:
                c[0] = sum(p[0] for p in cluster_pts) / len(cluster_pts)
                c[1] = sum(p[1] for p in cluster_pts) / len(cluster_pts)

    # ── WCSS ──────────────────────────────────────────────────────────
    def wcss(self):
        if self.phase <= INIT:
            return 0.0
        return sum(
            self._dist(p[0], p[1], self.centroids[p[2]][0], self.centroids[p[2]][1])**2
            for p in self.points if p[2] >= 0
        )

    # ── Next step ─────────────────────────────────────────────────────
    def step(self):
        if len(self.points) < self.K:
            return
        if self.phase == IDLE:
            self._init_centroids()
            self.phase = INIT
            self.iteration = 0
        elif self.phase == INIT:
            self.changes = self._assign()
            self.phase = ASSIGN
        elif self.phase == ASSIGN:
            self._update()
            self.iteration += 1
            self.phase = UPDATE
        elif self.phase == UPDATE:
            self.changes = self._assign()
            self.phase = CONVERGED if self.changes == 0 else ASSIGN
        elif self.phase == CONVERGED:
            self._init_centroids()
            self.phase = INIT
            self.iteration = 0

    # ── Sample dataset ────────────────────────────────────────────────
    def load_sample(self, canvas_w, canvas_h):
        self.reset()
        groups = [
            (canvas_w*0.25, canvas_h*0.25, 70),
            (canvas_w*0.70, canvas_h*0.20, 65),
            (canvas_w*0.50, canvas_h*0.60, 75),
            (canvas_w*0.20, canvas_h*0.72, 60),
            (canvas_w*0.78, canvas_h*0.68, 65),
        ]
        for gx, gy, spread in groups:
            for _ in range(18):
                angle = random.uniform(0, 2*math.pi)
                r     = random.uniform(0, spread)
                self.points.append([gx + math.cos(angle)*r,
                                    gy + math.sin(angle)*r, -1])


# ══════════════════════════════════════════════════════════════════════
#  GUI APPLICATION
# ══════════════════════════════════════════════════════════════════════
class App:
    def __init__(self, root):
        self.root    = root
        self.model   = KMeansModel()
        self.autorun = False
        self.auto_delay = 700   # ms between auto steps
        self._auto_job  = None
        self.show_lines = tk.BooleanVar(value=True)
        self.show_arrows = tk.BooleanVar(value=True)

        root.title("K-Means Clustering — Unsupervised Learning Visualizer")
        root.resizable(False, False)
        root.configure(bg="#0d1117")

        self._build_ui()
        self._bind_keys()
        self.refresh()

    # ── Build UI ──────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Left panel ────────────────────────────────────────────────
        self.left = tk.Frame(self.root, bg="#161b27", width=CANVAS_X,
                             height=SCREEN_H)
        self.left.pack(side=tk.LEFT, fill=tk.Y)
        self.left.pack_propagate(False)

        tk.Label(self.left, text="K-MEANS", bg="#161b27",
                 fg="#64c8ff", font=("Consolas",18,"bold")).pack(pady=(14,0))
        tk.Label(self.left, text="CLUSTERING", bg="#161b27",
                 fg="#64c8ff", font=("Consolas",18,"bold")).pack()
        tk.Frame(self.left, bg="#2a3a5a", height=2).pack(fill=tk.X, padx=10, pady=6)

        # Phase badge
        self.phase_badge = tk.Label(self.left, text="IDLE", bg="#555566",
                                    fg="white", font=("Consolas",13,"bold"),
                                    relief="flat", pady=4)
        self.phase_badge.pack(fill=tk.X, padx=12, pady=(0,8))

        # Lesson title
        self.lesson_title = tk.Label(self.left, text="", bg="#161b27",
                                     fg="#ffdc64", font=("Consolas",12,"bold"),
                                     anchor="w")
        self.lesson_title.pack(fill=tk.X, padx=12)
        tk.Frame(self.left, bg="#2a3a5a", height=1).pack(fill=tk.X, padx=10, pady=3)

        # Lesson body
        self.lesson_body = tk.Label(self.left, text="", bg="#161b27",
                                    fg="#d0d0e0", font=("Consolas",11),
                                    justify=tk.LEFT, anchor="nw")
        self.lesson_body.pack(fill=tk.BOTH, padx=14, pady=4)

        tk.Frame(self.left, bg="#2a3a5a", height=1).pack(fill=tk.X, padx=10, pady=4)

        # Progress steps
        tk.Label(self.left, text="ALGORITHM STEPS", bg="#161b27",
                 fg="#888899", font=("Consolas",9)).pack(anchor="w", padx=12)
        self.step_labels = []
        steps = [
            "1. Initialize centroids",
            "2. Assign to clusters",
            "3. Update centroids",
            "   Repeat 2-3...",
            "4. Convergence",
        ]
        for s in steps:
            lb = tk.Label(self.left, text=s, bg="#161b27",
                          fg="#505060", font=("Consolas",10), anchor="w")
            lb.pack(fill=tk.X, padx=14)
            self.step_labels.append(lb)

        tk.Frame(self.left, bg="#2a3a5a", height=1).pack(fill=tk.X, padx=10, pady=6)

        # K control
        kf = tk.Frame(self.left, bg="#161b27")
        kf.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(kf, text="K =", bg="#161b27", fg="#64c8ff",
                 font=("Consolas",13,"bold")).pack(side=tk.LEFT)
        self.k_label = tk.Label(kf, text="3", bg="#161b27", fg="#ffffff",
                                font=("Consolas",13,"bold"), width=2)
        self.k_label.pack(side=tk.LEFT, padx=4)
        tk.Button(kf, text="＋", command=lambda: self._change_k(1),
                  bg="#2a3a5a", fg="white", font=("Consolas",11,"bold"),
                  relief="flat", padx=6).pack(side=tk.LEFT, padx=2)
        tk.Button(kf, text="－", command=lambda: self._change_k(-1),
                  bg="#2a3a5a", fg="white", font=("Consolas",11,"bold"),
                  relief="flat", padx=6).pack(side=tk.LEFT, padx=2)

        # Toggles
        tk.Checkbutton(self.left, text="Show distance lines",
                       variable=self.show_lines, bg="#161b27", fg="#aaaacc",
                       selectcolor="#161b27", activebackground="#161b27",
                       font=("Consolas",10),
                       command=self.refresh).pack(anchor="w", padx=12, pady=2)
        tk.Checkbutton(self.left, text="Show move arrows",
                       variable=self.show_arrows, bg="#161b27", fg="#aaaacc",
                       selectcolor="#161b27", activebackground="#161b27",
                       font=("Consolas",10),
                       command=self.refresh).pack(anchor="w", padx=12)

        # ── Centre canvas area ─────────────────────────────────────────
        centre = tk.Frame(self.root, bg="#0d1117")
        centre.pack(side=tk.LEFT, fill=tk.BOTH)

        # Canvas
        self.canvas = tk.Canvas(centre, width=CANVAS_W, height=CANVAS_H,
                                bg="#0d1520", highlightthickness=1,
                                highlightbackground="#2a3a5a", cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Button-1>",        self._on_left_click)
        self.canvas.bind("<Button-3>",        self._on_right_click)
        # Re-bind keys on canvas so they work even after clicking
        self.canvas.bind("<space>",   lambda e: None if self._is_typing() else self._step())
        self.canvas.bind("<r>",       lambda e: None if self._is_typing() else self._reset())
        self.canvas.bind("<s>",       lambda e: None if self._is_typing() else self._sample())
        self.canvas.bind("<a>",       lambda e: None if self._is_typing() else self._toggle_auto())
        self.canvas.bind("<equal>",   lambda e: None if self._is_typing() else self._change_k(1))
        self.canvas.bind("<minus>",   lambda e: None if self._is_typing() else self._change_k(-1))
        self.canvas.bind("<plus>",    lambda e: None if self._is_typing() else self._change_k(1))
        self.canvas.bind("<KP_Add>",  lambda e: None if self._is_typing() else self._change_k(1))
        self.canvas.bind("<KP_Subtract>", lambda e: None if self._is_typing() else self._change_k(-1))
        # Make canvas focusable so key events reach it
        self.canvas.config(takefocus=True)
        self.canvas.bind("<Button-1>", self._on_left_click_focus, add="+")

        # Bottom bar
        bar = tk.Frame(centre, bg="#12161f", height=BAR_H)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        btn_style = dict(font=("Consolas",11,"bold"), relief="flat",
                         padx=10, pady=6, cursor="hand2")

        tk.Button(bar, text="⟳ RESET",   bg="#8c2020", fg="white",
                  command=self._reset,    **btn_style).pack(side=tk.LEFT, padx=6, pady=16)
        tk.Button(bar, text="⬡ SAMPLE",  bg="#205080", fg="white",
                  command=self._sample,  **btn_style).pack(side=tk.LEFT, padx=4, pady=16)
        # K controls in bottom bar (always visible, always work)
        tk.Label(bar, text="K:", bg="#12161f", fg="#64c8ff",
                 font=("Consolas",12,"bold")).pack(side=tk.LEFT, padx=(8,2))
        tk.Button(bar, text="−", bg="#1e3040", fg="white",
                  command=lambda: self._change_k(-1),
                  font=("Consolas",13,"bold"), relief="flat",
                  padx=8, pady=5, cursor="hand2").pack(side=tk.LEFT)
        self.k_bar_label = tk.Label(bar, text="3", bg="#12161f", fg="white",
                                    font=("Consolas",13,"bold"), width=2)
        self.k_bar_label.pack(side=tk.LEFT)
        tk.Button(bar, text="+", bg="#1e3040", fg="white",
                  command=lambda: self._change_k(1),
                  font=("Consolas",13,"bold"), relief="flat",
                  padx=8, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=(0,6))
        tk.Label(bar, text="(max 10)", bg="#12161f", fg="#445566",
                 font=("Consolas",9)).pack(side=tk.LEFT, padx=(0,6))
        tk.Button(bar, text="▶ STEP",    bg="#206040", fg="white",
                  command=self._step,    **btn_style).pack(side=tk.LEFT, padx=4, pady=16)
        self.auto_btn = tk.Button(bar, text="⏩ AUTO RUN", bg="#503080", fg="white",
                  command=self._toggle_auto, **btn_style)
        self.auto_btn.pack(side=tk.LEFT, padx=4, pady=16)

        # Status label
        self.status_label = tk.Label(bar, text="", bg="#12161f", fg="#aabbdd",
                                     font=("Consolas",10), wraplength=240, justify=tk.LEFT)
        self.status_label.pack(side=tk.LEFT, padx=12)

        # ── Right panel ────────────────────────────────────────────────
        self.right = tk.Frame(self.root, bg="#161b27", width=PANEL_R_W)
        self.right.pack(side=tk.LEFT, fill=tk.Y)
        self.right.pack_propagate(False)

        tk.Label(self.right, text="STATISTICS", bg="#161b27", fg="#888899",
                 font=("Consolas",10)).pack(anchor="w", padx=10, pady=(14,2))
        tk.Frame(self.right, bg="#2a3a5a", height=1).pack(fill=tk.X, padx=8, pady=2)

        self.stat_labels = {}
        for key in ["Points", "K", "Iteration", "Changes", "WCSS"]:
            f = tk.Frame(self.right, bg="#161b27")
            f.pack(fill=tk.X, padx=10, pady=1)
            tk.Label(f, text=f"{key}:", bg="#161b27", fg="#888899",
                     font=("Consolas",10), width=10, anchor="w").pack(side=tk.LEFT)
            lb = tk.Label(f, text="—", bg="#161b27", fg="white",
                          font=("Consolas",10,"bold"), anchor="w")
            lb.pack(side=tk.LEFT)
            self.stat_labels[key] = lb

        tk.Frame(self.right, bg="#2a3a5a", height=1).pack(fill=tk.X, padx=8, pady=6)
        tk.Label(self.right, text="CLUSTERS  (scroll ↕)", bg="#161b27", fg="#888899",
                 font=("Consolas",10)).pack(anchor="w", padx=10, pady=(0,2))

        # Scrollable container for clusters
        cluster_container = tk.Frame(self.right, bg="#161b27")
        cluster_container.pack(fill=tk.BOTH, expand=True, padx=4)

        self._cluster_canvas = tk.Canvas(cluster_container, bg="#161b27",
                                         highlightthickness=0)
        cluster_scroll = tk.Scrollbar(cluster_container, orient="vertical",
                                      command=self._cluster_canvas.yview)
        self._cluster_canvas.configure(yscrollcommand=cluster_scroll.set)

        cluster_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._cluster_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.cluster_frame = tk.Frame(self._cluster_canvas, bg="#161b27")
        self._cluster_window = self._cluster_canvas.create_window(
            (0, 0), window=self.cluster_frame, anchor="nw")

        # Resize inner frame when cluster_frame changes size
        def _on_frame_configure(e):
            self._cluster_canvas.configure(
                scrollregion=self._cluster_canvas.bbox("all"))
        self.cluster_frame.bind("<Configure>", _on_frame_configure)

        # Stretch inner frame to canvas width
        def _on_canvas_configure(e):
            self._cluster_canvas.itemconfig(
                self._cluster_window, width=e.width)
        self._cluster_canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse-wheel scrolling
        def _on_mousewheel(e):
            self._cluster_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        self._cluster_canvas.bind("<MouseWheel>", _on_mousewheel)
        self.cluster_frame.bind("<MouseWheel>", _on_mousewheel)

    # ── Key bindings ──────────────────────────────────────────────────
    def _is_typing(self):
        """Return True if a text Entry widget currently has keyboard focus."""
        w = self.root.focus_get()
        return isinstance(w, tk.Entry)

    def _bind_keys(self):
        self.root.bind("<space>",  lambda e: None if self._is_typing() else self._step())
        self.root.bind("<r>",      lambda e: None if self._is_typing() else self._reset())
        self.root.bind("<s>",      lambda e: None if self._is_typing() else self._sample())
        self.root.bind("<a>",      lambda e: None if self._is_typing() else self._toggle_auto())
        self.root.bind("<plus>",   lambda e: None if self._is_typing() else self._change_k(1))
        self.root.bind("<minus>",  lambda e: None if self._is_typing() else self._change_k(-1))
        self.root.bind("<equal>",  lambda e: None if self._is_typing() else self._change_k(1))

    # ── Actions ───────────────────────────────────────────────────────
    def _on_left_click_focus(self, e):
        self.canvas.focus_set()   # grab keyboard focus on every canvas click

    def _on_left_click(self, e):
        self.model.points.append([float(e.x), float(e.y), -1])
        if self.model.phase not in (IDLE, CONVERGED):
            self.model.phase = IDLE
            self.model.iteration = 0
        self.refresh()

    def _on_right_click(self, e):
        if not self.model.points:
            return
        best, bd = 0, float('inf')
        for i, p in enumerate(self.model.points):
            d = math.hypot(e.x - p[0], e.y - p[1])
            if d < bd:
                bd, best = d, i
        if bd < 20:
            self.model.points.pop(best)
            self.model.phase = IDLE
            self.model.iteration = 0
        self.refresh()

    def _step(self):
        self._stop_auto()
        self.model.step()
        self.refresh()

    def _reset(self):
        self._stop_auto()
        if hasattr(self, '_name_vars'):
            self._name_vars.clear()
        self.model.reset()
        self.refresh()

    def _sample(self):
        self._stop_auto()
        if hasattr(self, '_name_vars'):
            self._name_vars.clear()
        self.model.load_sample(CANVAS_W, CANVAS_H)
        self.refresh()

    def _change_k(self, delta):
        if hasattr(self, '_name_vars'):
            self._name_vars.clear()
        self.model.set_k(self.model.K + delta)
        self.refresh()

    def _toggle_auto(self):
        if self.autorun:
            self._stop_auto()
        else:
            self.autorun = True
            self.auto_btn.config(bg="#7030b0", text="⏹ STOP AUTO")
            self._auto_tick()

    def _stop_auto(self):
        self.autorun = False
        self.auto_btn.config(bg="#503080", text="⏩ AUTO RUN")
        if self._auto_job:
            self.root.after_cancel(self._auto_job)
            self._auto_job = None

    def _auto_tick(self):
        if not self.autorun:
            return
        if self.model.phase == CONVERGED:
            self._stop_auto()
            return
        self.model.step()
        self.refresh()
        self._auto_job = self.root.after(self.auto_delay, self._auto_tick)

    # ── Main render ───────────────────────────────────────────────────
    def refresh(self):
        m = self.model
        self._draw_canvas()
        self._update_left_panel()
        self._update_right_panel()

        # Status bar message
        msgs = {
            IDLE:      f"Add points (have {len(m.points)}), then press STEP or AUTO RUN.",
            INIT:      "Centroids placed via KMeans++. Press STEP to assign points.",
            ASSIGN:    f"Points assigned — {m.changes} change(s). Press STEP to update centroids.",
            UPDATE:    f"Iteration {m.iteration} complete. Centroids moved. Press STEP.",
            CONVERGED: "✓ CONVERGED! No points changed. Press STEP to restart or RESET.",
        }
        self.status_label.config(text=msgs[m.phase])

    # ── Canvas drawing ────────────────────────────────────────────────
    def _draw_canvas(self):
        c = self.canvas
        c.delete("all")
        m = self.model

        # Grid
        for x in range(0, CANVAS_W, 40):
            c.create_line(x, 0, x, CANVAS_H, fill="#1a2535", width=1)
        for y in range(0, CANVAS_H, 40):
            c.create_line(0, y, CANVAS_W, y, fill="#1a2535", width=1)

        # Hint if empty
        if not m.points:
            c.create_text(CANVAS_W//2, CANVAS_H//2 - 16,
                          text="Click anywhere to add data points",
                          fill="#2a4060", font=("Consolas", 16))
            c.create_text(CANVAS_W//2, CANVAS_H//2 + 16,
                          text="or press  [S]  to load sample data",
                          fill="#2a4060", font=("Consolas", 13))
            return

        # Warning if too few points
        if len(m.points) < m.K:
            c.create_text(CANVAS_W//2, CANVAS_H//2,
                          text=f"Need at least {m.K} points for K={m.K}",
                          fill="#dc8030", font=("Consolas", 14))

        # Distance lines (ASSIGN phase)
        if m.phase == ASSIGN and self.show_lines.get() and m.centroids:
            for p in m.points:
                if p[2] >= 0:
                    cent = m.centroids[p[2]]
                    col = COLORS_DIM[p[2] % len(COLORS_DIM)]
                    c.create_line(p[0], p[1], cent[0], cent[1],
                                  fill=col, width=1, dash=(3,4))

        # Arrow from old→new centroid (UPDATE phase)
        if m.phase == UPDATE and self.show_arrows.get():
            for k, cent in enumerate(m.centroids):
                px, py = cent[2], cent[3]
                nx, ny = cent[0], cent[1]
                if abs(px-nx) > 1 or abs(py-ny) > 1:
                    col = COLORS[k % len(COLORS)]
                    c.create_line(px, py, nx, ny, fill="#ffff80",
                                  width=2, arrow=tk.LAST, arrowshape=(10,12,4))

        # Data points
        R = 7
        for p in m.points:
            x, y, k = p
            if k >= 0 and m.phase > INIT:
                col  = COLORS[k % len(COLORS)]
                colD = COLORS_DIM[k % len(COLORS_DIM)]
                c.create_oval(x-R, y-R, x+R, y+R, fill=colD, outline=col, width=2)
            else:
                c.create_oval(x-R, y-R, x+R, y+R, fill="#303848", outline="#9090b0", width=2)

        # Centroids
        if m.phase >= INIT:
            CR = 13
            # Sync ALL name entries into cluster_names before drawing
            # so names show even if user never pressed Enter/FocusOut
            if hasattr(self, '_name_vars'):
                for ki, var in self._name_vars.items():
                    val = var.get().strip()
                    if val:
                        m.cluster_names[ki] = val

            for k, cent in enumerate(m.centroids):
                cx, cy = cent[0], cent[1]
                col = COLORS[k % len(COLORS)]
                c.create_oval(cx-CR, cy-CR, cx+CR, cy+CR,
                              fill="#0d1520", outline=col, width=3)
                # Cross (✕)
                c.create_line(cx-8, cy-8, cx+8, cy+8, fill=col, width=2)
                c.create_line(cx+8, cy-8, cx-8, cy+8, fill=col, width=2)

                # Always show name — fallback to "C{k+1}"
                label = m.cluster_names.get(k) or f"C{k+1}"

                # Estimate text width (approx 7px per char at font size 10)
                text_w = len(label) * 7

                # Choose side: prefer right, flip left if it would overflow
                if cx + CR + 6 + text_w > CANVAS_W - 4:
                    lx     = cx - CR - 6
                    anchor = "e"
                else:
                    lx     = cx + CR + 6
                    anchor = "w"

                # Clamp vertically so label stays inside canvas
                ly = max(10, min(cy, CANVAS_H - 14))

                # Draw a subtle dark backing rect so label is readable on any bg
                pad = 2
                if anchor == "w":
                    rx1, rx2 = lx - pad, lx + text_w + pad
                else:
                    rx1, rx2 = lx - text_w - pad, lx + pad
                c.create_rectangle(rx1, ly - 8, rx2, ly + 8,
                                   fill="#0d1520", outline="", stipple="")
                c.create_text(lx, ly, text=label,
                              fill=col, font=("Consolas", 10, "bold"), anchor=anchor)

        # Canvas label
        c.create_text(6, 6, text="CANVAS  (left-click = add point,  right-click = remove)",
                      fill="#2a3a5a", font=("Consolas",9), anchor="nw")

    # ── Left panel update ─────────────────────────────────────────────
    def _update_left_panel(self):
        m = self.model
        phase = m.phase

        self.phase_badge.config(text=PHASE_NAMES[phase],
                                bg=PHASE_COLORS[phase])
        lesson = LESSONS[phase]
        self.lesson_title.config(text=lesson["title"])
        self.lesson_body.config(text=lesson["body"])
        self.k_label.config(text=str(m.K))
        if hasattr(self, 'k_bar_label'):
            self.k_bar_label.config(text=str(m.K))

        # Progress steps coloring
        done_map    = [INIT, ASSIGN, UPDATE, UPDATE, CONVERGED]
        current_map = [INIT, ASSIGN, UPDATE, UPDATE, CONVERGED]
        for i, lb in enumerate(self.step_labels):
            if phase > done_map[i]:
                lb.config(fg="#50c878")
            elif phase == current_map[i] and i < 4:
                lb.config(fg="#ffdc64")
            elif phase == CONVERGED and i == 4:
                lb.config(fg="#ffdc64")
            else:
                lb.config(fg="#404050")

    # ── Right panel update ────────────────────────────────────────────
    def _update_right_panel(self):
        m = self.model

        self.stat_labels["Points"].config(text=str(len(m.points)))
        self.stat_labels["K"].config(text=str(m.K))
        self.stat_labels["Iteration"].config(text=str(m.iteration))
        self.stat_labels["Changes"].config(
            text=str(m.changes), fg="#ffdc64" if m.changes > 0 else "#50c878")
        wcss = m.wcss()
        self.stat_labels["WCSS"].config(
            text=f"{wcss:.1f}" if wcss > 0 else "—")

        # Cluster boxes
        for w in self.cluster_frame.winfo_children():
            w.destroy()

        # Keep a reference to name StringVars so they persist
        if not hasattr(self, '_name_vars'):
            self._name_vars = {}

        for k in range(m.K):
            col   = COLORS[k % len(COLORS)]
            count = m.centroids[k][4] if k < len(m.centroids) else 0
            pct   = (100 * count / len(m.points)) if m.points else 0

            # Outer frame with colored border
            f = tk.Frame(self.cluster_frame, bg="#1a2035",
                         highlightbackground=col, highlightthickness=1)
            f.pack(fill=tk.X, pady=2)

            # Header row: color swatch + editable name + stats
            hdr = tk.Frame(f, bg="#1a2035")
            hdr.pack(fill=tk.X)

            tk.Frame(hdr, bg=col, width=8).pack(side=tk.LEFT, fill=tk.Y)

            # Editable cluster name
            if k not in self._name_vars:
                self._name_vars[k] = tk.StringVar(value=m.cluster_names.get(k, f"Cluster {k+1}"))

            def _save_name(event, ki=k):
                self.model.cluster_names[ki] = self._name_vars[ki].get()
                self._draw_canvas()

            name_entry = tk.Entry(hdr, textvariable=self._name_vars[k],
                                  bg="#0d1520", fg=col, insertbackground=col,
                                  font=("Consolas", 10, "bold"),
                                  relief="flat", bd=2, width=13)
            name_entry.pack(side=tk.LEFT, padx=4, pady=3)
            name_entry.bind("<Return>",    _save_name)
            name_entry.bind("<FocusOut>",  _save_name)
            # Allow scrolling while hovering over the entry
            name_entry.bind("<MouseWheel>",
                lambda e: self._cluster_canvas.yview_scroll(
                    int(-1*(e.delta/120)), "units"))

            # Sync name_var from model if already set
            stored = m.cluster_names.get(k)
            if stored and self._name_vars[k].get() != stored:
                self._name_vars[k].set(stored)

            tk.Label(hdr, text=f"{count}pts {pct:.0f}%",
                     bg="#1a2035", fg="#888899",
                     font=("Consolas", 9)).pack(side=tk.RIGHT, padx=4)

            # Mini progress bar
            bar_outer = tk.Frame(f, bg="#0d1520", height=7)
            bar_outer.pack(fill=tk.X, padx=4, pady=(0,3))
            bar_outer.update_idletasks()
            bw = int((PANEL_R_W - 28) * pct / 100)
            if bw > 0:
                tk.Frame(bar_outer, bg=col, width=bw, height=7).place(x=0, y=0)

            # Centroid coords
            if k < len(m.centroids):
                cx, cy = m.centroids[k][0], m.centroids[k][1]
                tk.Label(f, text=f"  centroid: ({cx:.0f}, {cy:.0f})",
                         bg="#1a2035", fg="#555566",
                         font=("Consolas", 8)).pack(anchor="w")


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry(f"{SCREEN_W}x{SCREEN_H}")
    app = App(root)
    root.mainloop()
