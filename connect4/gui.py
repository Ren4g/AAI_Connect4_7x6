from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from .agents import (
    Agent,
    AlphaBetaAgent,
    HeuristicAgent,
    MCTSAgent,
    PUCTAgent,
    PolicyAgent,
    RandomAgent,
)
from .engine import COLS, ROWS, ConnectFour, GameResult, Player
from .engine_bit import ConnectFour as BitboardConnectFour
from .state import GameState


# ── Bảng màu ────────────────────────────────────────────────────────────────
BG = "#0f1626"
PANEL = "#182236"
PANEL_LIGHT = "#223149"
PANEL_HOVER = "#30435f"
TEXT = "#edf3fc"
MUTED = "#9aabc2"
ACCENT = "#54d6c7"
ACCENT_HOVER = "#82e9db"
RED = "#ff657a"
RED_DARK = "#d9435c"
YELLOW = "#ffd166"
YELLOW_DARK = "#d9a93f"
BOARD = "#3158a5"
BOARD_EDGE = "#527bc8"
BOARD_HOVER = "#3d68bd"
HOLE = "#15203a"
WIN = "#69f0ae"

# ── Chế độ chơi ─────────────────────────────────────────────────────────────
MODE_HUMAN_AI = "human_ai"
MODE_HUMAN_HUMAN = "human_human"
MODE_AI_AI = "ai_ai"

MODES = {
    MODE_HUMAN_AI: ("Người – Máy", "Bạn cầm quân Đỏ và đi trước. Máy cầm quân Vàng, chọn tác tử bên dưới."),
    MODE_HUMAN_HUMAN: ("Người – Người", "Hai người luân phiên bấm vào cột trên cùng một máy. Đỏ đi trước."),
    MODE_AI_AI: ("Máy – Máy", "Hai tác tử tự đấu với nhau. Chọn tác tử cho từng bên."),
}

MODE_HINTS = {
    MODE_HUMAN_AI: "Bấm vào một cột để thả quân  ·  Esc: ván mới",
    MODE_HUMAN_HUMAN: "Lần lượt bấm vào cột để thả quân  ·  Esc: ván mới",
    MODE_AI_AI: "Hai tác tử tự đánh  ·  Esc: ván mới",
}

AGENT_LEVELS = {
    "Cấp 1  ·  Random": 1,
    "Cấp 2  ·  Heuristic": 2,
    "Cấp 3  ·  Alpha-Beta": 3,
    "Cấp 4  ·  MCTS-UCT": 4,
    "Cấp 5  ·  CNN Policy": 5,
    "Cấp 6  ·  Policy Gradient": 6,
    "Cấp 7  ·  AlphaGo-inspired PUCT": 7,
    "Cấp 8  ·  AlphaZero-inspired": 8,
}

FONT = "Segoe UI"


def _existing_or_none(value: str | None, fallback: str) -> str | None:
    candidate = Path(value).expanduser() if value else Path(fallback)
    return str(candidate) if candidate.is_file() else None


class ConnectFourApp:
    """Tkinter interface: Người–Máy, Người–Người và Máy–Máy."""

    CELL = 80
    DISC_RADIUS = 30
    BOARD_LEFT = 24
    BOARD_TOP = 56
    CANVAS_WIDTH = BOARD_LEFT * 2 + COLS * CELL
    CANVAS_HEIGHT = BOARD_TOP + ROWS * CELL + 44

    def __init__(
        self,
        root: tk.Tk,
        engine_factory=ConnectFour,
        checkpoint: str | None = None,
        device: str = "cpu",
        rl_checkpoint: str | None = None,
        puct_checkpoint: str | None = None,
        puct_simulations: int = 200,
        az_checkpoint: str | None = None,
        az_simulations: int = 400,
    ) -> None:
        self.root = root
        self.root.title("Connect Four · Game AI")
        self.root.configure(bg=BG)
        self.root.minsize(1040, 760)
        self.root.geometry("1160x810")
        self.engine_factory = engine_factory
        self.state: GameState = engine_factory()
        self.device = device
        self.busy = False
        self._ai_after_id: str | None = None
        self._hover_column: int | None = None
        self._agents: dict[tuple[int, str], Agent] = {}
        self._scored = False
        self.scores = {1: 0, 2: 0, 0: 0}

        self.depth_var = tk.IntVar(value=4)
        self.mcts_simulations_var = tk.IntVar(value=300)
        self.puct_simulations_var = tk.IntVar(value=puct_simulations)
        self.az_simulations_var = tk.IntVar(value=az_simulations)
        for var in (
            self.depth_var,
            self.mcts_simulations_var,
            self.puct_simulations_var,
            self.az_simulations_var,
        ):
            # Đổi thông số tìm kiếm thì tác tử đã cache phải được tạo lại.
            var.trace_add("write", lambda *_: self._agents.clear())

        self.mode = tk.StringVar(value=MODE_HUMAN_AI)

        self.policy_agent = self._load_policy(
            _existing_or_none(checkpoint, "results/colab/cnn_v3_30k_seed20261122/checkpoint_best.pt")
        )
        self.rl_agent = self._load_policy(
            _existing_or_none(rl_checkpoint, "results/colab/rl_seed20261120_run1/checkpoint_policy.pt")
        )
        self.puct_checkpoint = _existing_or_none(
            puct_checkpoint, "results/colab/az_seed20261120_run1/checkpoint_policy.pt"
        )
        self.az_checkpoint = _existing_or_none(
            az_checkpoint,
            "results/phase_c_local/az_v2_warm_seed20261121/checkpoints/checkpoint_iteration_0000.pt",
        )

        self.agent_choices = list(AGENT_LEVELS)
        if self.policy_agent is None:
            self.agent_choices.remove("Cấp 5  ·  CNN Policy")
        if self.rl_agent is None:
            self.agent_choices.remove("Cấp 6  ·  Policy Gradient")
        if self.puct_checkpoint is None:
            self.agent_choices.remove("Cấp 7  ·  AlphaGo-inspired PUCT")
        if self.az_checkpoint is None:
            self.agent_choices.remove("Cấp 8  ·  AlphaZero-inspired")

        self.player_one_agent = tk.StringVar(value="Cấp 2  ·  Heuristic")
        self.player_two_agent = tk.StringVar(value="Cấp 3  ·  Alpha-Beta")

        self._configure_style()
        self._build_layout()
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self.root.bind("<Escape>", lambda _e: self.new_game())
        self.root.bind("<Return>", lambda _e: self.new_game() if self.state.is_terminal() else None)
        self.refresh()

    # ── Khởi tạo ────────────────────────────────────────────────────────────
    def _load_policy(self, path: str | None) -> PolicyAgent | None:
        if path is None:
            return None
        try:
            return PolicyAgent.from_checkpoint(path, device=self.device)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            print(f"Could not load checkpoint {path}: {error}")
            return None

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("PanelTitle.TLabel", background=PANEL, foreground=TEXT,
                        font=(FONT, 11, "bold"))
        style.configure("Body.TLabel", background=PANEL, foreground=TEXT, font=(FONT, 10))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=(FONT, 9))
        style.configure("TCombobox", fieldbackground=PANEL_LIGHT, background=PANEL_LIGHT,
                        foreground=TEXT, arrowcolor=TEXT, padding=6)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL_LIGHT)],
                  foreground=[("readonly", TEXT)])
        style.configure("TSpinbox", fieldbackground=PANEL_LIGHT, background=PANEL_LIGHT,
                        foreground=TEXT, arrowcolor=TEXT, padding=4)
        style.configure("TSeparator", background=PANEL_LIGHT)
        self.root.option_add("*TCombobox*Listbox.background", PANEL_LIGHT)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", BG)

    @staticmethod
    def _make_button(parent, text, command, bg, fg, hover, font=(FONT, 10, "bold"), pady=10) -> tk.Label:
        """Nút dạng Label để màu hiển thị đúng trên mọi hệ điều hành."""
        button = tk.Label(parent, text=text, bg=bg, fg=fg, font=font, cursor="hand2", pady=pady)
        button.normal_bg = bg
        button.hover_bg = hover
        button.bind("<Button-1>", lambda _e: command())
        button.bind("<Enter>", lambda _e: button.configure(bg=button.hover_bg))
        button.bind("<Leave>", lambda _e: button.configure(bg=button.normal_bg))
        return button

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Header
        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(18, 8))
        tk.Label(header, text="CONNECT FOUR", bg=BG, fg=TEXT, font=(FONT, 22, "bold")).pack(side="left")
        tk.Label(header, text="Game AI  ·  7 cột × 6 hàng", bg=BG, fg=MUTED, font=(FONT, 10)).pack(
            side="left", padx=(14, 0), pady=(10, 0)
        )

        content = tk.Frame(self.root, bg=BG)
        content.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 22))
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        # Bàn cờ
        board_panel = tk.Frame(content, bg=PANEL, padx=18, pady=16)
        board_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        board_panel.columnconfigure(0, weight=1)
        board_panel.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            board_panel,
            width=self.CANVAS_WIDTH,
            height=self.CANVAS_HEIGHT,
            bg=PANEL,
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.grid(row=0, column=0)

        # Thanh kết quả + nút chơi lại (chỉ hiện khi hết ván)
        self.result_bar = tk.Frame(board_panel, bg=PANEL_LIGHT, padx=16, pady=10)
        self.result_bar.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        text_box = tk.Frame(self.result_bar, bg=PANEL_LIGHT)
        text_box.pack(side="left", fill="x", expand=True)
        self.result_title = tk.Label(text_box, text="", bg=PANEL_LIGHT, fg=WIN,
                                     font=(FONT, 15, "bold"), anchor="w")
        self.result_title.pack(fill="x")
        self.result_detail = tk.Label(text_box, text="", bg=PANEL_LIGHT, fg=MUTED,
                                      font=(FONT, 9), anchor="w")
        self.result_detail.pack(fill="x")
        self._make_button(
            self.result_bar, "↻  Chơi lại", self.new_game, ACCENT, BG, ACCENT_HOVER,
            font=(FONT, 11, "bold"), pady=9,
        ).pack(side="right", ipadx=18)
        self.result_bar.grid_remove()

        self.hint = tk.Label(board_panel, text="", bg=PANEL, fg=MUTED, font=(FONT, 9), anchor="w")
        self.hint.grid(row=2, column=0, sticky="w", pady=(10, 0))

        # Bảng điều khiển
        side = ttk.Frame(content, style="Panel.TFrame", padding=20, width=380)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)

        # Nút dưới cùng (pack trước để luôn hiện)
        bottom = tk.Frame(side, bg=PANEL)
        bottom.pack(side="bottom", fill="x")
        self._make_button(bottom, "↻  Ván mới", self.new_game, ACCENT, BG, ACCENT_HOVER).pack(
            fill="x", pady=(0, 8)
        )
        self._make_button(bottom, "Thoát", self.root.destroy, PANEL_LIGHT, TEXT, PANEL_HOVER).pack(fill="x")

        # Chế độ chơi
        ttk.Label(side, text="CHẾ ĐỘ CHƠI", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 8))
        mode_row = tk.Frame(side, bg=PANEL)
        mode_row.pack(fill="x")
        self.mode_buttons: dict[str, tk.Label] = {}
        for index, (key, (title, _desc)) in enumerate(MODES.items()):
            button = self._make_button(
                mode_row, title, lambda k=key: self._set_mode(k),
                PANEL_LIGHT, TEXT, PANEL_HOVER, font=(FONT, 9, "bold"), pady=9,
            )
            button.pack(side="left", fill="x", expand=True, padx=(0 if index == 0 else 4, 0))
            self.mode_buttons[key] = button
        self.mode_desc = ttk.Label(side, text="", style="Muted.TLabel", wraplength=335, justify="left")
        self.mode_desc.pack(anchor="w", pady=(8, 12))

        # Hai người chơi
        self.cards: dict[int, tk.Frame] = {}
        self.roles: dict[int, tk.Label] = {}
        self.boxes: dict[int, ttk.Combobox] = {}
        self.score_labels: dict[int, tk.Label] = {}
        self._build_player_card(side, 1, "Người chơi 1  ·  Đỏ", RED, self.player_one_agent)
        self._build_player_card(side, 2, "Người chơi 2  ·  Vàng", YELLOW, self.player_two_agent)
        self.draw_label = ttk.Label(side, text="Hòa: 0", style="Muted.TLabel")
        self.draw_label.pack(anchor="e", pady=(4, 0))

        ttk.Separator(side).pack(fill="x", pady=(8, 10))
        ttk.Label(side, text="THIẾT LẬP TÌM KIẾM", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 6))
        self._spin_row(side, "Alpha-Beta depth", self.depth_var, 1, 10)
        self._spin_row(side, "MCTS simulations", self.mcts_simulations_var, 10, 5000, increment=50)
        self._spin_row(side, "PUCT simulations · cấp 7", self.puct_simulations_var, 10, 5000, increment=50)
        self._spin_row(side, "PUCT simulations · cấp 8", self.az_simulations_var, 10, 5000, increment=50)

        if len(self.agent_choices) < len(AGENT_LEVELS):
            ttk.Label(
                side,
                text="Một số cấp chưa khả dụng vì thiếu checkpoint; truyền bằng tham số CLI để bật.",
                style="Muted.TLabel", wraplength=335, justify="left",
            ).pack(anchor="w", pady=(8, 0))

        ttk.Separator(side).pack(fill="x", pady=(10, 10))
        self.status = tk.Label(side, text="", bg=PANEL_LIGHT, fg=ACCENT, font=(FONT, 12, "bold"),
                               pady=10, wraplength=335)
        self.status.pack(fill="x")
        self.detail = ttk.Label(side, text="", style="Muted.TLabel", anchor="center", wraplength=335)
        self.detail.pack(fill="x", pady=(6, 0))

        self._apply_mode_ui()

    def _build_player_card(self, parent, player: int, title: str, color: str, var: tk.StringVar) -> None:
        card = tk.Frame(parent, bg=PANEL_LIGHT, highlightthickness=2,
                        highlightbackground=PANEL_LIGHT, highlightcolor=PANEL_LIGHT, padx=12, pady=8)
        card.pack(fill="x", pady=(0, 8))
        head = tk.Frame(card, bg=PANEL_LIGHT)
        head.pack(fill="x")
        dot = tk.Canvas(head, width=20, height=20, bg=PANEL_LIGHT, highlightthickness=0)
        dot.create_oval(2, 2, 18, 18, fill=color, outline="")
        dot.pack(side="left")
        tk.Label(head, text=title, bg=PANEL_LIGHT, fg=TEXT, font=(FONT, 10, "bold")).pack(side="left", padx=(8, 0))
        score = tk.Label(head, text="Thắng: 0", bg=PANEL_LIGHT, fg=color, font=(FONT, 9, "bold"))
        score.pack(side="right")
        role = tk.Label(card, text="", bg=PANEL_LIGHT, fg=MUTED, font=(FONT, 9), anchor="w")
        role.pack(fill="x", pady=(3, 0))
        box = ttk.Combobox(card, textvariable=var, values=self.agent_choices, state="readonly")
        self.cards[player] = card
        self.roles[player] = role
        self.boxes[player] = box
        self.score_labels[player] = score

    @staticmethod
    def _spin_row(parent, label: str, variable: tk.IntVar, minimum: int, maximum: int, increment: int = 1) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, style="Muted.TLabel").pack(side="left")
        ttk.Spinbox(
            row, from_=minimum, to=maximum, increment=increment, textvariable=variable, width=7
        ).pack(side="right")

    # ── Chế độ chơi ─────────────────────────────────────────────────────────
    def _set_mode(self, mode: str) -> None:
        if mode == self.mode.get():
            return
        self.mode.set(mode)
        self._apply_mode_ui()
        self.new_game()

    def _apply_mode_ui(self) -> None:
        mode = self.mode.get()
        for key, button in self.mode_buttons.items():
            selected = key == mode
            button.normal_bg = ACCENT if selected else PANEL_LIGHT
            button.hover_bg = ACCENT if selected else PANEL_HOVER
            button.configure(bg=button.normal_bg, fg=BG if selected else TEXT)
        self.mode_desc.configure(text=MODES[mode][1])
        self.hint.configure(text=MODE_HINTS[mode])
        for player in (1, 2):
            human = mode == MODE_HUMAN_HUMAN or (mode == MODE_HUMAN_AI and player == 1)
            if human:
                self.boxes[player].pack_forget()
                if mode == MODE_HUMAN_HUMAN:
                    self.roles[player].configure(text=f"Người chơi trực tiếp", fg=TEXT)
                else:
                    self.roles[player].configure(text="Bạn (người chơi)", fg=TEXT)
            else:
                self.roles[player].configure(text="Máy · chọn tác tử:", fg=MUTED)
                self.boxes[player].pack(fill="x", pady=(6, 0))

    def _agent_name(self, player: int) -> str:
        label = self.player_one_agent.get() if player == 1 else self.player_two_agent.get()
        return label.split("·", 1)[-1].strip()

    def _human_turn(self) -> bool:
        if self.busy or self.state.is_terminal():
            return False
        mode = self.mode.get()
        if mode == MODE_HUMAN_HUMAN:
            return True
        if mode == MODE_HUMAN_AI:
            return int(self.state.current_player) == 1
        return False

    @staticmethod
    def _int(variable: tk.IntVar, default: int) -> int:
        try:
            return max(1, int(variable.get()))
        except (tk.TclError, ValueError):
            return default

    # ── Luồng ván đấu ───────────────────────────────────────────────────────
    def new_game(self) -> None:
        if self._ai_after_id is not None:
            self.root.after_cancel(self._ai_after_id)
            self._ai_after_id = None
        self.state = self.engine_factory()
        self._agents.clear()
        self.busy = False
        self._scored = False
        self._hover_column = None
        self.refresh()
        if self.mode.get() == MODE_AI_AI:
            self._schedule_ai_turn(350)

    def _on_click(self, event: tk.Event) -> None:
        if not self._human_turn():
            return
        column = int((event.x - self.BOARD_LEFT) // self.CELL)
        if column not in self.state.legal_actions():
            return
        try:
            self.state.step(int(column))
        except ValueError:
            return
        self._hover_column = None
        self.refresh()
        if not self.state.is_terminal() and self.mode.get() == MODE_HUMAN_AI:
            self._schedule_ai_turn(250)

    def _schedule_ai_turn(self, delay_ms: int) -> None:
        self.busy = True
        self._ai_after_id = self.root.after(delay_ms, self._play_ai_turn)

    def _play_ai_turn(self) -> None:
        self._ai_after_id = None
        if self.state.is_terminal():
            self.busy = False
            self.refresh()
            return
        player = int(self.state.current_player)
        label = self.player_one_agent.get() if player == 1 else self.player_two_agent.get()
        try:
            action = self._agent_for(player, label).choose_action(self.state)
            self.state.step(int(action))
        except (RuntimeError, ValueError) as error:
            self.busy = False
            self.status.configure(text="Không thể chọn nước đi", fg=RED)
            self.detail.configure(text=str(error))
            self.draw()
            return
        self.busy = False
        self.refresh()
        if self.mode.get() == MODE_AI_AI and not self.state.is_terminal():
            self._schedule_ai_turn(300)

    def _agent_for(self, player: int, label: str) -> Agent:
        key = (player, label)
        if key in self._agents:
            return self._agents[key]
        level = AGENT_LEVELS[label]
        if level == 1:
            agent: Agent = RandomAgent(seed=100 + player)
        elif level == 2:
            agent = HeuristicAgent()
        elif level == 3:
            agent = AlphaBetaAgent(depth=self._int(self.depth_var, 4), time_check_interval=1024)
        elif level == 4:
            agent = MCTSAgent(simulations=self._int(self.mcts_simulations_var, 300), seed=200 + player)
        elif level == 5 and self.policy_agent is not None:
            agent = self.policy_agent
        elif level == 6 and self.rl_agent is not None:
            agent = self.rl_agent
        elif level == 7 and self.puct_checkpoint is not None:
            agent = PUCTAgent.from_checkpoint(
                self.puct_checkpoint,
                simulations=self._int(self.puct_simulations_var, 200),
                device=self.device,
            )
        elif level == 8 and self.az_checkpoint is not None:
            agent = PUCTAgent.from_checkpoint(
                self.az_checkpoint,
                simulations=self._int(self.az_simulations_var, 400),
                device=self.device,
                safety_filter=True,
            )
        else:
            raise RuntimeError(f"Checkpoint cho {label} chưa được nạp.")
        self._agents[key] = agent
        return agent

    def _on_motion(self, event: tk.Event) -> None:
        if not self._human_turn():
            if self._hover_column is not None:
                self._hover_column = None
                self.draw()
            return
        column = int((event.x - self.BOARD_LEFT) // self.CELL)
        new_hover = column if 0 <= column < COLS and column in self.state.legal_actions() else None
        if new_hover != self._hover_column:
            self._hover_column = new_hover
            self.draw()

    def _on_leave(self, _event: tk.Event) -> None:
        if self._hover_column is not None:
            self._hover_column = None
            self.draw()

    # ── Vẽ & trạng thái ─────────────────────────────────────────────────────
    def refresh(self) -> None:
        self.draw()
        self._update_status()

    def _winning_line(self) -> set[tuple[int, int]]:
        move = getattr(self.state, "last_move", None)
        if move is None:
            return set()
        row, column, player = move.row, move.column, int(move.player)
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            negative: list[tuple[int, int]] = []
            r, c = row - dr, column - dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.state.board[r][c] == player:
                negative.append((r, c))
                r, c = r - dr, c - dc
            positive: list[tuple[int, int]] = []
            r, c = row + dr, column + dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.state.board[r][c] == player:
                positive.append((r, c))
                r, c = r + dr, c + dc
            sequence = list(reversed(negative)) + [(row, column)] + positive
            pivot = len(negative)
            if len(sequence) >= 4:
                start = max(0, min(pivot - 3, len(sequence) - 4))
                return set(sequence[start:start + 4])
        return set()

    def draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        left, top, cell = self.BOARD_LEFT, self.BOARD_TOP, self.CELL
        board_width, board_height = COLS * cell, ROWS * cell
        radius = self.DISC_RADIUS

        # Bóng đổ + khung bàn
        canvas.create_rectangle(left - 4, top - 2, left + board_width + 10, top + board_height + 12,
                                fill="#0a101d", outline="")
        canvas.create_rectangle(left - 8, top - 8, left + board_width + 8, top + board_height + 8,
                                fill=BOARD, outline=BOARD_EDGE, width=2)

        result = self.state.result()
        ongoing = result is GameResult.ONGOING
        winning_cells = (
            self._winning_line() if result in (GameResult.PLAYER_ONE, GameResult.PLAYER_TWO) else set()
        )

        # Cột đang hover + quân xem trước
        if self._hover_column is not None and ongoing:
            x0 = left + self._hover_column * cell
            canvas.create_rectangle(x0, top - 4, x0 + cell, top + board_height + 4,
                                    fill=BOARD_HOVER, outline="")
            x = x0 + cell / 2
            color = RED if self.state.current_player is Player.ONE else YELLOW
            canvas.create_oval(x - 18, top - 46, x + 18, top - 10, fill=color, outline="")

        # Số cột
        for column in range(COLS):
            x = left + column * cell + cell / 2
            canvas.create_text(x, top + board_height + 28, text=str(column + 1),
                               fill=MUTED, font=(FONT, 10, "bold"))

        # Ô cờ
        for row in range(ROWS):
            for column in range(COLS):
                x = left + column * cell + cell / 2
                y = top + row * cell + cell / 2
                value = self.state.board[row][column]
                if value == 0:
                    canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                       fill=HOLE, outline="#28436f", width=2)
                    continue
                if value == int(Player.ONE):
                    fill, outline, inner = RED, "#ff9aa8", RED_DARK
                else:
                    fill, outline, inner = YELLOW, "#ffe5a1", YELLOW_DARK
                canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                   fill=fill, outline=outline, width=2)
                canvas.create_oval(x - radius * 0.6, y - radius * 0.6, x + radius * 0.6, y + radius * 0.6,
                                   fill="", outline=inner, width=2)

        # Đánh dấu nước đi gần nhất
        move = getattr(self.state, "last_move", None)
        if move is not None and not winning_cells:
            mx = left + move.column * cell + cell / 2
            my = top + move.row * cell + cell / 2
            canvas.create_oval(mx - radius - 4, my - radius - 4, mx + radius + 4, my + radius + 4,
                               fill="", outline=TEXT, width=2, dash=(4, 3))

        # Chuỗi thắng
        for row, column in winning_cells:
            x = left + column * cell + cell / 2
            y = top + row * cell + cell / 2
            canvas.create_oval(x - radius - 5, y - radius - 5, x + radius + 5, y + radius + 5,
                               fill="", outline=WIN, width=5)
            canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill=WIN, outline="")

    def _result_summary(self, result: GameResult) -> tuple[str, str, str]:
        mode = self.mode.get()
        if result is GameResult.DRAW:
            return "HÒA", "Bàn cờ đã kín, không bên nào thắng", YELLOW
        winner = int(result)
        color_name = "Đỏ" if winner == 1 else "Vàng"
        if mode == MODE_HUMAN_AI:
            if winner == 1:
                return "BẠN THẮNG!", f"Quân {color_name} nối được bốn", WIN
            return "MÁY THẮNG", f"{self._agent_name(2)} · quân {color_name}", WIN
        if mode == MODE_HUMAN_HUMAN:
            return f"NGƯỜI CHƠI {winner} THẮNG", f"Quân {color_name} nối được bốn", WIN
        return f"NGƯỜI CHƠI {winner} THẮNG", f"{self._agent_name(winner)} · quân {color_name}", WIN

    def _update_scores(self) -> None:
        for player in (1, 2):
            self.score_labels[player].configure(text=f"Thắng: {self.scores[player]}")
        self.draw_label.configure(text=f"Hòa: {self.scores[0]}")

    def _highlight_card(self, active: int | None, color: str = ACCENT) -> None:
        for player, card in self.cards.items():
            border = color if player == active else PANEL_LIGHT
            card.configure(highlightbackground=border, highlightcolor=border)

    def _update_status(self) -> None:
        result = self.state.result()
        mode = self.mode.get()

        if result is GameResult.ONGOING:
            self.result_bar.grid_remove()
            player = int(self.state.current_player)
            color = RED if player == 1 else YELLOW
            color_name = "Đỏ" if player == 1 else "Vàng"
            if mode == MODE_HUMAN_HUMAN:
                text, detail = f"Lượt Người chơi {player} · {color_name}", "Bấm vào một cột để thả quân"
            elif mode == MODE_HUMAN_AI:
                if player == 1:
                    text, detail = "Đến lượt bạn", "Bạn cầm quân Đỏ"
                else:
                    text, color = "Máy đang suy nghĩ…", ACCENT
                    detail = self._agent_name(2)
            else:
                text = f"{self._agent_name(player)} đang suy nghĩ…"
                detail = f"Người chơi {player} · quân {color_name}"
            self.status.configure(text=text, fg=color)
            self.detail.configure(text=detail)
            self._highlight_card(player, color)
            return

        # Hết ván
        if not self._scored:
            self._scored = True
            key = 0 if result is GameResult.DRAW else int(result)
            self.scores[key] += 1
            self._update_scores()
        headline, detail, color = self._result_summary(result)
        self.status.configure(text=headline, fg=color)
        self.detail.configure(text="Bấm ‘Chơi lại’ hoặc phím Enter để đấu ván mới")
        self.result_title.configure(text=headline, fg=color)
        self.result_detail.configure(text=detail)
        self.result_bar.grid()
        self._highlight_card(None if result is GameResult.DRAW else int(result), WIN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Connect Four against a selected AI level")
    parser.add_argument("--engine", choices=("grid", "bitboard"), default="grid")
    parser.add_argument("--checkpoint", default=None, help="CNN policy checkpoint")
    parser.add_argument("--rl-checkpoint", default=None, help="Policy-gradient checkpoint")
    parser.add_argument("--puct-checkpoint", default=None, help="Level 7 policy/value checkpoint")
    parser.add_argument("--az-checkpoint", default=None, help="Level 8 self-play policy/value checkpoint")
    parser.add_argument("--puct-simulations", type=int, default=200)
    parser.add_argument("--az-simulations", type=int, default=400)
    parser.add_argument("--device", default="cpu", help="Inference device for neural checkpoints")
    args = parser.parse_args()
    root = tk.Tk()
    factory = BitboardConnectFour if args.engine == "bitboard" else ConnectFour
    ConnectFourApp(
        root,
        factory,
        args.checkpoint,
        args.device,
        args.rl_checkpoint,
        args.puct_checkpoint,
        args.puct_simulations,
        args.az_checkpoint,
        args.az_simulations,
    )
    root.mainloop()


if __name__ == "__main__":
    main()