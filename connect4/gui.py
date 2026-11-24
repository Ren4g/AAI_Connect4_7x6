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


BG = "#101827"
PANEL = "#192437"
PANEL_LIGHT = "#223149"
TEXT = "#edf3fc"
MUTED = "#9aabc2"
ACCENT = "#54d6c7"
RED = "#ff657a"
YELLOW = "#ffd166"
BOARD = "#3158a5"
HOLE = "#17233a"
WIN = "#69f0ae"

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


def _existing_or_none(value: str | None, fallback: str) -> str | None:
    candidate = Path(value).expanduser() if value else Path(fallback)
    return str(candidate) if candidate.is_file() else None


class ConnectFourApp:
    """Tkinter interface for human-vs-agent and agent-vs-agent games."""

    CANVAS_WIDTH = 650
    CANVAS_HEIGHT = 590
    BOARD_LEFT = 22
    BOARD_TOP = 54
    CELL = 86
    DISC_RADIUS = 32

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
        self.root.minsize(1000, 700)
        self.root.geometry("1120x770")
        self.engine_factory = engine_factory
        self.state: GameState = engine_factory()
        self.device = device
        self.busy = False
        self._ai_after_id: str | None = None
        self._last_result: GameResult | None = None
        self._hover_column: int | None = None
        self._agents: dict[tuple[int, str], Agent] = {}
        self.depth_var = tk.IntVar(value=4)
        self.mcts_simulations_var = tk.IntVar(value=300)
        self.puct_simulations_var = tk.IntVar(value=puct_simulations)
        self.az_simulations_var = tk.IntVar(value=az_simulations)
        self.mode = tk.StringVar(value="Người chơi  vs  AI")

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
        self._puct_defaults = puct_simulations
        self._az_defaults = az_simulations

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
        self.root.bind("<Escape>", lambda _event: self.new_game())
        self.draw()

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
        style.configure("App.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Title.TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI", 23, "bold"))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED,
                        font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background=PANEL, foreground=TEXT,
                        font=("Segoe UI", 13, "bold"))
        style.configure("Body.TLabel", background=PANEL, foreground=TEXT,
                        font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED,
                        font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=PANEL_LIGHT, foreground=ACCENT,
                        font=("Segoe UI", 12, "bold"), padding=(12, 10))
        style.configure("TCombobox", fieldbackground=PANEL_LIGHT, background=PANEL_LIGHT,
                        foreground=TEXT, arrowcolor=TEXT, padding=7)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL_LIGHT)],
                  foreground=[("readonly", TEXT)])
        style.configure("TSpinbox", fieldbackground=PANEL_LIGHT, background=PANEL_LIGHT,
                        foreground=TEXT, arrowcolor=TEXT, padding=5)
        style.configure("Primary.TButton", background=ACCENT, foreground=BG,
                        font=("Segoe UI", 10, "bold"), padding=(14, 10), borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#82e9db")])
        style.configure("Secondary.TButton", background=PANEL_LIGHT, foreground=TEXT,
                        font=("Segoe UI", 10, "bold"), padding=(12, 9), borderwidth=0)
        style.map("Secondary.TButton", background=[("active", "#30435f")])

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="App.TFrame", padding=(26, 20, 26, 12))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="CONNECT FOUR", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Game AI · 7 cột × 6 hàng", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))

        content = ttk.Frame(self.root, style="App.TFrame", padding=(24, 4, 24, 24))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        board_panel = ttk.Frame(content, style="Panel.TFrame", padding=18)
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
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ttk.Label(board_panel, text="Chọn một cột để thả quân", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )

        side = ttk.Frame(content, style="Panel.TFrame", padding=20, width=350)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        ttk.Label(side, text="VÁN ĐẤU", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 12))
        ttk.Label(side, text="Chế độ", style="Body.TLabel").pack(anchor="w", pady=(0, 5))
        self.mode_box = ttk.Combobox(
            side,
            textvariable=self.mode,
            values=("Người chơi  vs  AI", "AI  vs  AI"),
            state="readonly",
        )
        self.mode_box.pack(fill="x", pady=(0, 12))
        self.mode_box.bind("<<ComboboxSelected>>", self._on_mode_changed)

        ttk.Label(side, text="Tác tử Người chơi 1", style="Body.TLabel").pack(anchor="w", pady=(0, 5))
        self.player_one_box = ttk.Combobox(
            side, textvariable=self.player_one_agent, values=self.agent_choices, state="readonly"
        )
        self.player_one_box.pack(fill="x", pady=(0, 11))

        ttk.Label(side, text="Tác tử Người chơi 2", style="Body.TLabel").pack(anchor="w", pady=(0, 5))
        self.player_two_box = ttk.Combobox(
            side, textvariable=self.player_two_agent, values=self.agent_choices, state="readonly"
        )
        self.player_two_box.pack(fill="x", pady=(0, 17))

        ttk.Separator(side).pack(fill="x", pady=(0, 14))
        ttk.Label(side, text="THIẾT LẬP TÌM KIẾM", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 10))
        self._spin_row(side, "Alpha-Beta depth", self.depth_var, 1, 10)
        self._spin_row(side, "MCTS simulations", self.mcts_simulations_var, 10, 5000, increment=50)
        self._spin_row(side, "PUCT simulations · cấp 7", self.puct_simulations_var, 10, 5000, increment=50)
        self._spin_row(side, "PUCT simulations · cấp 8", self.az_simulations_var, 10, 5000, increment=50)

        ttk.Separator(side).pack(fill="x", pady=(14, 14))
        self.status = ttk.Label(side, text="Lượt của Người chơi 1", style="Status.TLabel", anchor="center")
        self.status.pack(fill="x", pady=(0, 12))
        self.detail = ttk.Label(side, text="Đỏ đi trước", style="Muted.TLabel", anchor="center")
        self.detail.pack(fill="x", pady=(0, 18))

        buttons = ttk.Frame(side, style="Panel.TFrame")
        buttons.pack(fill="x", side="bottom")
        ttk.Button(buttons, text="Ván mới", style="Primary.TButton", command=self.new_game).pack(
            fill="x", pady=(0, 8)
        )
        ttk.Button(buttons, text="Thoát", style="Secondary.TButton", command=self.root.destroy).pack(fill="x")

        if self.policy_agent is None:
            note = "Một số cấp cần checkpoint; truyền bằng tham số CLI để bật."
            ttk.Label(side, text=note, style="Muted.TLabel", wraplength=300).pack(anchor="w", pady=(14, 0))
        self._on_mode_changed()

    @staticmethod
    def _spin_row(
        parent: ttk.Frame,
        label: str,
        variable: tk.IntVar,
        minimum: int,
        maximum: int,
        increment: int = 1,
    ) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, style="Muted.TLabel").pack(side="left")
        ttk.Spinbox(
            row, from_=minimum, to=maximum, increment=increment, textvariable=variable, width=7
        ).pack(side="right")

    def _on_mode_changed(self, _event: tk.Event | None = None) -> None:
        if self.mode.get() == "Người chơi  vs  AI":
            self.player_one_box.configure(state="disabled")
            self.player_two_box.configure(state="readonly")
            self.detail.configure(text="Bạn là Đỏ · AI là Vàng")
        else:
            self.player_one_box.configure(state="readonly")
            self.player_two_box.configure(state="readonly")
            self.detail.configure(text="Hai tác tử sẽ lần lượt đánh")

    def new_game(self) -> None:
        if self._ai_after_id is not None:
            self.root.after_cancel(self._ai_after_id)
            self._ai_after_id = None
        self.state = self.engine_factory()
        self._agents.clear()
        self.busy = False
        self._last_result = None
        self.status.configure(text="Lượt của Người chơi 1", foreground=ACCENT)
        self.detail.configure(text="Đỏ đi trước" if self.mode.get() == "Người chơi  vs  AI" else "Hai tác tử sẽ lần lượt đánh")
        self.draw()
        if self.mode.get() == "AI  vs  AI":
            self._schedule_ai_turn(350)

    def _on_click(self, event: tk.Event) -> None:
        if self.busy or self.state.is_terminal() or self.mode.get() != "Người chơi  vs  AI":
            return
        if not self.BOARD_TOP <= event.y < self.BOARD_TOP + ROWS * self.CELL:
            return
        column = int((event.x - self.BOARD_LEFT) // self.CELL)
        if column not in self.state.legal_actions():
            return
        try:
            self.state.step(int(column))
        except ValueError:
            return
        self.draw()
        if not self.state.is_terminal():
            self._schedule_ai_turn(250)

    def _schedule_ai_turn(self, delay_ms: int) -> None:
        self.busy = True
        self._ai_after_id = self.root.after(delay_ms, self._play_ai_turn)

    def _play_ai_turn(self) -> None:
        self._ai_after_id = None
        if self.state.is_terminal():
            self.busy = False
            return
        player = int(self.state.current_player)
        label = self.player_one_agent.get() if player == 1 else self.player_two_agent.get()
        try:
            action = self._agent_for(player, label).choose_action(self.state)
            self.state.step(int(action))
        except (RuntimeError, ValueError) as error:
            self.busy = False
            self.status.configure(text="Không thể chọn nước đi", foreground=RED)
            self.detail.configure(text=str(error))
            self.draw()
            return
        self.busy = False
        self.draw()
        if self.mode.get() == "AI  vs  AI" and not self.state.is_terminal():
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
            agent = AlphaBetaAgent(depth=max(1, self.depth_var.get()), time_check_interval=1024)
        elif level == 4:
            agent = MCTSAgent(simulations=max(1, self.mcts_simulations_var.get()), seed=200 + player)
        elif level == 5 and self.policy_agent is not None:
            agent = self.policy_agent
        elif level == 6 and self.rl_agent is not None:
            agent = self.rl_agent
        elif level == 7 and self.puct_checkpoint is not None:
            agent = PUCTAgent.from_checkpoint(
                self.puct_checkpoint,
                simulations=max(1, self.puct_simulations_var.get()),
                device=self.device,
            )
        elif level == 8 and self.az_checkpoint is not None:
            agent = PUCTAgent.from_checkpoint(
                self.az_checkpoint,
                simulations=max(1, self.az_simulations_var.get()),
                device=self.device,
                safety_filter=True,
            )
        else:
            raise RuntimeError(f"Checkpoint cho {label} chưa được nạp.")
        self._agents[key] = agent
        return agent

    def _on_motion(self, event: tk.Event) -> None:
        if self.mode.get() != "Người chơi  vs  AI" or self.busy or self.state.is_terminal():
            return
        column = int((event.x - self.BOARD_LEFT) // self.CELL)
        self._hover_column = column if 0 <= column < COLS and column in self.state.legal_actions() else None
        self.draw()

    def _on_leave(self, _event: tk.Event) -> None:
        self._hover_column = None
        self.draw()

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
        self.canvas.delete("all")
        left = self.BOARD_LEFT
        top = self.BOARD_TOP
        board_width = COLS * self.CELL
        board_height = ROWS * self.CELL
        self.canvas.create_rectangle(
            left - 7, top - 7, left + board_width + 7, top + board_height + 7,
            fill=BOARD, outline="#527bc8", width=2,
        )
        winning_cells = self._winning_line() if self.state.result() in (GameResult.PLAYER_ONE, GameResult.PLAYER_TWO) else set()

        if self._hover_column is not None and self.state.result() is GameResult.ONGOING:
            x = left + self._hover_column * self.CELL + self.CELL / 2
            color = RED if self.state.current_player is Player.ONE else YELLOW
            self.canvas.create_oval(x - 15, 12, x + 15, 42, fill=color, outline="")

        for row in range(ROWS):
            for column in range(COLS):
                x = left + column * self.CELL + self.CELL / 2
                y = top + row * self.CELL + self.CELL / 2
                value = self.state.board[row][column]
                if value == 0:
                    fill = HOLE
                    outline = "#28436f"
                elif value == int(Player.ONE):
                    fill = RED
                    outline = "#ff9aa8"
                else:
                    fill = YELLOW
                    outline = "#ffe5a1"
                radius = self.DISC_RADIUS
                self.canvas.create_oval(
                    x - radius, y - radius, x + radius, y + radius,
                    fill=fill, outline=outline, width=2,
                )
                if (row, column) in winning_cells:
                    self.canvas.create_oval(
                        x - radius - 5, y - radius - 5, x + radius + 5, y + radius + 5,
                        fill="", outline=WIN, width=5,
                    )
                    self.canvas.create_oval(
                        x - 7, y - 7, x + 7, y + 7, fill=WIN, outline="",
                    )

        result = self.state.result()
        if result is GameResult.ONGOING:
            player = int(self.state.current_player)
            color = RED if player == 1 else YELLOW
            player_name = "Người chơi 1" if player == 1 else "Người chơi 2"
            if self.mode.get() == "AI  vs  AI":
                player_name += " · " + (self.player_one_agent.get() if player == 1 else self.player_two_agent.get()).split("·", 1)[-1].strip()
            elif player == 2:
                player_name = "AI đang suy nghĩ…"
                color = ACCENT
            self.status.configure(text=f"Lượt: {player_name}", foreground=color)
            self.detail.configure(text="Đỏ" if player == 1 else "Vàng")
        elif result is GameResult.DRAW:
            self.status.configure(text="HÒA · Bàn cờ đã kín", foreground=YELLOW)
            self.detail.configure(text="Bấm ‘Ván mới’ để chơi lại")
        else:
            winner = int(result)
            winner_name = "Người chơi 1 · Đỏ" if winner == 1 else "Người chơi 2 · Vàng"
            if self.mode.get() == "AI  vs  AI":
                label = self.player_one_agent.get() if winner == 1 else self.player_two_agent.get()
                winner_name = f"Người chơi {winner} · {label.split('·', 1)[-1].strip()}"
            self.status.configure(text=f"THẮNG · {winner_name}", foreground=WIN)
            self.detail.configure(text="Chuỗi bốn quân thắng đã được tô sáng")


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
