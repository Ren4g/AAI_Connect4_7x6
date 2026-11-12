from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .agents import Agent, AlphaBetaAgent, HeuristicAgent, MCTSAgent, RandomAgent
from .engine import COLS, ROWS, ConnectFour, GameResult, Player


class ConnectFourApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Connect Four 7x6")
        self.state = ConnectFour()
        self.busy = False

        self.mode = tk.StringVar(value="human_vs_ai")
        self.level = tk.StringVar(value="Alpha-Beta")
        self.canvas = tk.Canvas(root, width=560, height=420, bg="#1565c0", highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
        self.status = ttk.Label(root)
        self.status.grid(row=1, column=0, columnspan=4, pady=(0, 8))

        ttk.Label(root, text="Mode").grid(row=2, column=0, sticky="e")
        ttk.Combobox(
            root, textvariable=self.mode, values=("human_vs_ai", "ai_vs_ai"),
            state="readonly", width=16,
        ).grid(row=2, column=1, sticky="w")
        ttk.Label(root, text="AI").grid(row=2, column=2, sticky="e")
        ttk.Combobox(
            root, textvariable=self.level,
            values=("Random", "Heuristic", "Alpha-Beta", "MCTS"),
            state="readonly", width=16,
        ).grid(row=2, column=3, sticky="w")
        ttk.Button(root, text="New game", command=self.new_game).grid(
            row=3, column=0, columnspan=4, pady=10
        )
        self.canvas.bind("<Button-1>", self._on_click)
        self.draw()

    def new_game(self) -> None:
        self.state.reset()
        self.busy = False
        self.draw()
        if self.mode.get() == "ai_vs_ai":
            self.root.after(200, self._play_ai_turn)

    def _on_click(self, event: tk.Event) -> None:
        if self.busy or self.state.is_terminal() or self.mode.get() != "human_vs_ai":
            return
        column = min(COLS - 1, max(0, int(event.x / (560 / COLS))))
        try:
            self.state.step(column)
        except ValueError:
            return
        self.draw()
        if not self.state.is_terminal():
            self.root.after(150, self._play_ai_turn)

    def _play_ai_turn(self) -> None:
        if self.state.is_terminal():
            return
        self.busy = True
        try:
            self.state.step(self._agent().choose_action(self.state))
        finally:
            self.busy = False
        self.draw()
        if self.mode.get() == "ai_vs_ai" and not self.state.is_terminal():
            self.root.after(150, self._play_ai_turn)

    def _agent(self) -> Agent:
        level = self.level.get()
        if level == "Random":
            return RandomAgent(seed=42)
        if level == "Heuristic":
            return HeuristicAgent()
        if level == "MCTS":
            return MCTSAgent(simulations=150, seed=42)
        return AlphaBetaAgent(depth=4)

    def draw(self) -> None:
        self.canvas.delete("all")
        width, height = 560 / COLS, 420 / ROWS
        for row in range(ROWS):
            for column in range(COLS):
                value = self.state.board[row][column]
                color = "white" if value == 0 else (
                    "#e53935" if value == int(Player.ONE) else "#fdd835"
                )
                self.canvas.create_oval(
                    column * width + 8, row * height + 8,
                    (column + 1) * width - 8, (row + 1) * height - 8,
                    fill=color, outline="#0d47a1", width=2,
                )
        result = self.state.result()
        if result is GameResult.ONGOING:
            self.status.configure(text=f"Turn: Player {int(self.state.current_player)}")
        elif result is GameResult.DRAW:
            self.status.configure(text="Draw")
            messagebox.showinfo("Game over", "Draw")
        else:
            self.status.configure(text=f"Player {int(result)} wins")
            messagebox.showinfo("Game over", f"Player {int(result)} wins")


def main() -> None:
    root = tk.Tk()
    ConnectFourApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
