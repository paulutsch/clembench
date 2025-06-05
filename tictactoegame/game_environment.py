from typing import Dict, List, Literal, Optional

import numpy as np
from clemcore.clemgame import (
    Action,
    ActionSpace,
    GameEnvironment,
    GameState,
    Observation,
    Player,
)
from clemcore.utils.logger import setup_logger

logger = setup_logger(__name__)


class TicTacToeAction(Action):
    action_type: str
    row: int
    col: int


class TicTacToeObservation(Observation):
    role: Literal["user"]
    content: str


class TicTacToePlayer(Player):
    """Player for the TicTacToe game."""

    def __init__(self, model):
        super().__init__(model)
        self.symbol: Optional[Literal["X", "O"]] = None

    def _custom_response(self, context: Dict) -> str:
        return "Hello, world!"


class TicTacToeGameState(GameState):
    board: list[list[Literal["X", "O", "▢"]]]
    current_player: int
    winner: Optional[int]


class TicTacToeEnvironment(GameEnvironment):
    """Environment for the TicTacToe game."""

    def __init__(self):
        super().__init__()
        self.state: TicTacToeGameState = TicTacToeGameState(
            board=[["▢" for _ in range(3)] for _ in range(3)],
            current_player=1,  # 1 for X, 2 for O
            success=False,
            terminated=False,
            aborted=False,
            winner=None,
            moves=0,
            warning="",
        )
        self.base_prompt = ""
        self.config = {}
        self.observations: Dict[str, TicTacToeObservation] = {}
        self.action_spaces: Dict[str, ActionSpace] = {}

        self.players: List[TicTacToePlayer] = []

    def reset(
        self,
        initial_observations: Dict[str, Observation],
        initial_action_spaces: Dict[str, ActionSpace],
    ) -> None:
        """Reset the game environment."""
        super().reset(initial_observations, initial_action_spaces)
        self.state: TicTacToeGameState = TicTacToeGameState(
            board=[["▢" for _ in range(3)] for _ in range(3)],
            current_player=1,  # 1 for X, 2 for O
            success=False,
            terminated=False,
            aborted=False,
            winner=None,
            moves=0,
            warning="",
        )

    def format_board(self, board: list[list[Literal["X", "O", "▢"]]]) -> str:
        """Format the board for display with ASCII grid."""
        board_str = ""
        for i in range(3):
            for j in range(3):
                board_str += f"{board[i][j]}"
            board_str += "\n"
        return board_str

    def _is_action_valid_in_state(
        self, player: TicTacToePlayer, action: TicTacToeAction
    ) -> bool:
        """Check if an action is valid in the current state."""
        row = action.get("row")
        col = action.get("col")
        return self.is_valid_move(row, col)

    def is_valid_move(self, row: int | None, col: int | None) -> bool:
        """Check if a move is valid."""
        logger.info(f"Checking if move {row}, {col} is valid")

        if row is None or col is None:
            logger.warning(f"Invalid move: {row}, {col}")
            return False

        row_is_valid = 0 <= row < 3
        col_is_valid = 0 <= col < 3
        cell_is_empty = self.state["board"][row][col] == "▢"

        logger.info(
            f"Row is valid: {row_is_valid}, Col is valid: {col_is_valid}, Cell is empty: {cell_is_empty}"
        )

        return row_is_valid and col_is_valid and cell_is_empty

    def check_game_state(self) -> None:
        """Check if the game is over (win or draw)."""
        board = np.array(self.state["board"])

        self.state["success"] = True
        self.state["aborted"] = False
        self.state["terminated"] = False

        for i in range(3):
            if all(board[i, :] == "X") or all(board[:, i] == "X"):
                logger.info(f"X wins on row {i}")
                self.state["terminated"] = True
                self.state["winner"] = 1
                self.state["success"] = True
                return
            if all(board[i, :] == "O") or all(board[:, i] == "O"):
                logger.info(f"O wins on row {i}")
                self.state["terminated"] = True
                self.state["winner"] = 2
                self.state["success"] = True
                return

        if all(np.diag(board) == "X") or all(np.diag(np.fliplr(board)) == "X"):
            logger.info("X wins on diagonal")
            self.state["terminated"] = True
            self.state["winner"] = 1
            self.state["success"] = True
            return
        if all(np.diag(board) == "O") or all(np.diag(np.fliplr(board)) == "O"):
            logger.info("O wins on diagonal")
            self.state["terminated"] = True
            self.state["winner"] = 2
            self.state["success"] = True
            return

        if np.all(board != "▢"):
            logger.info("Draw")
            self.state["terminated"] = True
            self.state["winner"] = 0  # draw
            self.state["success"] = True  # let's be generous
            return

        logger.info("Game continues")

    def update_observations(self) -> None:
        """Update the observation for all players."""
        for player in self.players:
            if self.state["moves"] > 1:
                if self.state["warning"] != "":
                    prompt = self.state["warning"]
                    self.state["warning"] = ""
                else:
                    prompt = (
                        "The other player made a move. The new board is:\n\n"
                        + self.format_board(self.state["board"])
                        + "\n\nMake your next move in the format described before."
                    )
                self.observations[player.name]["content"] = prompt

            else:
                prompt = (
                    self.base_prompt
                    + f"\n\nYou are the player that plays {player.symbol}.\n\n"
                    + "The current board is:\n\n"
                    + self.format_board(self.state["board"])
                )
                self.observations[player.name]["content"] = prompt

    def _update_state_through_action(
        self, player: TicTacToePlayer, action: TicTacToeAction
    ) -> None:
        """Update the game state based on the action."""
        row = action["row"]
        col = action["col"]

        self.state["board"][row][col] = (
            "X" if self.state["current_player"] == 1 else "O"
        )

        self.check_game_state()

        self.state["current_player"] = 1 if self.state["current_player"] == 2 else 2
