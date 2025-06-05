import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from clemcore.backends import Model
from clemcore.clemgame import (
    Action,
    ActionSpace,
    GameEnvironment,
    GameState,
    Observation,
    Player,
)
from clemcore.utils.logger import setup_logger
from sudoku import Sudoku

logger = setup_logger(__name__)


class SudokuPlayer(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, context: Dict) -> str:
        return "Hello, world!"


class SudokuAction(Action):
    """Action for the Sudoku game."""

    row: int
    col: int
    value: int


class SudokuObservation(Observation):
    """Observation for the Sudoku game."""

    board: str


class SudokuGameState(GameState):
    board: List[List[int]]


class SudokuEnvironment(GameEnvironment):

    def __init__(
        self,
        board_size: int,
        difficulty: float = 0.5,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.board_size = board_size
        self.difficulty = difficulty

        puzzle = Sudoku(self.board_size).difficulty(self.difficulty)
        self.original_board = np.array(puzzle.board)
        self.original_board[self.original_board == None] = 0
        self.original_board = self.original_board.astype(int)

        self.observations: Dict[str, SudokuObservation] = {}
        self.action_spaces: Dict[str, ActionSpace] = {}
        self.base_prompt: str = ""

        self.state: SudokuGameState = SudokuGameState(
            terminated=False,
            success=False,
            aborted=False,
            board=self.original_board.tolist(),
            moves=0,
            warning="",
        )

    def reset(
        self,
        initial_observations: Dict[str, Observation],
        initial_action_spaces: Dict[str, ActionSpace],
    ):
        super().reset(initial_observations, initial_action_spaces)
        self.state: SudokuGameState = SudokuGameState(
            terminated=False,
            success=False,
            aborted=False,
            board=self.original_board.tolist(),
            moves=0,
            warning="",
        )

    def _is_board_valid(
        self, grid: List[List[int]], row: int, col: int, value: int
    ) -> bool:
        if grid[row][col] != 0:
            logger.warning(
                f"[_is_board_valid] Cell is already filled, row: {row}, col: {col}, value: {value}"
            )
            return False

        # Convert to numpy for easier array operations
        grid_np = np.array(grid)
        if value in grid_np[row] or value in grid_np[:, col]:
            logger.warning(
                f"[_is_board_valid] Value {value} already in row {row} or column {col}"
            )
            return False

        block_size = self.board_size
        block_row = (row // block_size) * block_size
        block_col = (col // block_size) * block_size
        if (
            value
            in grid_np[
                block_row : block_row + block_size, block_col : block_col + block_size
            ]
        ):
            logger.warning(
                f"[_is_board_valid] Value {value} already in block {block_row} {block_col}"
            )
            return False

        logger.info(f"[_is_board_valid] Board is valid")
        return True

    def _is_board_solved(self, grid: List[List[int]]) -> bool:
        if not all(0 not in row for row in grid):
            return False

        # Convert to numpy for easier array operations
        grid_np = np.array(grid)
        for i in range(9):
            for j in range(9):
                if not self._is_board_valid(grid, i, j, grid_np[i][j]):
                    return False

        return True

    def _get_current_board(self):
        return self.state["board"].copy()

    def _is_action_valid_in_state(
        self, player: SudokuPlayer, action: SudokuAction
    ) -> bool:
        """Check if an action is valid in the current state."""
        row = action.get("row")
        col = action.get("col")
        value = action.get("value")

        if row is None or col is None or value is None:
            return False

        return self._is_board_valid(self.state["board"], row, col, value)

    def _update_state_through_action(
        self, player: SudokuPlayer, action: SudokuAction
    ) -> None:
        """Update the game state based on the action."""
        row = action["row"]
        col = action["col"]
        value = action["value"]

        self.state["board"][row][col] = value

        if self._is_board_solved(self.state["board"]):
            self.state["terminated"] = True
            self.state["success"] = True
            self.state["aborted"] = False

        self.state["terminated"] = False
        self.state["success"] = True
        self.state["aborted"] = False

        logger.info(f"[_update_state_through_action] Board is valid, action: {action}")

    def update_observations(self):
        for player in self.players:
            self.observations[player.name]["board"] = self.format_board(
                np.array(self.state["board"])
            )
            if self.state["warning"] != "":
                self.observations[player.name]["content"] = self.state["warning"]
                self.state["warning"] = ""
            else:
                self.observations[player.name]["content"] = (
                    "The new board is:\n\n"
                    + self.format_board(np.array(self.state["board"]))
                    + "\n\nMake your next move in the format described before."
                )

    def format_board(self, board: np.ndarray) -> str:
        """Format the Sudoku board with box separators using | and - characters."""
        box_size = self.board_size
        board_size = box_size * 3

        output = []

        for i in range(board_size):
            row = []
            for j in range(board_size):
                val = str(board[i][j])
                row.append(val)
                if (j + 1) % box_size == 0 and j < board_size - 1:
                    row.append("|")
            output.append("".join(row))

            if (i + 1) % box_size == 0 and i < board_size - 1:
                output.append("-" * (board_size * 2 + 2))

        return "\n".join(output)
