from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from clemcore.clemgame import (
    Action,
    ActionSpace,
    GameEnvironment,
    GameState,
    Observation,
)
from clemcore.utils.logger import setup_logger
from sudoku import Sudoku

logger = setup_logger(__name__)


class SudokuAction(Action):
    """Action for the Sudoku game."""

    row: int
    col: int
    value: int


class SudokuObservation(Observation):
    """Observation for the Sudoku game."""

    board: np.ndarray


class SudokuGameState(GameState):
    board: np.ndarray
    puzzle: Sudoku


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

        self.original_puzzle = Sudoku(self.board_size).difficulty(self.difficulty)
        self.original_board = np.array(self.original_puzzle.board)

        self.state: SudokuGameState = SudokuGameState(
            terminated=False,
            success=False,
            aborted=False,
            puzzle=self.original_puzzle,
            board=self.original_board,
        )

    def reset(
        self,
        initial_observations: Dict[str, Observation],
        initial_action_spaces: Dict[str, ActionSpace],
    ):
        super().reset(initial_observations, initial_action_spaces)
        self.puzzle = Sudoku(self.board_size).difficulty(self.difficulty)
        self.state: SudokuGameState = SudokuGameState(
            terminated=False,
            success=False,
            aborted=False,
            puzzle=self.original_puzzle,
            board=self.original_board,
        )

    def _is_board_valid(self, grid: np.ndarray, row: int, col: int, value: int) -> bool:
        if value in grid[row] or value in grid[:, col]:
            return False

        block_row = (row // 3) * 3
        block_col = (col // 3) * 3
        if value in grid[block_row : block_row + 3, block_col : block_col + 3]:
            return False

        return True

    def _is_board_solved(self, grid: np.ndarray) -> bool:
        # check if the board is filled completely
        is_filled = np.all(grid != 0)

        if is_filled and self.puzzle.solve(grid.tolist()):
            return True

        return False

    def _get_current_board(self):
        return self.state["board"].copy()

    def _do_update_state(self, action: SudokuAction) -> None:
        """Update the game state based on the action."""
        row = action.get("row")
        col = action.get("col")
        value = action.get("value")

        if row is None or col is None or value is None:
            logger.warning("row or col or value missing in action")
            self.state["terminated"] = True
            self.state["success"] = False
            return

        if self.state["board"][row][col] is not None:
            logger.warning("cell is not None")
            self.state["terminated"] = True
            self.state["success"] = False
            return

        self.state["board"][row][col] = value

        if not self._is_board_valid(self.state["board"], row, col, value):
            self.state["terminated"] = True
            self.state["success"] = False
            return

        if self._is_board_solved(self.state["board"]):
            self.state["terminated"] = True
            self.state["success"] = True
