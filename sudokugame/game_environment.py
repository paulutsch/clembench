import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from clemcore.backends import Model
from clemcore.clemgame import (
    Action,
    ActionSpace,
    GridEnvironment,
    Object,
    Observation,
    Player,
)
from clemcore.clemgame.grid_environment import Position
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


class SudokuObject(Object):
    """Represents a cell in the Sudoku grid."""

    def __init__(self, position: Position, value: int):
        symbol = str(value)
        emoji_numbers = ["0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]
        pretty_symbol = emoji_numbers[value] if 0 <= value <= 9 else str(value)
        super().__init__(
            position, f"cell_{position[0]}_{position[1]}", symbol, pretty_symbol
        )


class SudokuEnvironment(GridEnvironment):

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config=config)
        self.grid_size = self.config.get("width", 9)

        puzzle = Sudoku(self.grid_size // 3).difficulty(
            self.config.get("difficulty", 0.5)
        )
        original_grid = puzzle.board
        self.original_grid = [
            [0 if cell is None else cell for cell in row] for row in original_grid
        ]
        logger.info(f"[_initialize_grid] Original grid: {self.original_grid}")

        self.observations: Dict[str, Observation] = {}
        self.action_spaces: Dict[str, ActionSpace] = {}

        self._initialize_grid()

    def _initialize_grid(self):
        """Initialize the grid with Sudoku objects."""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                value = self.original_grid[i][j]
                object = SudokuObject((i, j), value)
                self.add_object(object)

    def reset(self):
        super().reset()

        self._initialize_grid()
        self.update_observations()

        for player in self.players:
            self.set_action_space(player, ["fill_cell"])

    def _get_board_from_grid(self) -> List[List[int]]:
        """Extract grid state."""
        grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                objects = self.get_objects_at((i, j))
                if objects:
                    object = objects[0]
                    if isinstance(object, SudokuObject):
                        grid[i][j] = int(object.symbol)
        return grid

    def _is_grid_valid(
        self, grid: List[List[int]], row: int, col: int, value: int
    ) -> bool:
        if grid[row][col] != 0:
            logger.warning(
                f"[_is_grid_valid] Cell is already filled, row: {row}, col: {col}, value: {value}"
            )
            return False

        # Convert to numpy for easier array operations
        grid_np = np.array(grid)
        if value in grid_np[row] or value in grid_np[:, col]:
            logger.warning(
                f"[_is_grid_valid] Value {value} already in row {row} or column {col}"
            )
            return False

        block_size = 3
        block_row = (row // block_size) * block_size
        block_col = (col // block_size) * block_size
        if (
            value
            in grid_np[
                block_row : block_row + block_size, block_col : block_col + block_size
            ]
        ):
            logger.warning(
                f"[_is_grid_valid] Value {value} already in block {block_row} {block_col}"
            )
            return False

        logger.info(f"[_is_grid_valid] Grid is valid")
        return True

    def _is_grid_solved(self, grid: List[List[int]]) -> bool:
        if not all(0 not in row for row in grid):
            return False

        # Convert to numpy for easier array operations
        grid_np = np.array(grid)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if not self._is_grid_valid(grid, i, j, grid_np[i][j]):
                    return False

        return True

    def _is_action_valid_in_state(
        self, player: SudokuPlayer, action: SudokuAction
    ) -> tuple[bool, str]:
        """Check if an action is valid in the current state."""
        row = action.get("row")
        col = action.get("col")
        value = action.get("value")

        if row is None or col is None or value is None:
            return False, "Missing row, col, or value in action"

        board = self._get_board_from_grid()
        if self._is_grid_valid(board, row, col, value):
            return True, ""
        else:
            return (
                False,
                f"Invalid move: cannot place {value} at position ({row}, {col})",
            )

    def _update_state_through_action(
        self, player: SudokuPlayer, action: SudokuAction
    ) -> None:
        """Update the game state based on the action."""
        row = action["row"]
        col = action["col"]
        value = action["value"]

        objects = self.get_objects_at((row, col))
        if objects:
            self.remove_object(objects[0])

        new_cell = SudokuObject((row, col), value)
        self.add_object(new_cell)

        board = self._get_board_from_grid()
        if self._is_grid_solved(board):
            self.state["terminated"] = True
            self.state["success"] = True
            self.state["aborted"] = False
        else:
            self.state["terminated"] = False
            self.state["success"] = False
            self.state["aborted"] = False

        logger.info(f"[_update_state_through_action] Board updated, action: {action}")

    def update_observations(self) -> None:
        """Update the observation for all players."""
        for player in self.players:
            rendered_state = self.render_state()

            if self.state["warning"]:
                warning = "Warning: " + self.state["warning"]
            else:
                warning = ""

            text_content = (
                self.config.get("prompt", "") + "\n\n"
                if self.state["moves"] == 0
                else "" + (f"{warning}\n" if warning else "") + "The board is:\n\n"
            )

            observation = self._create_observation(text_content, rendered_state)

            self.state["warning"] = ""

            self.observations[player.name] = observation

    def _render_state_as_string(self, player_name: str | None = None) -> str:
        """Render state as string."""
        board = self._get_board_from_grid()
        board_size = self.grid_size
        box_size = int(self.grid_size / 3)

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

    def _render_state_as_image(self, player_name: str | None = None) -> bytes:
        """Render state as image (not implemented for Sudoku)."""
        return b""

    def _render_state_as_human_readable(self, player_name: str | None = None) -> str:
        """Render state as human readable."""
        return self._render_state_as_string()
