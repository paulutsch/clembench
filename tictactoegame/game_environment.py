from typing import Dict, List, Literal, Optional

import numpy as np
from clemcore.clemgame import (
    Action,
    ActionSpace,
    GridEnvironment,
    Object,
    Observation,
    Player,
)


class TicTacToeAction(Action):
    action_type: str
    row: int
    col: int


class TicTacToePlayer(Player):
    """Player for the TicTacToe game."""

    def __init__(self, model):
        super().__init__(model)
        self.symbol: Optional[Literal["X", "O"]] = None

    def _custom_response(self, context: Dict) -> str:
        return "Hello, world!"


class TicTacToeCell(Object):
    """Represents a cell in the TicTacToe grid."""

    def __init__(self, position: tuple[int, int], value: str = " "):
        symbol = "X" if value == "X" else "O" if value == "O" else "empty"
        pretty_symbol = "❎" if value == "X" else "🅾️" if value == "O" else "⬜️"
        super().__init__(
            position, f"cell_{position[0]}_{position[1]}", symbol, pretty_symbol
        )


class TicTacToeEnvironment(GridEnvironment):
    """Environment for the TicTacToe game."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config=config)

        self.players: List[TicTacToePlayer] = []

    def reset(self) -> None:
        """Reset the game environment."""
        super().reset()
        self._initialize_grid()

        self.update_observations()

        for player in self.players:
            self.set_action_space(player, ["make_move"])

    def _initialize_grid(self):
        """Initialize the grid with empty TicTacToe cells."""
        for i in range(3):
            for j in range(3):
                cell = TicTacToeCell((i, j), "")
                self.add_object(cell)

    def _get_board_from_grid(self) -> List[List[str]]:
        """Extract board state from grid objects."""
        board = [["empty" for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                objects = self.get_objects_at((i, j))
                if objects:
                    cell = objects[0]
                    if isinstance(cell, TicTacToeCell):
                        board[i][j] = cell.symbol

        return board

    def format_board(self, board: List[List[str]]) -> str:
        """Format the board for display with ASCII grid."""
        board_str = ""
        for i in range(3):
            for j in range(3):
                board_str += f"{board[i][j]}"
            board_str += "\n"
        return board_str

    def _is_action_valid_in_state(
        self, player: TicTacToePlayer, action: TicTacToeAction
    ) -> tuple[bool, str]:
        """Check if an action is valid in the current state."""
        row = action.get("row")
        col = action.get("col")

        if row is None or col is None:
            return False, "Missing row or col in action"

        if not (0 <= row < 3 and 0 <= col < 3):
            return False, f"Position ({row}, {col}) is out of bounds"

        board = self._get_board_from_grid()
        if board[row][col] != "empty":
            return False, f"Position ({row}, {col}) is already occupied"

        return True, ""

    def is_valid_move(self, row: int | None, col: int | None) -> bool:
        """Check if a move is valid."""

        if row is None or col is None:
            return False

        row_is_valid = 0 <= row < 3
        col_is_valid = 0 <= col < 3

        board = self._get_board_from_grid()
        cell_is_empty = board[row][col] == "empty"

        return row_is_valid and col_is_valid and cell_is_empty

    def check_won(self, player: Player) -> tuple[bool, bool]:
        """Check if the game is over (win or draw)."""
        board = self._get_board_from_grid()
        board_np = np.array(board)

        for i in range(3):
            if all(board_np[i, :] == "X") or all(board_np[:, i] == "X"):
                return True, True
            if all(board_np[i, :] == "O") or all(board_np[:, i] == "O"):
                return True, True

        if all(np.diag(board_np) == "X") or all(np.diag(np.fliplr(board_np)) == "X"):
            return True, True
        if all(np.diag(board_np) == "O") or all(np.diag(np.fliplr(board_np)) == "O"):
            return True, True

        if np.all(board_np != "empty"):
            return True, False

        return False, False

    def update_observations(self) -> None:
        """Update the observation for all players."""
        for player in self.players:
            rendered_state = self.render_state()
            board = self._get_board_from_grid()
            non_empty_cells = sum(1 for row in board for cell in row if cell != "empty")

            if self.state["warning"]:
                warning = "Warning: " + self.state["warning"]
            else:
                warning = ""

            text_content = (
                f"{warning}\n"
                if warning
                else (
                    (
                        self.config.get("prompt", "")
                        + "\n\n"
                        + f"You are the player that plays {player.symbol}.\n\n"
                    )
                    if non_empty_cells < 2
                    else ""
                )
            ) + "The board is:\n\n"

            observation = self._create_observation(text_content, rendered_state)
            self.observations[player.name] = observation

        self.state["warning"] = ""

    def _update_state_through_action(
        self, player: TicTacToePlayer, action: TicTacToeAction
    ) -> None:
        """Update the game state based on the action."""
        row = action["row"]
        col = action["col"]

        objects = self.get_objects_at((row, col))
        if objects:
            self.remove_object(objects[0])

        # count non-empty cells to determine next player
        board = self._get_board_from_grid()
        non_empty = sum(1 for r in board for c in r if c != "empty")
        next_symbol = "X" if non_empty % 2 == 0 else "O"

        new_cell = TicTacToeCell((row, col), value=next_symbol)
        self.add_object(new_cell)
