import random
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

    def _custom_response(self, context: Dict) -> str:
        i = random.randint(0, 2)
        j = random.randint(0, 2)

        return f"{i} {j}"


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

    def __init__(self, config: Dict):
        super().__init__(config=config)

    def reset(self) -> None:
        """Reset the game environment."""
        super().reset()

        self._update_observations()

        for player in self.players:
            self._set_action_space(player, ["make_move"])

    def _action_valid_in_state(
        self, player: TicTacToePlayer, action: TicTacToeAction
    ) -> tuple[bool, str]:
        """Check if an action is valid in the current state."""
        row = action.get("row")
        col = action.get("col")

        if row is None or col is None:
            return False, "Missing row or col in action"

        if not (0 <= row < 3 and 0 <= col < 3):
            return False, f"Position ({row}, {col}) is out of bounds"

        if self._get_objects_at((row, col)) != []:
            return False, f"Position ({row}, {col}) is already occupied"

        return True, ""

    def _check_won(self, player: Player) -> tuple[bool, bool]:
        """Check if the game is over (win or draw)."""
        board_np = np.array([["empty" for _ in range(3)] for _ in range(3)])
        for i in range(self.height):
            for j in range(self.width):
                if self._get_objects_at((i, j)) != []:
                    symbol = self._get_objects_at((i, j))[0].symbol
                    board_np[i, j] = symbol

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

    def _get_current_symbol(self, i: int = 0) -> str:
        """Get the current symbol of the player."""
        symbol_count = sum(
            1 for r in self.state["_grid"] for c in r if c["objects"] != []
        )
        return "X" if (symbol_count + i) % 2 == 0 else "O"

    def _update_observations(self) -> None:
        """Update the observation for all players."""
        for i, player in enumerate(self.players):
            rendered_state = self._render_state()
            non_empty_cells = sum(
                1
                for row in self.state["_grid"]
                for cell in row
                if cell["objects"] != []
            )

            if self.state["_warning"]:
                warning = "Warning: " + self.state["_warning"]
            else:
                warning = ""

            current_symbol = self._get_current_symbol(i)

            text_content = (
                f"{warning}\n"
                if warning
                else (
                    (
                        self.config.get("prompt", "")
                        + "\n\n"
                        + f"You are the player that plays {current_symbol}.\n\n"
                    )
                    if non_empty_cells < 2
                    else ""
                )
            ) + "The board is:\n\n"

            observation = self._create_observation(text_content, rendered_state)
            self.observations[player.name] = observation

        self.state["_warning"] = ""

    def _update_state_through_action(
        self, player: TicTacToePlayer, action: TicTacToeAction
    ) -> None:
        """Update the game state based on the action."""
        row = action["row"]
        col = action["col"]

        objects = self._get_objects_at((row, col))
        if objects:
            self._remove_object(objects[0])

        current_symbol = self._get_current_symbol()
        new_cell = TicTacToeCell((row, col), value=current_symbol)
        self._add_object(new_cell)
