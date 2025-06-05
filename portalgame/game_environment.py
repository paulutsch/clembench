from typing import Dict, List, Optional, Tuple

from clemcore.clemgame import (
    Action,
    ActionSpace,
    Grid,
    GridCell,
    GridEnvironment,
    GridObservation,
    GridState,
    Player,
    PlayerObject,
)
from clemcore.utils.logger import setup_logger

from portalgame.objects import Portal, ProjectedWall, Switch, Wall

logger = setup_logger(__name__)


class PortalAction(Action):
    """Action for the Portal game."""

    action_type: str
    direction: str  # 'n', 's', 'e', 'w'


class PortalObservation(GridObservation):
    """Observation for the Portal game."""

    role: str
    content: str
    grid: str


class PortalGameState(GridState):
    """State for the Portal game."""

    moves: int
    success: bool
    terminated: bool
    aborted: bool
    warning: str


class PortalGameEnvironment(GridEnvironment):
    """Environment for the Portal game."""

    def __init__(self, grid_size: int = 10):
        super().__init__(grid_size, grid_size)
        self.grid_size = grid_size
        self.observations: Dict[str, PortalObservation] = {}
        self.action_spaces: Dict[str, ActionSpace] = {}
        self.base_prompt = ""
        self.config = {}
        self.explored: Dict[str, List[List[bool]]] = {}
        self.state: PortalGameState
        self.max_moves: int

    def reset(self) -> None:
        """Reset the game environment."""
        self.state = PortalGameState(
            grid=[
                [GridCell(object=None, position=(i, j)) for j in range(self.grid_size)]
                for i in range(self.grid_size)
            ],
            player_positions={
                self.players[0].name: self.config["grid"]["player_start"]
            },
            partial_observability=False,
            success=False,
            terminated=False,
            aborted=False,
            moves=0,
            warning="",
        )
        self.max_moves = self.config["max_moves"]

        self._construct_grid()

        self.explored = {
            player.name: [
                [False for _ in range(self.grid_size)] for _ in range(self.grid_size)
            ]
            for player in self.players
        }
        for player in self.players:
            self._mark_explored(
                player.name, self.state["player_positions"][player.name]
            )

        self.base_prompt = self.config["prompt"]

        # Initialize the environment with the grid configuration
        player_observation: PortalObservation = {
            "role": "user",
            "content": (
                self.base_prompt
                + "\n\n"
                + "You initially see the following grid layout:\n"
                + self.render_state(self.players[0].name)
            ),
            "grid": self.render_state(self.players[0].name),
        }
        initial_observations: Dict[str, PortalObservation] = {
            self.players[0].name: player_observation,
        }
        initial_action_spaces: Dict[str, ActionSpace] = {self.players[0].name: ["move"]}

        self.observations = initial_observations
        self.action_spaces = initial_action_spaces

    def _construct_grid(self) -> None:
        """Construct the game grid based on the config."""
        if "grid" not in self.config:
            logger.warning("No grid configuration found, using default grid")
            return

        grid_config = self.config["grid"]

        for wall_pos in grid_config.get("walls", []):
            row, col = wall_pos
            self.state["grid"][row][col] = GridCell(
                object=Wall(position=(row, col)), position=(row, col)
            )

        portal_pos = grid_config.get("portal")
        if portal_pos:
            row, col = portal_pos
            self.state["grid"][row][col] = GridCell(
                object=Portal(position=(row, col)), position=(row, col)
            )

        switch_pos = grid_config.get("switch")
        if switch_pos:
            row, col = switch_pos
            self.state["grid"][row][col] = GridCell(
                object=Switch(position=(row, col)), position=(row, col)
            )

        projected_wall_pos = grid_config.get("projected_wall")
        if projected_wall_pos:
            row, col = projected_wall_pos
            self.state["grid"][row][col] = GridCell(
                object=ProjectedWall(position=(row, col)), position=(row, col)
            )

        player_start = grid_config.get("player_start", (0, 0))

        self.state["player_positions"][self.players[0].name] = tuple(player_start)

    def _is_valid_move(self, pos: Tuple[int, int], direction: str) -> bool:
        """Check if a move is valid."""
        row, col = pos
        if direction == "n":
            new_pos = (row - 1, col)
        elif direction == "s":
            new_pos = (row + 1, col)
        elif direction == "e":
            new_pos = (row, col + 1)
        elif direction == "w":
            new_pos = (row, col - 1)
        else:
            return False

        new_row, new_col = new_pos
        # check if the new position is within the grid
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            self.state["warning"] = (
                "You cannot move outside the grid! Please try again."
            )
            return False

        # check if the new position is a wall
        cell = self.state["grid"][new_row][new_col]
        if isinstance(cell["object"], Wall):
            self.state["warning"] = "You cannot pass through walls! Please try again."
            return False

        return True

    def _mark_explored(self, player_name: str, pos: Tuple[int, int]) -> None:
        """Mark cells around a position as explored for the given player."""
        row, col = pos
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                    self.explored[player_name][i][j] = True

    def _update_state_through_action(
        self, player: Player, action: PortalAction
    ) -> None:
        """Update the game state based on the action."""
        direction = action.get("direction")

        row, col = self.state["player_positions"][player.name]
        if direction == "n":
            self.state["player_positions"][player.name] = (row - 1, col)
        elif direction == "s":
            self.state["player_positions"][player.name] = (row + 1, col)
        elif direction == "e":
            self.state["player_positions"][player.name] = (row, col + 1)
        elif direction == "w":
            self.state["player_positions"][player.name] = (row, col - 1)

        self._mark_explored(player.name, self.state["player_positions"][player.name])

        current_cell = self.state["grid"][
            self.state["player_positions"][player.name][0]
        ][self.state["player_positions"][player.name][1]]

        if isinstance(current_cell["object"], Portal):
            self.state["terminated"] = True
            self.state["success"] = True
            self.state["aborted"] = False
            return

        self.state["aborted"] = False
        self.state["terminated"] = False
        self.state["success"] = True

        if isinstance(current_cell["object"], Switch):
            current_cell["object"].activated = not current_cell["object"].activated
            for row in self.state["grid"]:
                for cell in row:
                    if isinstance(cell["object"], ProjectedWall):
                        cell["object"].toggle_visibility()

    def render_state(self, player_name: Optional[str] = None) -> str:
        """Format the grid for display.

        Args:
            grid: The grid to format
            player_name: Optional player name. If provided, uses the explored map of that player
                to render explored vs unexplored cells and marks the player's current position with 'P'.
                If None, shows the entire grid without fog of war.
        """
        grid_str = ""
        player_pos = None
        explored = None
        if player_name is not None:
            player_pos = self.state["player_positions"][player_name]
            explored = self.explored[player_name]

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.state["grid"][i][j]
                if explored is not None:
                    if explored[i][j]:
                        if (i, j) == player_pos:
                            grid_str += "👤"
                        else:
                            grid_str += (
                                cell["object"].symbol
                                if cell["object"] is not None
                                else "⬜"
                            )
                    else:
                        grid_str += "⬜"
                else:
                    if (player_pos is not None) and (i, j) == player_pos:
                        grid_str += "👤"
                    elif cell["object"] is not None:
                        grid_str += cell["object"].symbol
                    else:
                        grid_str += "⬜"

            grid_str += "\n"
            if i < self.grid_size - 1:
                grid_str += "\n"
        return grid_str

    def _is_action_valid_in_state(self, player: Player, action: PortalAction) -> bool:
        # action_type is already checked in the base class — need to only check the direction
        direction = action.get("direction")
        if not direction or not self._is_valid_move(
            self.state["player_positions"][player.name], direction
        ):
            return False

        return True

    def update_observations(self) -> None:
        """Update the observation for all players."""
        for player in self.players:
            player_pos = self.state["player_positions"][player.name]
            grid_str = self.render_state(player.name)

            switch_state = False
            projected_wall_state = True
            for row in self.state["grid"]:
                for cell in row:
                    if isinstance(cell["object"], Switch):
                        switch_state = cell["object"].activated
                    elif isinstance(cell["object"], ProjectedWall):
                        projected_wall_state = cell["object"].is_visible

            if self.state["warning"]:
                warning = "Warning: " + self.state["warning"]
            else:
                warning = ""

            observation: PortalObservation = {
                "role": "user",
                "content": (
                    f"{warning}\n"
                    f"Current position: {player_pos}\n"
                    f"Switch active: {switch_state}\n"
                    f"Projected wall active: {projected_wall_state}\n"
                    f"Visible grid:\n{grid_str}"
                ),
                "grid": grid_str,
            }
            self.state["warning"] = ""

            self.observations[player.name] = observation
