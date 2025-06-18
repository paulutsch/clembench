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

from portalgame.objects import Door, Portal, Switch, Wall

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

    def __init__(
        self, height: int = 10, width: int = 10, limited_visibility: bool = True
    ):
        super().__init__(height, width)
        self.limited_visibility = limited_visibility
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
                [GridCell(object=None, position=(i, j)) for j in range(self.width)]
                for i in range(self.height)
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
                [False for _ in range(self.width)] for _ in range(self.height)
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

        door_pos = grid_config.get("door")
        if door_pos:
            row, col = door_pos
            self.state["grid"][row][col] = GridCell(
                object=Door(position=(row, col)), position=(row, col)
            )

        player_start = grid_config.get("player_start", (0, 0))

        self.state["player_positions"][self.players[0].name] = tuple(player_start)

    def _mark_explored(self, player_name: str, pos: Tuple[int, int]) -> None:
        """Mark cells around a position as explored for the given player."""
        row, col = pos
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < self.height and 0 <= j < self.width:
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
                    if isinstance(cell["object"], Door):
                        cell["object"].toggle_state()

    def _is_action_valid_in_state(
        self, player: Player, action: PortalAction
    ) -> Tuple[bool, str]:
        # action_type is already checked in the base class — need to only check the direction
        direction = action.get("direction")
        """Check if a move is valid."""
        row, col = self.state["player_positions"][player.name]
        if direction == "n":
            new_pos = (row - 1, col)
        elif direction == "s":
            new_pos = (row + 1, col)
        elif direction == "e":
            new_pos = (row, col + 1)
        elif direction == "w":
            new_pos = (row, col - 1)
        else:
            return False, f"Invalid direction: {direction}! Please try again."

        new_row, new_col = new_pos
        # check if the new position is within the grid
        if not (0 <= new_row < self.height and 0 <= new_col < self.width):
            return (
                False,
                f"The cell ({new_row}, {new_col}) is outside the grid! Please try again.",
            )

        # check if the new position is a wall or closed door
        cell = self.state["grid"][new_row][new_col]
        if isinstance(cell["object"], Wall):
            return (
                False,
                f"The object at cell ({new_row}, {new_col}) is a wall! You cannot pass through walls! Please try again.",
            )
        if isinstance(cell["object"], Door) and not cell["object"].is_open:
            return (
                False,
                f"The object at cell ({new_row}, {new_col}) is a closed door! You need to open it first.",
            )

        return True, ""

    def update_observations(self) -> None:
        """Update the observation for all players."""
        for player in self.players:
            player_pos = self.state["player_positions"][player.name]
            grid_str = self.render_state(player.name)

            switch_state = None
            door_state = None
            for row in self.state["grid"]:
                for cell in row:
                    if isinstance(cell["object"], Switch):
                        switch_state = cell["object"].activated
                    elif isinstance(cell["object"], Door):
                        door_state = cell["object"].is_open

            if self.state["warning"]:
                warning = "Warning: " + self.state["warning"]
            else:
                warning = ""

            observation: PortalObservation = {
                "role": "user",
                "content": (
                    (f"{warning}\n" if warning else "")
                    + f"Current position: {player_pos}\n"
                    + (
                        f"Switch active: {switch_state}\n"
                        if switch_state is not None
                        else ""
                    )
                    + (
                        f"Door state: {'open' if door_state else 'closed'}\n"
                        if door_state is not None
                        else ""
                    )
                    + f"\nGrid (Visible Area):\n{grid_str}"
                ),
                "grid": grid_str,
            }
            self.state["warning"] = ""

            self.observations[player.name] = observation
