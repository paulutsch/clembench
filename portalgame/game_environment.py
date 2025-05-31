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
    direction: Optional[str]  # 'n', 's', 'e', 'w'
    target: Optional[Tuple[int, int]]  # for inspect/use actions


class PortalObservation(GridObservation):
    """Observation for the Portal game."""

    role: str
    content: str
    grid: str


class PortalGameState(GridState):
    """State for the Portal game."""

    # actually not needed yet, but will keep it in case it'll be needed later

    moves: int
    success: bool
    terminated: bool
    aborted: bool


class PortalGameEnvironment(GridEnvironment):
    """Environment for the Portal game."""

    def __init__(self, grid_size: int = 10):
        super().__init__(grid_size, grid_size)
        self.grid_size = grid_size
        self.observations: Dict[str, PortalObservation] = {}
        self.action_spaces: Dict[str, ActionSpace] = {}
        self.base_prompt = ""
        self.config = {}

    def reset(
        self,
        # initial_observations: Dict[str, PortalObservation],
        # initial_action_spaces: Dict[str, ActionSpace],
    ) -> None:
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
        )

        self._construct_grid()

        self.base_prompt = self.config["prompt"]

        # Initialize the environment with the grid configuration
        player_observation: PortalObservation = {
            "role": "user",
            "content": (
                self.base_prompt
                + "\n\n"
                + "You initially see the following grid layout:\n"
                + self.format_grid(
                    self.state["grid"],
                    self.state["player_positions"][self.players[0].name],
                )
            ),
            "grid": self.format_grid(
                self.state["grid"],
                self.state["player_positions"][self.players[0].name],
            ),
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
        if player_start:
            row, col = player_start
            self.state["grid"][row][col] = GridCell(
                object=PlayerObject(position=(row, col), player=self.players[0]),
                position=(row, col),
            )
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
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            return False

        cell = self.state["grid"][new_row][new_col]
        if cell["object"] == Wall:
            return False
        if isinstance(cell["object"], ProjectedWall) and cell["object"].is_visible:
            return False

        return True

    def _do_update_state(self, player: Player, action: PortalAction) -> None:
        """Update the game state based on the action."""
        action_type = action.get("action_type")

        if action_type == "move":
            direction = action.get("direction")
            if not direction or not self._is_valid_move(
                self.state["player_positions"][player.name], direction
            ):
                self.state["terminated"] = True
                self.state["success"] = False
                self.state["aborted"] = True
                return

            row, col = self.state["player_positions"][player.name]
            if direction == "n":
                self.state["player_positions"][player.name] = (row - 1, col)
            elif direction == "s":
                self.state["player_positions"][player.name] = (row + 1, col)
            elif direction == "e":
                self.state["player_positions"][player.name] = (row, col + 1)
            elif direction == "w":
                self.state["player_positions"][player.name] = (row, col - 1)

            current_cell = self.state["grid"][
                self.state["player_positions"][player.name][0]
            ][self.state["player_positions"][player.name][1]]

            if isinstance(current_cell["object"], Portal):
                self.state["terminated"] = True
                self.state["success"] = True
                self.state["aborted"] = False
                return

            if isinstance(current_cell["object"], Switch):
                current_cell["object"].activated = not current_cell["object"].activated
                for row in self.state["grid"]:
                    for cell in row:
                        if isinstance(cell["object"], ProjectedWall):
                            cell["object"].toggle_visibility()

            self.state["moves"] += 1

    def format_grid(
        self, grid: List[List[GridCell]], player_pos: Optional[Tuple[int, int]] = None
    ) -> str:
        """Format the grid for display.

        Args:
            grid: The grid to format
            player_pos: Optional player position. If provided, shows the full grid but marks unexplored areas as "■".
                       If None, shows the entire grid without fog of war.
        """
        grid_str = ""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = grid[i][j]
                if player_pos is not None:
                    row, col = player_pos
                    is_visible = abs(i - row) <= 1 and abs(j - col) <= 1

                    if is_visible:
                        if cell["object"] is not None:
                            grid_str += cell["object"].symbol
                        else:
                            grid_str += "▢"
                    else:
                        grid_str += "■"
                else:
                    if cell["object"] is not None:
                        grid_str += cell["object"].symbol
                    else:
                        grid_str += "▢"

                if j < self.grid_size - 1:
                    grid_str += "|"
            grid_str += "\n"
            if i < self.grid_size - 1:
                grid_str += "-" * (self.grid_size * 2 - 1) + "\n"
        return grid_str

    def update_observations(self) -> None:
        """Update the observation for all players."""
        for player in self.players:
            player_pos = self.state["player_positions"][player.name]
            grid_str = self.format_grid(self.state["grid"], player_pos)

            switch_state = False
            projected_wall_state = True
            for row in self.state["grid"]:
                for cell in row:
                    if isinstance(cell["object"], Switch):
                        switch_state = cell["object"].activated
                    elif isinstance(cell["object"], ProjectedWall):
                        projected_wall_state = cell["object"].is_visible

            observation: PortalObservation = {
                "role": "user",
                "content": (
                    f"Current position: {player_pos}\n"
                    f"Switch active: {switch_state}\n"
                    f"Projected wall active: {projected_wall_state}\n"
                    f"Visible grid:\n{grid_str}"
                ),
                "grid": grid_str,
            }

            self.observations[player.name] = observation
