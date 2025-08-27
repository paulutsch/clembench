from typing import Dict, List, Optional, Tuple

from clemcore.clemgame import (
    Action,
    ActionSpace,
    GridCell,
    GridEnvironment,
    GridState,
    Observation,
    Player,
    PlayerObject,
)

from portalgame.objects import Door, Portal, Switch, Wall


class PortalAction(Action):
    """Action for the Portal game."""

    action_type: str  # 'move
    direction: str  # 'n', 's', 'e', 'w'


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
        self,
        config: Optional[Dict] = None,
    ):
        super().__init__(config=config)
        self.observations: Dict[str, Observation] = {}
        self.action_spaces: Dict[str, ActionSpace] = {}
        self.config = config or {}
        self.explored: Dict[str, List[List[bool]]] = {}
        self.state: PortalGameState

    def reset(self) -> None:
        """Reset the game environment."""
        super().reset()

        self._construct_grid()

        self.explored = {
            player.name: [
                [False for _ in range(self.width)] for _ in range(self.height)
            ]
            for player in self.players
        }
        for player in self.players:
            self._mark_explored(
                player.name, self.state["_player_positions"][player.name]
            )

        self.update_observations()

        for player in self.players:
            self.set_action_space(player, ["move"])

    def _construct_grid(self) -> None:
        """Construct the game grid based on the config."""
        if "grid" not in self.config:
            return

        grid_config = self.config["grid"]

        for wall_pos in grid_config.get("walls", []):
            row, col = wall_pos
            self.add_object(Wall(position=(row, col)))

        portal_pos = grid_config.get("portal")
        if portal_pos:
            row, col = portal_pos
            self.add_object(Portal(position=(row, col)))

        switch_pos = grid_config.get("switch")
        if switch_pos:
            row, col = switch_pos
            self.add_object(Switch(position=(row, col)))

        door_pos = grid_config.get("door")
        if door_pos:
            row, col = door_pos
            self.add_object(Door(position=(row, col)))

        player_start = grid_config.get("player_start", (0, 0))
        self.add_object(PlayerObject(position=player_start, player=self.players[0]))

        self.state["_player_positions"][self.players[0].name] = tuple(player_start)

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

        y, x = self.state["_player_positions"][player.name]
        current_cell = self.get_objects_at((y, x))
        player_object = current_cell[-1]
        self.remove_object(player_object)

        if direction == "n":
            self.state["_player_positions"][player.name] = (y - 1, x)
        elif direction == "s":
            self.state["_player_positions"][player.name] = (y + 1, x)
        elif direction == "e":
            self.state["_player_positions"][player.name] = (y, x + 1)
        elif direction == "w":
            self.state["_player_positions"][player.name] = (y, x - 1)

        player_object.position = self.state["_player_positions"][player.name]
        self.add_object(player_object)
        self._mark_explored(player.name, self.state["_player_positions"][player.name])

        new_cell_objects = self.get_objects_at(
            self.state["_player_positions"][player.name]
        )

        if new_cell_objects != [] and isinstance(new_cell_objects[0], Switch):
            new_cell_objects[0].activated = not new_cell_objects[0].activated
            for y in self.state["_grid"]:
                for cell in y:
                    if cell["objects"] != [] and isinstance(cell["objects"][0], Door):
                        cell["objects"][0].toggle_state()

    def check_won(self, player: Player) -> Tuple[bool, bool]:
        """
        Check if the player has won.
        """
        new_cell_objects = self.get_objects_at(
            self.state["_player_positions"][player.name]
        )

        if new_cell_objects != [] and isinstance(new_cell_objects[0], Portal):
            return True, True

        return False, False

    def _is_action_valid_in_state(
        self, player: Player, action: PortalAction
    ) -> Tuple[bool, str]:
        # action_type is already checked in the base class — need to only check the direction
        direction = action.get("direction")
        """Check if a move is valid."""
        y, x = self.state["_player_positions"][player.name]
        if direction == "n":
            new_pos = (y - 1, x)
        elif direction == "s":
            new_pos = (y + 1, x)
        elif direction == "e":
            new_pos = (y, x + 1)
        elif direction == "w":
            new_pos = (y, x - 1)
        else:
            return False, f"Invalid direction: {direction}! Please try again."

        new_y, new_x = new_pos
        # check if the new position is within the grid
        if not (0 <= new_y < self.height and 0 <= new_x < self.width):
            return (
                False,
                f"The cell ({new_y}, {new_x}) is outside the grid! Please try again.",
            )

        # check if the new position is a wall or closed door
        cell = self.state["_grid"][new_y][new_x]
        if cell["objects"] != [] and isinstance(cell["objects"][0], Wall):
            return (
                False,
                f"The object at cell ({new_y}, {new_x}) is a wall! You cannot pass through walls! Please try again.",
            )
        if (
            cell["objects"] != []
            and isinstance(cell["objects"][0], Door)
            and not cell["objects"][0].is_open
        ):
            return (
                False,
                f"The object at cell ({new_y}, {new_x}) is a closed door! You need to open it first.",
            )

        return True, ""

    def update_observations(self) -> None:
        """Update the observation for all players."""
        for player in self.players:
            player_pos = self.state["_player_positions"][player.name]
            rendered_state = self.render_state(player.name)

            switch_state = None
            door_state = None
            for row in self.state["_grid"]:
                for cell in row:
                    if cell["objects"] != [] and isinstance(cell["objects"][0], Switch):
                        switch_state = cell["objects"][0].activated
                    elif cell["objects"] != [] and isinstance(cell["objects"][0], Door):
                        door_state = cell["objects"][0].is_open

            if self.state["warning"]:
                warning = "Warning: " + self.state["warning"]
            else:
                warning = ""

            if self.state["moves"] == 0:
                text_content = (
                    self.config.get("prompt", "")
                ) + "\n\nYou initially see the following grid layout:\n"
            else:
                text_content = (
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
                    + f"\nGrid (Visible Area):\n"
                )

            observation = self._create_observation(text_content, rendered_state)

            self.observations[player.name] = observation

        self.state["warning"] = ""
