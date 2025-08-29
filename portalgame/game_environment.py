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
        self.state: PortalGameState

    def reset(self) -> None:
        """Reset the game environment."""
        super().reset()

        self._populate_portal_grid()

        self.update_observations()

        for player in self.players:
            self.set_action_space(player, ["move"])

    def _populate_portal_grid(self) -> None:
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

    def _update_state_through_action(
        self, player: Player, action: PortalAction
    ) -> None:
        """Update the game state based on the action."""
        direction = action.get("direction")
        self.move_player(player.name, direction)

        new_cell_objects = self.get_objects_at(self.get_player_position(player.name))

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
        new_cell_objects = self.get_objects_at(self.get_player_position(player.name))

        if new_cell_objects != [] and isinstance(new_cell_objects[0], Portal):
            return True, True

        return False, False

    def _is_action_valid_in_state(
        self, player: Player, action: PortalAction
    ) -> Tuple[bool, str]:
        """Check if a move is valid."""
        direction = action.get("direction")
        valid, reason = super()._is_action_valid_in_state(player, direction)
        if not valid:
            return valid, reason

        y, x = self.get_player_position(player.name)
        new_y = y - 1 if direction == "n" else y + 1 if direction == "s" else y
        new_x = x + 1 if direction == "e" else x - 1 if direction == "w" else x

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
            player_pos = self.get_player_position(player.name)
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
