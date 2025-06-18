from typing import Tuple

from clemcore.clemgame import Object, PlayerObject


class Wall(Object):
    """A regular wall that blocks movement."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Wall", "wall")

    def can_interact_with(self, other: Object) -> bool:
        return False

    def interact_with(self, other: Object) -> None:
        pass


class Portal(Object):
    """A portal that leads to transcendence."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Portal", "portal")

    def can_interact_with(self, other: Object) -> bool:
        return isinstance(other, PlayerObject)

    def interact_with(self, other: Object) -> None:
        pass


class Door(Object):
    """A door that can be opened and closed."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Door", "door")
        self.is_open = False

    def can_interact_with(self, other: Object) -> bool:
        return isinstance(other, PlayerObject)

    def interact_with(self, other: Object) -> None:
        pass

    def toggle_state(self) -> None:
        """Toggle the door's open/closed state."""
        self.is_open = not self.is_open


class Switch(Object):
    """A switch that can be activated by stepping on it."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Switch", "switch")
        self.activated = False

    def can_interact_with(self, other: Object) -> bool:
        return isinstance(other, PlayerObject)

    def interact_with(self, other: Object) -> None:
        if isinstance(other, PlayerObject):
            self.activated = True
