from typing import Tuple

from clemcore.clemgame import Object


class Wall(Object):
    """A regular wall that blocks movement."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Wall", "W", "⬛️")


class Portal(Object):
    """A portal that leads to transcendence."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Portal", "O", "🌀")


class Door(Object):
    """A door that can be opened and closed."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Door", "D", "🚪")
        self.is_open = False

    def toggle_state(self) -> None:
        """Toggle the door's open/closed state."""
        self.is_open = not self.is_open


class Switch(Object):
    """A switch that can be activated by stepping on it."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Switch", "S", "🔘")
        self.activated = False
