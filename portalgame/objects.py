from typing import Tuple

from clemcore.clemgame import Object, PlayerObject


class Wall(Object):
    """A regular wall that blocks movement."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Wall", "W")

    def can_interact_with(self, other: Object) -> bool:
        return False

    def interact_with(self, other: Object) -> None:
        pass


class Portal(Object):
    """A portal that leads to transcendence."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Portal", "O")

    def can_interact_with(self, other: Object) -> bool:
        return isinstance(other, PlayerObject)

    def interact_with(self, other: Object) -> None:
        pass


class ProjectedWall(Object):
    """A wall that exists only in description but not in reality."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "ProjectedWall", "X")
        self.is_visible = True

    def can_interact_with(self, other: Object) -> bool:
        return False

    def interact_with(self, other: Object) -> None:
        pass

    def toggle_visibility(self) -> None:
        self.is_visible = not self.is_visible


class Switch(Object):
    """A switch that can be activated by stepping on it."""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Switch", "S")
        self.activated = False

    def can_interact_with(self, other: Object) -> bool:
        return isinstance(other, PlayerObject)

    def interact_with(self, other: Object) -> None:
        if isinstance(other, PlayerObject):
            self.activated = True
