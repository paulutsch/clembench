from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from clemcore.clemgame.player import Player

from ..world_environments.game_environment import GameEnvironment


class SudokuEnvironment(GameEnvironment):

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        # action_spaces: Optional[Dict[str, List[Any]]] = None,
        # observation_spaces: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        action_spaces = {"player_one": ["write_number"]}
        observation_spaces

        super().__init__(action_spaces, observation_spaces, config)

        self.grid_shape = grid_shape
        self.grid = np.zeros(grid_shape)

    def reset(self):
        super().reset()
        self.grid = np.zeros(self.grid_shape)

    def _update_state_through_action(self, player: Player, action: Dict[str, Any]):
        pass


if __name__ == "__main__":
    grid_world = SudokuEnvironment((3, 3))
