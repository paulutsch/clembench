from typing import Any, Dict, Optional, Tuple

import numpy as np
from clemcore.clemgame.player import Player

from world_environments import GameEnvironment


class SudokuEnvironment(GameEnvironment):

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.grid_shape = grid_shape
        self.grid = np.zeros(grid_shape)

    def reset(self):
        super().reset()
        self.grid = np.zeros(self.grid_shape)

    def _update_state_through_action(self, player: Player, action: Dict[str, Any]):
        pass
