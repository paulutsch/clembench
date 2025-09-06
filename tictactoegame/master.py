from typing import Dict, List, Optional, Tuple

import numpy as np
from clemcore.backends import Model
from clemcore.clemgame import (
    ActionSpace,
    EnvGameMaster,
    GameBenchmark,
    GameScorer,
    GameSpec,
    Observation,
)
from clemcore.clemgame.metrics import BENCH_SCORE, METRIC_ABORTED, METRIC_SUCCESS

from tictactoegame.game_environment import (
    TicTacToeAction,
    TicTacToeEnvironment,
    TicTacToePlayer,
)


class TicTacToeGame(EnvGameMaster):
    """Game master for the TicTacToe game."""

    def __init__(
        self,
        game_spec: GameSpec,
        experiment: Dict,
        player_models: List[Model],
    ):
        super().__init__(game_spec, experiment, player_models)

    def _on_setup(self, **game_instance):
        """
        Called during game setup. Configure game parameters and initialize players.

        Args:
            game_instance: Game instance parameters from instances.json
        """
        self.game_environment = TicTacToeEnvironment(game_instance)

        for player in self.player_models:
            self.add_player(TicTacToePlayer(player))

        self.game_environment.reset()

    def _response_valid(
        self, player: TicTacToePlayer, utterance: str
    ) -> bool:
        """
        Validate the player's response.

        Args:
            player: The player making the response
            utterance: The player's response

        Returns:
            bool: Whether the response is valid
        """
        try:
            row, col = map(int, utterance.strip().split())
            return 0 <= row < 3 and 0 <= col < 3
        except (ValueError, IndexError):
            return False

    def _parse_action_from_response(self, response: str) -> TicTacToeAction:
        """Create an action from a player's response.

        Args:
            response: The textual response from the player

        Returns:
            TicTacToeAction: The parsed action
        """
        row, col = map(int, response.strip().split())
        action: TicTacToeAction = {
            "action_type": "make_move",
            "row": row,
            "col": col,
        }
        return action


class TicTacToeGameScorer(GameScorer):
    """Scorer for the TicTacToe game."""

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_episode_scores(self, interactions: Dict) -> None:
        """Compute episode-level scores for the TicTacToe game.

        Args:
            interactions: Dict containing the logged episode's interactions.
        """
        aborted = interactions.get(METRIC_ABORTED, False)
        success = interactions.get(METRIC_SUCCESS, False)

        if aborted:
            bench_score = np.nan
        else:
            bench_score = 100.0 if success else 0.0

        self.log_episode_score(BENCH_SCORE, bench_score)


class TicTacToeGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> TicTacToeGame:
        return TicTacToeGame(
            self.game_spec,
            experiment,
            player_models,
        )

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        """
        Create a scorer for the TicTacToe Game.

        Args:
            experiment: Experiment configuration dictionary
            game_instance: Game instance dictionary with specific parameters

        Returns:
            A TicTacToeGameScorer instance
        """
        return TicTacToeGameScorer(self.game_name, experiment, game_instance)
