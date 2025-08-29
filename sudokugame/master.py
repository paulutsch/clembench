from typing import Dict, List

from clemcore.backends import Model
from clemcore.clemgame import (
    ActionSpace,
    EnvGameMaster,
    GameBenchmark,
    GameScorer,
    GameSpec,
    Observation,
    Player,
)
from clemcore.clemgame.metrics import BENCH_SCORE, METRIC_ABORTED, METRIC_SUCCESS

from sudokugame.game_environment import SudokuAction, SudokuEnvironment, SudokuPlayer


class SudokuGame(EnvGameMaster):
    """Game master for the Sudoku game."""

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
        self.game_environment = SudokuEnvironment(game_instance)

        for player in self.player_models:
            self.add_player(SudokuPlayer(player))

        self.game_environment.reset()

    def _player_response_in_expected_format(
        self, player: Player, utterance: str
    ) -> bool:
        """
        Validate the player's response.

        Args:
            player: The player making the response
            utterance: The player's response

        Returns:
            bool: Whether the response is valid
        """
        return True

    def _parse_action_from_response(self, response: str) -> SudokuAction:
        """Create an action from a player's response.

        Default: return action

        Args:
            response: The textual response from the player
            action_type: The type of action to create

        Returns:
            {"action_type": "fill_cell", "row": row, "col": col, "value": value}
        """
        row, col, value = map(int, response.strip().split())
        action: SudokuAction = {
            "action_type": "fill_cell",
            "row": row,
            "col": col,
            "value": value,
        }
        return action


class SudokuGameScorer(GameScorer):
    """Scorer for the Sudoku game."""

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_episode_scores(self, interactions: Dict) -> None:
        """Compute episode-level scores for the Sudoku game.

        Args:
            interactions: Dict containing the logged episode's interactions.
        """
        aborted = interactions.get(METRIC_ABORTED, False)
        success = interactions.get(METRIC_SUCCESS, False)

        if aborted:
            bench_score = 0.0
        else:
            bench_score = 100.0 if success else 0.0

        self.log_episode_score(BENCH_SCORE, bench_score)


class SudokuGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> SudokuGame:
        return SudokuGame(
            self.game_spec,
            experiment,
            player_models,
        )

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        """
        Create a scorer for the Sudoku Game.

        Args:
            experiment: Experiment configuration dictionary
            game_instance: Game instance dictionary with specific parameters

        Returns:
            A SudokuGameScorer instance
        """
        return SudokuGameScorer(self.game_name, experiment, game_instance)
