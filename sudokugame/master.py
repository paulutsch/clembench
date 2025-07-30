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
from clemcore.utils.logger import format_json, setup_logger

from sudokugame.game_environment import SudokuAction, SudokuEnvironment, SudokuPlayer

logger = setup_logger(__name__)


class SudokuGame(EnvGameMaster):
    """Game master for the Sudoku game."""

    def __init__(
        self,
        game_spec: GameSpec,
        experiment: Dict,
        player_models: List[Model],
    ):
        logger.info(f"[_init] Initializing SudokuGame GameMaster with spec={game_spec}")
        logger.debug(f"[_init] Experiment parameters: {experiment}")

        super().__init__(game_spec, experiment, player_models)
        logger.info("[_init] SudokuGame initialization complete")

    def _on_setup(self, **game_instance):
        """
        Called during game setup. Configure game parameters and initialize players.

        Args:
            game_instance: Game instance parameters from instances.json
        """
        logger.info("[_on_setup] Setting up SudokuGame")

        logger.debug(f"[_on_setup] Game instance: {game_instance}")

        self.game_environment = SudokuEnvironment(game_instance)
        logger.info(f"[_on_setup] Game environment: {self.game_environment}")
        self.game_environment.config = game_instance

        self.player = SudokuPlayer(self.player_models[0])
        logger.debug(f"[_on_setup] Created player: {self.player}")

        self.add_player(self.player)
        logger.info(f"[_on_setup] Added player: {self.player.name}")

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

    def compute_turn_score(self, response: str, context: Dict) -> float:
        """
        Compute a score for the player's response based on the environment state.

        Args:
            response: The player's response
            context: Additional context for scoring

        Returns:
            float: The score for the response
        """
        logger.debug(
            f"[_compute_turn_score] Computing response score for response: {response}"
        )

        score = 1.0 if self.game_environment.state["success"] else 0.0
        logger.debug(f"[_compute_turn_score] Response score: {score}")
        return score

    def compute_episode_score(self) -> float:
        """
        Compute the overall episode score.

        Returns:
            float: The episode score
        """
        logger.info("[_compute_episode_score] Computing episode score")

        success = self.game_environment.state["success"]
        not_aborted = not self.game_environment.state["aborted"]

        return (not_aborted + success) / 2


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
        logger.info(f"SudokuGameBenchmark initialized with game spec: {game_spec}")

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> SudokuGame:
        logger.info(f"Creating SudokuGame master with experiment: {experiment}")
        logger.debug(
            f"Player models: {[model.__class__.__name__ for model in player_models]}"
        )
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
        logger.info(f"Creating SudokuGameScorer with experiment: {experiment}")
        logger.debug(f"Game instance for scorer: \n{format_json(game_instance)}")
        return SudokuGameScorer(self.game_name, experiment, game_instance)
