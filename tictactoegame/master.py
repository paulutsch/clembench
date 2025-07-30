from typing import Dict, List, Optional, Tuple

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
from clemcore.utils.logger import format_json, setup_logger

from tictactoegame.game_environment import (
    TicTacToeAction,
    TicTacToeEnvironment,
    TicTacToePlayer,
)

logger = setup_logger(__name__)


class TicTacToeGame(EnvGameMaster):
    """Game master for the TicTacToe game."""

    def __init__(
        self,
        game_spec: GameSpec,
        experiment: Dict,
        player_models: List[Model],
    ):
        logger.info(
            f"[_init] Initializing TicTacToeGame GameMaster with spec={game_spec}"
        )
        logger.debug(f"[_init] Experiment parameters: {experiment}")

        super().__init__(game_spec, experiment, player_models)
        logger.info("[_init] TicTacToeGame initialization complete")

    def _on_setup(self, **game_instance):
        """
        Called during game setup. Configure game parameters and initialize players.

        Args:
            game_instance: Game instance parameters from instances.json
        """
        logger.info("[_on_setup] Setting up TicTacToeGame")
        logger.debug(f"[_on_setup] Game instance: {game_instance}")

        self.game_environment = TicTacToeEnvironment(game_instance)

        self.player_x = TicTacToePlayer(self.player_models[0])
        self.player_o = TicTacToePlayer(self.player_models[1])
        self.player_x.symbol = "X"
        self.player_o.symbol = "O"

        logger.debug(f"[_on_setup] Created players: {self.player_x}, {self.player_o}")

        self.add_player(self.player_x)
        self.add_player(self.player_o)
        logger.info(
            f"[_on_setup] Added players: {self.player_x.name}, {self.player_o.name}"
        )

        self.game_environment.reset()

    def _player_response_in_expected_format(
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

    def compute_turn_score(self) -> float:
        """
        Compute a score for the player's response based on the environment state.

        Args:
            response: The player's response
            context: Additional context for scoring

        Returns:
            float: The score for the response
        """
        score = 1.0 if self.game_environment.state["success"] else 0.0
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
            bench_score = 0.0
        else:
            bench_score = 100.0 if success else 0.0

        self.log_episode_score(BENCH_SCORE, bench_score)


class TicTacToeGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)
        logger.info(f"TicTacToeGameBenchmark initialized with game spec: {game_spec}")

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> TicTacToeGame:
        logger.info(f"Creating TicTacToeGame master with experiment: {experiment}")
        logger.debug(
            f"Player models: {[model.__class__.__name__ for model in player_models]}"
        )
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
        logger.info(f"Creating TicTacToeGameScorer with experiment: {experiment}")
        logger.debug(f"Game instance for scorer: \n{format_json(game_instance)}")
        return TicTacToeGameScorer(self.game_name, experiment, game_instance)
