import re
from typing import Dict, List

from clemcore.backends import Model
from clemcore.clemgame import EnvGameMaster, GameBenchmark, GameScorer, GameSpec, Player
from clemcore.clemgame.metrics import BENCH_SCORE, METRIC_ABORTED, METRIC_SUCCESS
from clemcore.utils.logger import setup_logger

from portalgame.game_environment import PortalAction, PortalGameEnvironment

logger = setup_logger(__name__)


class PortalPlayer(Player):
    """Player for the Portal game."""

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, context: Dict) -> str:
        """Custom response for the player."""
        return ""


class PortalGame(EnvGameMaster):
    """Game master for the Portal game."""

    def __init__(
        self,
        game_spec: GameSpec,
        experiment: Dict,
        player_models: List[Model],
    ):
        logger.info(f"[_init] Initializing PortalGame GameMaster with spec={game_spec}")
        logger.debug(f"[_init] Experiment parameters: {experiment}")
        super().__init__(game_spec, experiment, player_models)
        logger.info("[_init] PortalGame initialization complete")

    def _on_setup(self, **game_instance):
        """
        Called during game setup. Configure game parameters and initialize players.

        Args:
            game_instance: Game instance parameters from instances.json
        """
        logger.info("[_on_setup] Setting up PortalGame")
        logger.debug(f"[_on_setup] Game instance: {game_instance}")

        self.game_environment = PortalGameEnvironment(config=game_instance)
        logger.info(f"[_on_setup] Game environment: {self.game_environment}")

        self.player = PortalPlayer(self.player_models[0])
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
        try:
            # look for 'DIRECTION:' and extract the last non-whitespace char after it

            match = re.search(
                r"DIRECTION:\s*([nsewNSEW])", utterance, re.IGNORECASE | re.DOTALL
            )
            if match:
                direction = match.group(1).lower()
            else:
                # fallback: last non-whitespace char
                direction = utterance.strip()[-1].lower()
            return direction in ["n", "s", "e", "w"]
        except Exception:
            return False

    def _parse_action_from_response(self, response: str) -> PortalAction:
        """Create an action from a player's response.

        Args:
            response: The textual response from the player

        Returns:
            PortalAction: The parsed action
        """
        match = re.search(
            r"DIRECTION:\s*([nsewNSEW])", response, re.IGNORECASE | re.DOTALL
        )
        if match:
            direction = match.group(1).lower()
        else:
            direction = response.strip()[-1].lower()
        action: PortalAction = {
            "action_type": "move",
            "direction": direction,
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


class PortalGameScorer(GameScorer):
    """Scorer for the Portal game."""

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_episode_scores(self, interactions: Dict) -> None:
        """Compute episode-level scores for the Portal game.

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


class PortalGameBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> EnvGameMaster:
        return PortalGame(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return PortalGameScorer(self.game_name, experiment, game_instance)
