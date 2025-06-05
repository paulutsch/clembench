import re
from typing import Dict, List

from clemcore.backends import Model
from clemcore.clemgame import (
    ActionSpace,
    EnvGameMaster,
    GameBenchmark,
    GameScorer,
    GameSpec,
    Player,
)
from clemcore.clemgame.metrics import BENCH_SCORE
from clemcore.utils.logger import format_json, setup_logger

from portalgame.game_environment import (
    PortalAction,
    PortalGameEnvironment,
    PortalObservation,
)

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
        game_name: str,
        game_path: str,
        experiment: Dict,
        player_models: List[Model],
    ):
        logger.info(
            f"[_init] Initializing PortalGame GameMaster with name={game_name}, path={game_path}"
        )
        logger.debug(f"[_init] Experiment parameters: {experiment}")

        super().__init__(game_name, game_path, experiment, player_models)
        logger.info("[_init] PortalGame initialization complete")

    def _on_setup(self, **game_instance):
        """
        Called during game setup. Configure game parameters and initialize players.

        Args:
            game_instance: Game instance parameters from instances.json
        """
        logger.info("[_on_setup] Setting up PortalGame")
        logger.debug(f"[_on_setup] Game instance: {game_instance}")

        grid_size = game_instance.get("grid_size", 10)
        logger.info(f"[_on_setup] Grid size: {grid_size}")

        self.game_environment = PortalGameEnvironment(grid_size)
        logger.info(f"[_on_setup] Game environment: {self.game_environment}")

        # Set the game configuration including grid layout
        self.game_environment.config = game_instance
        logger.debug(f"[_on_setup] Set game configuration: {game_instance}")

        self.player = PortalPlayer(self.player_models[0])
        logger.debug(f"[_on_setup] Created player: {self.player}")

        self.add_player(self.player)
        logger.info(f"[_on_setup] Added player: {self.player.name}")

        self.game_environment.reset()

    def _player_response_in_expected_format(self, player: Player, utterance: str) -> bool:
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

    def compute_response_score(self, response: str, context: Dict) -> float:
        """
        Compute a score for the player's response based on the environment state.

        Args:
            response: The player's response
            context: Additional context for scoring

        Returns:
            float: The score for the response
        """
        logger.debug(
            f"[_compute_response_score] Computing response score for response: {response}"
        )

        score = 1.0 if self.game_environment.state["success"] else 0.0
        logger.debug(f"[_compute_response_score] Response score: {score}")
        return score

    def compute_episode_score(self) -> float:
        """
        Compute the overall episode score.

        Returns:
            float: The episode score
        """
        logger.info("[_compute_episode_score] Computing episode score")

        score = 1.0 if self.game_environment.state["success"] else 0.0
        logger.info(f"[_compute_episode_score] Episode score: {score}")
        return score


class PortalGameScorer(GameScorer):
    """Scorer for the Portal game."""

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        """
        Compute scores for the episode based on the interactions.
        The Hello Game is scored based on whether the greeting was successful.

        Args:
            episode_interactions: Dictionary containing the episode interactions
        """
        logger.debug(
            f"Computing scores for episode interactions: \n{format_json(episode_interactions)}"
        )

        success = 1 if episode_interactions["success"] else 0
        aborted = 1 if episode_interactions["aborted"] else 0
        episode_score = episode_interactions["episode_score"]

        self.log_episode_score("Success", success)
        logger.info(f"Episode success: {success}")

        self.log_episode_score("Aborted", aborted)
        logger.info(f"Episode aborted: {aborted}")

        self.log_episode_score("Episode Score", episode_score)
        logger.info(f"Final episode score: {episode_score}")

        # bench score based on following instructions (not aborted) and winning (success)
        not_aborted = 1 if not aborted else 0
        bench_score = (not_aborted + success) / 2
        self.log_episode_score(BENCH_SCORE, bench_score)
        logger.info(f"Final bench score: {bench_score}")


class PortalGameBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> EnvGameMaster:
        return PortalGame(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return PortalGameScorer(self.game_name, experiment, game_instance)
