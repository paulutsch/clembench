from typing import Any, Dict, List, Tuple

from clemcore.backends import CustomResponseModel, Model
from clemcore.clemgame import GameBenchmark, GameScorer, GameSpec

from _logger import format_json, setup_logger
from sudoku.game_environment import SudokuEnvironment
from world_environments import DialogueGameMaster, Player

logger = setup_logger(__name__)


class SudokuPlayer(Player):

    def __init__(self, model: Model):
        super().__init__(model)


class SudokuGame(DialogueGameMaster):
    """This class implements a greeting game in which player A
    is greeting another player with a target name.

    This version uses the GameEnvironment approach for state management.
    """

    def __init__(
        self,
        game_name: str,
        game_path: str,
        experiment: Dict,
        player_models: List[Model],
        grid_shape: Tuple[int, int],
    ):
        logger.info(
            f"[_init] Initializing HelloGame GameMaster with name={game_name}, path={game_path}"
        )
        logger.debug(f"[_init] Experiment parameters: {experiment}")

        game_environment = SudokuEnvironment(grid_shape)

        super().__init__(
            game_name, game_path, experiment, player_models, game_environment
        )
        logger.info("[_init] HelloGame initialization complete")

    # TODO: can _on_setup be generalized in the parent class? should we encapsulate adding players/config/observations/actions here? what's most intuitive for the developer?
    def _on_setup(self, **game_instance):
        """
        Called during game setup. Configure game parameters and initialize players.

        Args:
            game_instance: Game instance parameters from instances.json
        """
        logger.info("[_on_setup] Setting up HelloGame")

        logger.debug(f"[_on_setup] Game instance: {game_instance}")
        self.game_environment.config = game_instance

        self.player = SudokuPlayer(self.player_models[0])
        logger.debug(f"[_on_setup] Created player: {self.player}")

        self.add_player(self.player)
        logger.info(f"[_on_setup] Added player: {self.player.name}")

        self.game_environment.set_observation_space(
            self.player, game_instance["prompt"]
        )
        self.game_environment.set_action_space(self.player, ["verbal_response"])

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """
        Validate the player's response.

        Will be called in GameMaster.play(), after the response is received, and before it is parsed.
        """
        logger.debug(
            f"[_validate_player_response] Validating response from {player.name}: {utterance}"
        )

        is_valid = bool(utterance.strip())
        logger.debug(
            f"[_validate_player_response] Greeter response validation result: {is_valid}"
        )
        return is_valid

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        logger.debug(
            f"[_on_valid_player_response] Processing valid response from {player.name}: {parsed_response}"
        )

        self.game_environment.set_observation_space(self.player, parsed_response)
        logger.debug(f"Set observation for player based on player's response")

    def compute_response_score(self, response: str, context: Dict):
        """
        Compute a score for the player's response based on the environment state.
        """
        logger.debug(
            f"[_compute_response_score] Computing response score for response: {response}"
        )

        score = 1.0 if self.game_environment.state["success"] else 0.0
        logger.debug(f"[_compute_response_score] Response score: {score}")
        return score

    def compute_episode_score(self):
        """
        Compute the overall episode score.
        In SudokuGame, this is the same as the response score.
        """
        logger.info("[_compute_episode_score] Computing episode score")

        score = 1.0 if self.game_environment.state["success"] else 0.0
        logger.info(f"[_compute_episode_score] Episode score: {score}")
        return score


class SudokuGameScorer(GameScorer):
    """
    Scorer for the Sudoku Game.
    Computes scores based on the game environment state.
    """

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.grid_shape = game_instance["grid_shape"]
        logger.debug(f"SudokuGameScorer initialized for grid shape: {self.grid_shape}")

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

        success = episode_interactions["success"]
        aborted = episode_interactions["aborted"]
        episode_score = episode_interactions["episode_score"]

        self.log_episode_score("Success", 1 if success else 0)
        logger.debug(f"Episode success: {success}")

        self.log_episode_score("Aborted", 1 if aborted else 0)
        logger.debug(f"Episode aborted: {aborted}")

        self.log_episode_score("Episode Score", episode_score)
        logger.debug(f"Final episode score: {episode_score}")

        self.log_episode_score("bench_score", 1.0 if success else 0.0)


class SudokuGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)
        logger.info(f"SudokuGameBenchmark initialized with game spec: {game_spec}")
        # Default to 9x9 grid if not specified
        self.grid_shape = (9, 9)

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> DialogueGameMaster:
        logger.info(f"Creating SudokuGame master with experiment: {experiment}")
        logger.debug(
            f"Player models: {[model.__class__.__name__ for model in player_models]}"
        )
        # Get grid_shape from experiment config if available, otherwise use default
        grid_shape = experiment.get("grid_shape", self.grid_shape)
        return SudokuGame(
            self.game_name, self.game_path, experiment, player_models, grid_shape
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
