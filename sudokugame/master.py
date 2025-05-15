from typing import Dict, List, Tuple

from clemcore.backends import Model
from clemcore.clemgame import (
    Action,
    ActionSpace,
    EnvGameMaster,
    GameBenchmark,
    GameScorer,
    GameSpec,
    Observation,
    Player,
)
from clemcore.utils.logger import format_json, setup_logger

from sudokugame.game_environment import (
    SudokuAction,
    SudokuEnvironment,
    SudokuObservation,
)

logger = setup_logger(__name__)


class SudokuPlayer(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, context: Dict) -> str:
        return "Hello, world!"


class SudokuGame(EnvGameMaster):
    """Game master for the Sudoku game."""

    def __init__(
        self,
        game_name: str,
        game_path: str,
        experiment: Dict,
        player_models: List[Model],
        board_size: int,
        difficulty: float,
    ):
        logger.info(
            f"[_init] Initializing SudokuGame GameMaster with name={game_name}, path={game_path}"
        )
        logger.debug(f"[_init] Experiment parameters: {experiment}")

        self.game_environment: SudokuEnvironment = SudokuEnvironment(
            board_size, difficulty
        )

        super().__init__(
            game_name, game_path, experiment, player_models, self.game_environment
        )
        logger.info("[_init] SudokuGame initialization complete")

    def _on_setup(self, **game_instance):
        """
        Called during game setup. Configure game parameters and initialize players.

        Args:
            game_instance: Game instance parameters from instances.json
        """
        logger.info("[_on_setup] Setting up SudokuGame")

        logger.debug(f"[_on_setup] Game instance: {game_instance}")
        self.game_environment.config = game_instance

        self.player = SudokuPlayer(self.player_models[0])
        logger.debug(f"[_on_setup] Created player: {self.player}")

        self.add_player(self.player)
        logger.info(f"[_on_setup] Added player: {self.player.name}")

        player_observation: SudokuObservation = {
            "role": "user",
            "prompt": game_instance["prompt"],
            "board": self.game_environment.state["board"],
        }
        initial_observations: Dict[str, Observation] = {
            self.player.name: player_observation,
        }
        initial_action_spaces: Dict[str, ActionSpace] = {
            self.player.name: ["fill_cell"]
        }
        self.game_environment.reset(initial_observations, initial_action_spaces)

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """
        Validate the player's response.

        Args:
            player: The player making the response
            utterance: The player's response

        Returns:
            bool: Whether the response is valid
        """
        logger.debug(
            f"[_validate_player_response] Validating response from {player.name}: {utterance}"
        )

        # check if the response is in the correct format
        # example response: "1 2 3"
        try:
            row, col, value = map(int, utterance.strip().split())
            if not (
                0 <= row < self.game_environment.board_size * 3
                and 0 <= col < self.game_environment.board_size * 3
                and 0 <= value < self.game_environment.board_size
            ):
                return False
            return True
        except (ValueError, TypeError):
            return False

    def parse_action_from_response(self, response: str) -> SudokuAction:
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
        In SudokuGame, this is the same as the response score.

        Returns:
            float: The episode score
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
    ) -> SudokuGame:
        logger.info(f"Creating SudokuGame master with experiment: {experiment}")
        logger.debug(
            f"Player models: {[model.__class__.__name__ for model in player_models]}"
        )
        # Get grid_shape from experiment config if available, otherwise use default
        board_size = experiment.get("board_size", 3)
        difficulty = experiment.get("difficulty", 0.5)
        return SudokuGame(
            self.game_name,
            self.game_path,
            experiment,
            player_models,
            board_size,
            difficulty,
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
