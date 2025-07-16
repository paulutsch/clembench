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
from clemcore.clemgame.metrics import BENCH_SCORE
from clemcore.utils.logger import format_json, setup_logger

from sudokugame.game_environment import SudokuAction, SudokuEnvironment, SudokuPlayer

logger = setup_logger(__name__)


class SudokuGame(EnvGameMaster):
    """Game master for the Sudoku game."""

    def __init__(
        self,
        game_name: str,
        game_path: str,
        experiment: Dict,
        player_models: List[Model],
    ):
        logger.info(
            f"[_init] Initializing SudokuGame GameMaster with name={game_name}, path={game_path}"
        )
        logger.debug(f"[_init] Experiment parameters: {experiment}")

        super().__init__(game_name, game_path, experiment, player_models)
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

        self.game_environment.base_prompt = game_instance["prompt"]

        grid = self.game_environment.format_board(
            self.game_environment._get_board_from_grid()
        )
        player_observation: Observation = {
            "role": "user",
            "content": game_instance["prompt"] + "\n\n" + grid,
        }
        initial_observations: Dict[str, Observation] = {
            self.player.name: player_observation,
        }
        initial_action_spaces: Dict[str, ActionSpace] = {
            self.player.name: ["fill_cell"]
        }
        self.game_environment.reset(initial_observations, initial_action_spaces)

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
            self.game_name,
            self.game_path,
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
