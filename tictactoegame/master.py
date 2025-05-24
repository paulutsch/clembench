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
from clemcore.clemgame.metrics import BENCH_SCORE
from clemcore.utils.logger import format_json, setup_logger

from tictactoegame.game_environment import (
    TicTacToeAction,
    TicTacToeEnvironment,
    TicTacToeObservation,
    TicTacToePlayer,
)

logger = setup_logger(__name__)


class TicTacToeGame(EnvGameMaster):
    """Game master for the TicTacToe game."""

    def __init__(
        self,
        game_name: str,
        game_path: str,
        experiment: Dict,
        player_models: List[Model],
    ):
        logger.info(
            f"[_init] Initializing TicTacToeGame GameMaster with name={game_name}, path={game_path}"
        )
        logger.debug(f"[_init] Experiment parameters: {experiment}")

        super().__init__(game_name, game_path, experiment, player_models)
        logger.info("[_init] TicTacToeGame initialization complete")

    def _on_setup(self, **game_instance):
        """
        Called during game setup. Configure game parameters and initialize players.

        Args:
            game_instance: Game instance parameters from instances.json
        """
        logger.info("[_on_setup] Setting up TicTacToeGame")
        logger.debug(f"[_on_setup] Game instance: {game_instance}")

        self.game_environment = TicTacToeEnvironment()
        logger.info(f"[_on_setup] Game environment: {self.game_environment}")
        self.game_environment.config = game_instance

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

        self.game_environment.base_prompt = game_instance["prompt"]

        board = self.game_environment.format_board(self.game_environment.state["board"])
        player_x_observation: TicTacToeObservation = {
            "role": "user",
            "content": game_instance["prompt"] + "\n\n" + board,
        }
        player_o_observation: TicTacToeObservation = {
            "role": "user",
            "content": game_instance["prompt"] + "\n\n" + board,
        }
        initial_observations: Dict[str, Observation] = {
            self.player_x.name: player_x_observation,
            self.player_o.name: player_o_observation,
        }
        initial_action_spaces: Dict[str, ActionSpace] = {
            self.player_x.name: ["make_move"],
            self.player_o.name: ["make_move"],
        }
        self.game_environment.reset(initial_observations, initial_action_spaces)

    def _validate_player_response(
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

    def parse_action_from_response(self, response: str) -> TicTacToeAction:
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

    def _does_game_proceed(self) -> bool:
        """Check if the game should continue."""
        return not self.game_environment.state["terminated"]

    def _on_after_turn(self, player: TicTacToePlayer) -> None:
        """Handle actions after a player's turn."""
        if self.game_environment.state["terminated"]:
            winner = self.game_environment.state["winner"]
            if winner == 0:
                message = "Game ended in a draw!"
            else:
                winner_player = self.player_x if winner == 1 else self.player_o
                message = (
                    f"Game over! {winner_player.name} ({winner_player.symbol}) wins!"
                )

            for p in [self.player_x, self.player_o]:
                self.game_environment.observations[p.name][
                    "content"
                ] += f"\n\n{message}"


class TicTacToeGameScorer(GameScorer):
    """
    Scorer for the TicTacToe Game.
    Computes scores based on the game environment state.
    """

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        """
        Compute scores for the episode based on the interactions.

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

        not_aborted = 1 if not aborted else 0
        bench_score = (not_aborted + success) / 2
        self.log_episode_score(BENCH_SCORE, bench_score)
        logger.info(f"Final bench score: {bench_score}")


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
            self.game_name,
            self.game_path,
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
