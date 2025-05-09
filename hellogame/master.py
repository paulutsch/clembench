import json
import string
from typing import Any, Dict, List

from clemcore.backends import CustomResponseModel, Model
from clemcore.clemgame import GameBenchmark, GameScorer, GameSpec, Player

from _logger import setup_logger
from hellogame.game_environment import HelloGameEnvironment
from world_environments.game_master import DialogueGameMaster

logger = setup_logger(__name__)


def format_json(data: Any) -> str:
    """Format a dictionary or object as a pretty JSON string."""
    return json.dumps(data, indent=2, sort_keys=True, default=str)


class Greeted(Player):

    def __init__(self, target_name):
        super().__init__(CustomResponseModel())
        self.target_name = target_name
        logger.debug(f"Greeted player initialized with target_name={target_name}")

    def _custom_response(self, context):
        response = f"{self.target_name}: Hi, thanks for having me!"
        logger.debug(f"Greeted player responding with: {response}")
        return response


class Greeter(Player):

    def __init__(self, model: Model):
        super().__init__(model)
        logger.debug(f"Greeter player initialized with model={model}")

    def _custom_response(self, context):
        response = "GREET: Hello Ted!"
        logger.debug(f"Greeter player custom response: {response}")
        return response


class HelloGame(DialogueGameMaster):
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
    ):
        logger.info(
            f"[_init] Initializing HelloGame GameMaster with name={game_name}, path={game_path}"
        )
        logger.debug(f"[_init] Experiment parameters: {experiment}")

        game_environment = HelloGameEnvironment()

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

        self.greeted = Greeted(game_instance["target_name"])
        self.greeter = Greeter(self.player_models[0])
        logger.debug(
            f"[_on_setup] Created players: greeter={self.greeter}, greeted={self.greeted}"
        )

        # Add the players: these will be logged to the records interactions.json
        # Note: During game play the players will be called in the order added here
        self.add_player(self.greeter)
        self.add_player(self.greeted)
        logger.info(
            f"[_on_setup] Added players: greeter={self.greeter.name}, greeted={self.greeted.name}"
        )

        self.game_environment.set_observation_space(
            self.greeter, game_instance["prompt"]
        )
        self.game_environment.set_action_space(self.greeter, ["verbal_response"])

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """
        Validate the player's response.

        Will be called in GameMaster.play(), after the response is received, and before it is parsed.
        """
        logger.debug(
            f"[_validate_player_response] Validating response from {player.name}: {utterance}"
        )

        # For HelloGame, basic validation only - detailed validation happens in the environment
        if player == self.greeter:
            # Minimal validation, check if the response is not empty
            is_valid = bool(utterance.strip())
            logger.debug(
                f"[_validate_player_response] Greeter response validation result: {is_valid}"
            )
            return is_valid

        # The greeted player's response is automated, so always valid
        logger.debug(
            "[_validate_player_response] Greeted player response is always valid"
        )
        return True

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        logger.debug(
            f"[_on_valid_player_response] Processing valid response from {player.name}: {parsed_response}"
        )

        if player == self.greeter:
            self.game_environment.set_observation_space(self.greeted, parsed_response)
            logger.debug(
                f"Set observation for greeted player based on greeter's response"
            )

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
        In HelloGame, this is the same as the response score.
        """
        logger.info("[_compute_episode_score] Computing episode score")

        score = 1.0 if self.game_environment.state["success"] else 0.0
        logger.info(f"[_compute_episode_score] Episode score: {score}")
        return score


class HelloGameScorer(GameScorer):
    """
    Scorer for the Hello Game.
    Computes scores based on the game environment state.
    """

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.target_name = game_instance["target_name"]
        logger.debug(f"HelloGameScorer initialized for target: {self.target_name}")

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

        # in the Hello Game, we score based on:
        # 1. successful greeting (True/False)
        # 2. instruction followed (True/False)
        # 3. missing words (list of words)

        success = episode_interactions["success"]
        aborted = episode_interactions["aborted"]
        missing_words = episode_interactions["missing_words"]
        episode_score = episode_interactions["episode_score"]

        if "final_info" in episode_interactions:
            final_info = episode_interactions["final_info"]
            success = success or final_info["success"]
            aborted = aborted or final_info["aborted"]
            missing_words = missing_words or final_info["missing_words"]
            episode_score = episode_score or final_info["episode_score"]

        if "final_state" in episode_interactions:
            final_state = episode_interactions["final_state"]
            success = success or final_state["success"]
            aborted = aborted or final_state["aborted"]
            missing_words = missing_words or final_state["missing_words"]

        if not success and not aborted and not missing_words and not episode_score:
            if "turns" in episode_interactions and episode_interactions["turns"]:
                last_turn = episode_interactions["turns"][-1]
                for event in last_turn:
                    if "info" in event:
                        info = event["info"]
                        success = success or info["success"]
                        aborted = aborted or info["aborted"]
                        missing_words = missing_words or info["missing_words"]
                        episode_score = episode_score or info["episode_score"]
                        break

        self.log_episode_score("Success", 1 if success else 0)
        logger.debug(f"Episode success: {success}")

        self.log_episode_score("Aborted", 1 if aborted else 0)
        logger.debug(f"Episode aborted: {aborted}")

        if missing_words:
            missing_words_str = ", ".join(missing_words)
            self.log_episode_score("Missing Words", missing_words_str)
            logger.debug(f"Missing words: {missing_words_str}")

        self.log_episode_score("Episode Score", episode_score)
        logger.debug(f"Final episode score: {episode_score}")

        self.log_episode_score("bench_score", 1.0 if success else 0.0)


class HelloGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)
        logger.info(f"HelloGameBenchmark initialized with game spec: {game_spec}")

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> DialogueGameMaster:
        logger.info(f"Creating HelloGame master with experiment: {experiment}")
        logger.debug(
            f"Player models: {[model.__class__.__name__ for model in player_models]}"
        )
        return HelloGame(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        """
        Create a scorer for the Hello Game.

        Args:
            experiment: Experiment configuration dictionary
            game_instance: Game instance dictionary with specific parameters

        Returns:
            A HelloGameScorer instance
        """
        logger.info(f"Creating HelloGameScorer with experiment: {experiment}")
        logger.debug(f"Game instance for scorer: \n{format_json(game_instance)}")
        return HelloGameScorer(self.game_name, experiment, game_instance)
