import json
import logging
import string
from typing import Dict, List, cast

import colorlog
from clemcore.backends import CustomResponseModel, Model
from clemcore.clemgame import GameBenchmark, GameRecorder, GameScorer, GameSpec, Player

from hellogame.game_environment import HelloGameEnvironment
from world_environments.game_master import DialogueGameMaster

# Set up logger for the HelloGame module
logger = logging.getLogger(__name__)
# Set the logger level itself to DEBUG
logger.setLevel(logging.DEBUG)

# Configure logger to print to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# Helper function to pretty-format JSON objects
def format_json(data) -> str:
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

        # Create the game environment with the experiment parameters
        game_environment = HelloGameEnvironment(config=experiment)

        # Initialize with the game environment
        super().__init__(
            game_name, game_path, experiment, player_models, game_environment
        )
        logger.info("[_init] HelloGame initialization complete")

    def _on_setup(self, **game_instance):
        """
        Called during game setup. Configure game parameters and initialize players.

        Args:
            game_instance: Game instance parameters from instances.json
        """
        logger.info("[_on_setup] Setting up HelloGame")

        self.game_environment.update_config(game_instance)
        logger.debug(
            f"[_on_setup] Updated environment config: \n{format_json(self.game_environment.get_config())}"
        )

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

    def _does_game_proceed(self):
        # Check if the environment is in a terminal state
        should_proceed = not self.game_environment.is_terminal()
        logger.debug(f"[_does_game_proceed] Game proceed check: {should_proceed}")
        return should_proceed

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
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

        # Only set context for the next player if the current player is the greeter
        if player == self.greeter:
            # Cast to the specific environment type
            env = cast(HelloGameEnvironment, self.game_environment)
            # Use the game environment to set the observation for the greeted player
            env.set_observation_for(self.greeted, parsed_response)
            logger.debug(
                f"Set observation for greeted player based on greeter's response"
            )

    def create_action_from_response(
        self, response: str, action_type: str = "text"
    ) -> Dict:
        """Convert the player's text response to an action for the environment"""
        action = {"action_type": 0, "text": response}
        logger.debug(
            f"[_create_action_from_response] Created action from response: {action}"
        )
        return action

    def get_response_feedback(self, response: str, context: Dict):
        """
        Provide feedback on the player's response.
        """
        logger.debug(
            f"[_get_response_feedback] Getting feedback for response: {response}"
        )

        # Cast to the specific environment type
        env = cast(HelloGameEnvironment, self.game_environment)
        env_info = env.get_info()
        feedback = env_info.get("message", "No feedback available")
        logger.debug(f"[_get_response_feedback] Response feedback: {feedback}")
        return feedback

    def compute_response_score(self, response: str, context: Dict):
        """
        Compute a score for the player's response based on the environment state.
        """
        logger.debug(
            f"[_compute_response_score] Computing score for response: {response}"
        )

        # Cast to the specific environment type
        env = cast(HelloGameEnvironment, self.game_environment)
        env_state = env.get_state()
        score = 1.0 if env_state.get("success", False) else 0.0
        logger.debug(f"[_compute_response_score] Response score: {score}")
        return score

    def compute_episode_score(self):
        """
        Compute the overall episode score.
        In HelloGame, this is the same as the response score.
        """
        logger.info("[_compute_episode_score] Computing final episode score")

        # Cast to the specific environment type
        env = cast(HelloGameEnvironment, self.game_environment)
        env_state = env.get_state()
        score = 1.0 if env_state.get("success", False) else 0.0
        logger.info(f"[_compute_episode_score] Episode score: {score}")
        return score

    def _on_after_game(self):
        """
        Called after the game is complete.
        Log final game state and clean up.
        """
        # Call the parent class's _on_after_game method
        super()._on_after_game()

        logger.info("[_on_after_game] Game completed, processing final state")

        # Get the environment as a HelloGameEnvironment to access its specific attributes
        env = cast(HelloGameEnvironment, self.game_environment)

        # Get the final state and info
        final_state = env.get_state()
        final_info = env.get_info()

        logger.debug(f"Final game state: \n{format_json(final_state)}")
        logger.debug(f"Final info: \n{format_json(final_info)}")

        # Log the game state and info for the scorer using log_key
        self.log_key("final_state", final_state)
        self.log_key("final_info", final_info)

        # Also log specific information that's important for scoring
        success = final_state.get("success", False)
        self.log_key("success", success)

        aborted = final_state.get("aborted", False)
        self.log_key("aborted", aborted)

        missing_words = final_state.get("missing_words", [])
        self.log_key("missing_words", missing_words)

        # Log the final episode score
        episode_score = final_info.get("episode_score", 0)
        self.log_key("episode_score", episode_score)

        logger.info(
            f"[_on_after_game] Game completed with success={success}, score={episode_score}"
        )


class HelloGameScorer(GameScorer):
    """
    Scorer for the Hello Game.
    Computes scores based on the game environment state.
    """

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.target_name = game_instance.get("target_name", "User")
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

        # For Hello Game, we consider:
        # 1. Whether the greeting was successful
        # 2. Whether it was aborted due to format issues
        # 3. If unsuccessful, which required words were missing

        # First check if the data was logged via log_key
        success = episode_interactions.get("success", False)
        aborted = episode_interactions.get("aborted", False)
        missing_words = episode_interactions.get("missing_words", [])
        episode_score = episode_interactions.get("episode_score", 0)

        # If not found directly, try looking at the final_info or final_state
        if "final_info" in episode_interactions:
            final_info = episode_interactions["final_info"]
            success = success or final_info.get("success", False)
            aborted = aborted or final_info.get("aborted", False)
            missing_words = missing_words or final_info.get("missing_words", [])
            episode_score = episode_score or final_info.get("episode_score", 0)

        if "final_state" in episode_interactions:
            final_state = episode_interactions["final_state"]
            success = success or final_state.get("success", False)
            aborted = aborted or final_state.get("aborted", False)
            missing_words = missing_words or final_state.get("missing_words", [])

        # If still not found, try to extract from turns as a last resort
        if not success and not aborted and not missing_words and not episode_score:
            if "turns" in episode_interactions and episode_interactions["turns"]:
                last_turn = episode_interactions["turns"][-1]
                for event in last_turn:
                    if "info" in event:
                        info = event["info"]
                        success = success or info.get("success", False)
                        aborted = aborted or info.get("aborted", False)
                        missing_words = missing_words or info.get("missing_words", [])
                        episode_score = episode_score or info.get("episode_score", 0)
                        break

        # Log success status
        self.log_episode_score("Success", 1 if success else 0)
        logger.debug(f"Episode success score: {1 if success else 0}")

        # Log whether the game was aborted
        self.log_episode_score("Aborted", 1 if aborted else 0)
        logger.debug(f"Episode aborted score: {1 if aborted else 0}")

        # Log missing words if any
        if missing_words:
            missing_words_str = ", ".join(missing_words)
            self.log_episode_score("Missing Words", missing_words_str)
            logger.debug(f"Missing words: {missing_words_str}")

        # Store the final episode score
        self.log_episode_score("Episode Score", episode_score)
        logger.debug(f"Final episode score: {episode_score}")

        # Compute a benchmark score
        self.log_episode_score("bench_score", 1.0 if success else 0.0)


class HelloGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)
        logger.info(f"HelloGameBenchmark initialized with game spec: {game_spec}")

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> DialogueGameMaster:
        logger.info(f"Creating HelloGame master with experiment: {experiment}")
        if logger.isEnabledFor(logging.DEBUG):
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Game instance for scorer: \n{format_json(game_instance)}")
        return HelloGameScorer(self.game_name, experiment, game_instance)
