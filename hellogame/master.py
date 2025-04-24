import logging
import string
from typing import Dict, List

from clemcore.backends import CustomResponseModel, Model
from clemcore.clemgame import GameBenchmark, GameRecorder, GameSpec, Player

from hellogame.game_environment import HelloGameEnvironment
from world_environments.game_master import DialogueGameMaster


class Greeted(Player):

    def __init__(self, target_name):
        super().__init__(CustomResponseModel())
        self.target_name = target_name

    def _custom_response(self, context):
        return f"{self.target_name}: Hi, thanks for having me!"


class Greeter(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, context):
        return "GREET: Hello Ted!"


class HelloGame(DialogueGameMaster):
    """This class implements a greeting game in which player A
    is greeting another player with a target name.

    This version uses the new GameEnvironment approach for state management.
    """

    def __init__(
        self,
        game_name: str,
        game_path: str,
        experiment: Dict,
        player_models: List[Model],
    ):
        # Create the game environment with the experiment parameters
        game_environment = HelloGameEnvironment(config=experiment)

        # Initialize with the game environment
        super().__init__(
            game_name, game_path, experiment, player_models, game_environment
        )

    def _on_setup(self, **game_instance):
        # Update the environment config with the game instance parameters
        self.game_environment.config.update(game_instance)

        # Reset the environment to apply the updated config
        observation, _ = self.game_environment.reset()

        # Create the players
        self.greeted = Greeted(game_instance["target_name"])
        self.greeter = Greeter(self.player_models[0])

        # Add the players: these will be logged to the records interactions.json
        # Note: During game play the players will be called in the order added here
        self.add_player(self.greeter)
        self.add_player(self.greeted)

    def _on_before_game(self):
        # Get the prompt from the environment state
        prompt = self.game_environment.state["prompt"]
        # Set the initial context for the greeter player
        self.set_context_for(self.greeter, prompt)

    def _does_game_proceed(self):
        # Check if the environment is in a terminal state
        return not self.game_environment.is_terminal()

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        # Validation is now handled by the environment in the step method
        # Always return True here as we'll let the environment handle validation
        return True

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        if player == self.greeter:
            self.set_context_for(self.greeted, parsed_response)

    def create_action_from_response(
        self, response: str, action_type: str = "text"
    ) -> Dict:
        """Convert the player's text response to an action for the environment"""
        return {"action_type": 0, "text": response}

    def _process_step_result(
        self, response: str, observation: Dict, terminated: bool, truncated: bool
    ) -> tuple[bool, Dict]:
        """Process the result from the environment step"""
        # All game logic is now handled in the environment
        info = {}

        # Get data from the environment
        env_info = self.game_environment.get_info()
        env_state = self.game_environment.get_state()

        # Set score information
        info["response_score"] = 1.0 if env_state["success"] else 0.0
        info["response_feedback"] = env_info.get("message", "")
        info["episode_score"] = 1.0 if env_state["success"] else 0.0

        # Return whether the game is done and the info dictionary
        return terminated or truncated, info

    def compute_episode_score(self):
        # Get the success status from the environment
        return 1.0 if self.game_environment.state["success"] else 0.0


class HelloGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> DialogueGameMaster:
        return HelloGame(self.game_name, self.game_path, experiment, player_models)
