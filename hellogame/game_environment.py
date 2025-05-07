"""
Hello Game Environment - implements the GameEnvironment interface for the Hello Game.
"""

import json
import string
from typing import Any, Dict, Tuple

from clemcore.clemgame.player import Player

from _logger import setup_logger
from world_environments.game_environment import GameEnvironment

logger = setup_logger(__name__)


# Helper function to pretty-format JSON objects
def format_json(data: Any) -> str:
    """Format a dictionary or object as a pretty JSON string."""
    return json.dumps(data, indent=2, sort_keys=True, default=str)


class HelloGameEnvironment(GameEnvironment):
    """
    Environment for the Hello Game where one player greets another.

    This environment tracks:
    - Required greeting words
    - Target name
    - Game success/failure status
    """

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            Tuple containing:
                - Initial observation dictionary
                - Information dictionary
        """
        logger.info("[reset] Resetting environment")

        target_name = self.config["target_name"]

        required_words = ["welcome", "hello", target_name.lower()]

        self.state = {
            "required_words": required_words,
            "missing_words": [],
            "success": False,
            "aborted": False,
            "current_context": "",
        }
        logger.debug(f"[reset] Reset state — new state: \n{format_json(self.state)}")

        self.player_observations = {}
        logger.debug("[reset] Reset player observations")

        self.terminated = False
        # self.info = {"message": "Environment reset"}
        logger.debug("[reset] Reset state flags and info")

        logger.info("[reset] Environment reset complete")
        # logger.debug(f"[reset] Initial info: \n{format_json(self.info)}")

        # return self.info

    def step(
        self, player: Player, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool]:
        """
        Take a step in the environment using the provided action from a specific player.

        Args:
            player: The player making the action
            action: Action dictionary with:
                - action_type: Type of action (always 'text' for this game)
                - text: The greeting text from the player

        Returns:
            Tuple containing:
                - Next observation
                - Reward value (1.0 for success, 0.0 for failure)
                - Whether the episode is terminated
                - Information dictionary
        """
        logger.info(
            f"[step] Environment step with player: {player.name} and action: {action}"
        )

        # TODO: does it make sense to update state before validation?
        action_body = action["body"]
        logger.debug(f"[step] Action body: {action_body}")
        self.state["current_context"] = action_body

        logger.debug("[step] Validating response")
        self._validate_action(player, action)

        score = 1.0 if self.state["success"] else 0.0
        logger.debug(f"[step] Score: {score}")

        logger.debug(f"[step] Game state: \n{format_json(self.state)}")
        if self.state["aborted"]:
            result_message = "invalid format (missing GREET:)"
            logger.warning(f"[step] Game aborted: {action_body}")
        elif self.state["success"]:
            result_message = "greeting successful"
            logger.info(f"[step] Successful step: {action_body}")
        else:
            missing = ",".join(self.state["missing_words"])
            result_message = f"greeting failed due to missing words: {missing}"
            logger.warning(f"[step] Unsuccessful step: {missing}")

        # self.info = {
        #     "message": result_message,
        #     "success": self.state["success"],
        #     "aborted": self.state["aborted"],
        #     "missing_words": self.state["missing_words"],
        #     "response_score": score,
        #     "response_feedback": result_message,
        # }
        # logger.debug(f"[step] Info dictionary: \n{format_json(self.info)}")

        observation = {
            "context": self.state["current_context"],
            "success": self.state["success"],
        }
        logger.debug(f"[step] Observation: \n{format_json(observation)}")

        self.set_observation_space(player, observation["context"])
        logger.debug(
            f"[step] Updated observation for player: {player.name if hasattr(player, 'name') else 'unknown'}"
        )

        self.terminated = True  # HelloGame only has one round
        logger.debug(f"[step] Game terminated: {self.terminated}")

        return observation, score, self.terminated

    def _validate_action(self, player, action: Dict[str, Any]) -> bool:
        """
        Validate if the response meets the requirements.

        Args:
            greeting: The greeting text to validate
        """
        logger.debug(f"[_validate_action] Validating action: {action}")
        if not super()._validate_action(player, action):
            return False

        response = action["body"]

        self.state["aborted"] = False
        self.state["success"] = True

        if not response.startswith("GREET:"):
            logger.warning(
                f"[_validate_action] Invalid action: action body doesn't start with 'GREET:': {action}"
            )
            self.state["aborted"] = True
            self.state["success"] = False
            return False

        response_lower = response.lower()
        response_clean = response_lower.translate(
            str.maketrans("", "", string.punctuation)
        )
        logger.debug(f"[_validate_action] Cleaned response: {response_clean}")

        missing_words = []
        for required_word in self.state["required_words"]:
            if required_word not in response_clean:
                logger.debug(
                    f"[_validate_action] Missing required word: {required_word}"
                )
                self.state["success"] = False
                missing_words.append(required_word)

        self.state["missing_words"] = missing_words

        if not missing_words:
            logger.info("[_validate_action] All required words found in response")
        else:
            logger.warning(
                f"[_validate_action] Missing words in response: {missing_words}"
            )

        return True
