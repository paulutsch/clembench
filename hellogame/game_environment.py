"""
Hello Game Environment - implements the GameEnvironment interface for the Hello Game.
"""

import json
import string
from typing import Any, Dict, List, Tuple

from clemcore.clemgame.player import Player

from _logger import format_json, setup_logger
from world_environments.game_environment import GameEnvironment, GameState

logger = setup_logger(__name__)


# Example of how to extend GameState for a specific game:
class HelloGameState(GameState):
    """Example of extending GameState for a specific game.

    Additional fields:
    - score: The current game score
    - level: The current game level
    - inventory: List of items in player's inventory
    """

    required_words: List[str]
    missing_words: List[str]


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

        self.state: HelloGameState = {
            "required_words": required_words,
            "missing_words": [],
            "success": False,
            "aborted": False,
            "current_context": "",
            "terminated": False,
        }
        logger.debug(f"[reset] Reset state — new state: \n{format_json(self.state)}")

        self.player_observations = {}
        logger.debug("[reset] Reset player observations")

        logger.info("[reset] Environment reset complete")

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

        action_body = action["body"]
        logger.debug(f"[step] Action body: {action_body}")

        logger.debug("[step] Validating response")
        self._validate_action(player, action)
        self._update_state_through_action(player, action)

        logger.debug(f"[step] Game state: \n{format_json(self.state)}")
        if self.state["aborted"]:
            logger.warning(f"[step] Game aborted: {action_body}")
        elif self.state["success"]:
            logger.info(f"[step] Successful step: {action_body}")
        else:
            missing = ",".join(self.state["missing_words"])
            logger.warning(f"[step] Unsuccessful step: {missing}")

        observation = {
            "context": self.state["current_context"],
            "success": self.state["success"],
        }
        logger.debug(f"[step] Observation: \n{format_json(observation)}")

        self.set_observation_space(player, observation["context"])
        logger.debug(
            f"[step] Updated observation for player: {player.name if hasattr(player, 'name') else 'unknown'}"
        )

        self.state["terminated"] = True  # HelloGame only has one round
        logger.debug(f"[step] Game terminated: {self.state['terminated']}")

        return observation, self.state["success"], self.state["terminated"]

    def _update_state_through_action(self, player: Player, action: Dict[str, Any]):
        """
        Update the state based on the action.
        """
        response = action["body"]
        self.state["current_context"] = response

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
            self.state["terminated"] = True
