"""
Hello Game Environment - implements the GameEnvironment interface for the Hello Game.
"""

import string
from typing import Dict, List

from clemcore.clemgame import (
    Action,
    ActionSpace,
    GameEnvironment,
    GameState,
    Observation,
)
from clemcore.clemgame.player import Player
from clemcore.utils.logger import format_json, setup_logger

logger = setup_logger(__name__)


class HelloGameState(GameState):
    """Example of extending GameState for a specific game.

    Additional fields:
    - score: The current game score
    - level: The current game level
    - inventory: List of items in player's inventory
    """

    required_words: List[str]
    missing_words: List[str]


class HelloGameAction(Action):
    """Action for the HelloGame."""

    message: str  # the conversational response of the greeter


class HelloGameEnvironment(GameEnvironment):
    """
    Environment for the HelloGame in which one player greets the other.

    This environment tracks:
    - Required greeting words
    - Target name
    - Game success/failure status
    """

    def reset(self):
        """
        Reset the environment to an initial state.

        Args:
            initial_observations: Dictionary of initial observations
            initial_action_spaces: Dictionary of initial action spaces

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
            "terminated": False,
            "moves": 0,
            "warning": "",
        }
        logger.debug(f"[reset] Reset state — new state: \n{format_json(self.state)}")

        logger.info("[reset] Environment reset complete")

        greeter_observation: Observation = {
            "role": "user",
            "content": self.config["prompt"],
        }
        initial_observations: Dict[str, Observation] = {
            self.players[0].name: greeter_observation
        }
        initial_action_spaces: Dict[str, ActionSpace] = {
            self.players[0].name: ["verbal_response"]
        }

        self.observations = initial_observations
        self.action_spaces = initial_action_spaces

    def update_observation(self, player: Player):
        """
        Update the observation for a specific player.
        """
        observation: Observation = {
            "role": "user",
            "content": (
                "You won the game!" if self.state["success"] else "You lost the game!"
            ),
        }
        self.observations[player.name] = observation
        logger.debug(f"[step] Observation: \n{format_json(observation)}")

    def _update_state_through_action(self, player: Player, action: HelloGameAction):
        """
        Update the state based on the action.
        """
        response = action["message"]

        self.state["aborted"] = False
        self.state["success"] = True
        self.state["terminated"] = True

        if not response.startswith("GREET:"):
            logger.warning(
                f"[_validate_action] Invalid action: action body doesn't start with 'GREET:': {action}"
            )
            self.state["aborted"] = True
            self.state["success"] = False
            self.state["terminated"] = True

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
                self.state["terminated"] = True
                missing_words.append(required_word)

        self.state["missing_words"] = missing_words

    def _is_action_valid_in_state(
        self, player: Player, action: HelloGameAction
    ) -> tuple[bool, str]:
        """Check if action is valid in current state."""
        return True, ""

    def update_observations(self):
        """Update observations for all players."""
        for player in self.players:
            self.update_observation(player)

    def _render_state_as_string(self, player_name: str | None = None) -> str:
        """Render state as string."""
        return (
            f"Success: {self.state['success']}, Missing: {self.state['missing_words']}"
        )

    def _render_state_as_image(self, player_name: str | None = None) -> bytes:
        """Render state as image (not implemented for HelloGame)."""
        return b""

    def _render_state_as_human_readable(self, player_name: str | None = None) -> str:
        """Render state in human readable format."""
        return self._render_state_as_string(player_name)
