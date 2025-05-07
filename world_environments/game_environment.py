"""
Base class for all clembench environments.

Environments:
- are self-contained systems that manage their own state
- include an action space of actions that can be taken within them to alter their state
- include an observation space of observations that can be made of the state of the environment
- include a termination condition that defines when the environment is finished
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from clemcore.clemgame.player import Player

from _logger import setup_logger

logger = setup_logger(__name__)


def format_json(data: Any) -> str:
    """Format a dictionary or object as a pretty JSON string."""
    return json.dumps(data, indent=2, sort_keys=True, default=str)


class GameEnvironment(ABC):
    """
    Base class for game environments in Clem.

    This class follows both the Gymnasium interface and the clembench framework.
    """

    def __init__(
        self,
        action_spaces: Optional[Dict[str, List[Any]]] = None,
        observation_spaces: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a game environment.

        Args:
            action_spaces: Dictionary of action spaces, one key per player
            observation_spaces: Dictionary of observation spaces, one key per player
        """
        super().__init__()
        logger.info(
            f"Initializing game environment with action spaces: {action_spaces} and observation spaces: {observation_spaces}"
        )

        self.action_spaces = action_spaces or {}
        self.observation_spaces = observation_spaces or {}

        self.config = config or {}

        self.state: Dict[str, Any] = {}
        self.terminated: bool = False

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to its initial state.

        Returns:
            Information dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, player: Player, action_dict: Dict[str, Any]
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
        """
        logger.info(f"[step] Environment step with player: {player.name}")
        logger.debug(f"[step] Action: {action_dict}")

        if not self._validate_action(player, action_dict):
            # TODO: implement an option to continue the game on invalid action?
            raise ValueError(f"[step] Invalid action: {action_dict}")

        text_response = action_dict["body"]
        logger.debug(f"[step] Text response: {text_response}")
        self.state["current_context"] = text_response

        logger.debug("[step] Validating action")
        is_valid = self._validate_action(player, action_dict)
        if not is_valid:
            # TODO: implement an option to continue the game on invalid action?
            raise ValueError(f"[step] Invalid action: {action_dict}")

        self.state["round"] += 1
        logger.debug(f"[step] Round updated to: {self.state['round']}")

        self.terminated = True  # Hello game only has one round
        logger.debug(f"[step] Game terminated: {self.terminated}")

        score = 1.0 if self.state["success"] else 0.0
        logger.debug(f"[step] Reward: {score}")

        logger.debug(f"[step] Game state: \n{format_json(self.state)}")
        if self.state["aborted"]:
            result_message = "invalid format: abort game"
            logger.warning(f"[step] Game aborted: {text_response}")
        elif self.state["success"]:
            result_message = "greeting successful: end game"
            logger.info("[step] Greeting was successful")
        else:
            missing = ",".join(self.state["missing_words"])
            result_message = f"greeting failed: missing words=[{missing}]"
            logger.warning(f"[step] Greeting failed, missing words: {missing}")

        observation = {
            "context": self.state["current_context"],
            "success": self.state["success"],
        }
        logger.debug(f"[step] Observation: \n{format_json(observation)}")

        self.set_observation_space(player, observation["context"])
        logger.debug(
            f"[step] Updated observation for player: {player.name if hasattr(player, 'name') else 'unknown'}"
        )

        return observation, score, self.terminated

    def _validate_action(self, player: Player, action: Dict[str, Any]) -> bool:
        """
        Validate if an action is legal in the current state.
        """
        action_type = action["action_type"]
        if action_type not in self.action_spaces[player.name]:
            return False
        if not self._is_action_valid_in_state(player, action_type):
            return False
        return True

    def _is_action_valid_in_state(self, player: Player, action_type: str) -> bool:
        """
        Validate if an action is legal in the current state.

        Overwrite this method in your subclass to implement custom validation logic based on the current state.
        """
        return True

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current environment state.

        Returns:
            The current state as a dictionary
        """
        return self.state

    def is_terminal(self) -> bool:
        """
        Check if the environment is in a terminal state.

        Returns:
            True if the environment is terminated, False otherwise
        """
        return self.terminated

    def set_observation_space(self, player: Player, content: str):
        """
        Set the observation space for a specific player.

        Args:
            player: The player to set the observation for
            content: The text content for the observation
        """
        observation_space = {"role": "user", "content": content}

        self.observation_spaces[player.name] = observation_space

        logger.info(
            f"[set_observation_space_for] Set observation space for player: {player.name}"
        )

    def get_observation_space(self, player: Player) -> Dict[str, Any]:
        """
        Get the current observation for a specific player.

        Args:
            player: The player to get the observation for

        Returns:
            The observation for the player
        """
        logger.debug(f"[observe_for] Getting observation for player: {player.name}")

        if player.name not in self.observation_spaces:
            logger.warning(
                f"[observe_for] No observation found for player: {player.name}. Creating default."
            )
            raise ValueError(
                f"[observe_for] No observation found for player: {player.name}"
            )

        observation = self.observation_spaces[player.name]
        logger.debug(f"[observe_for] Observation for {player.name}: {observation}")
        return observation

    def set_action_space(self, player: Player, action_space: List[Any]):
        """
        Set the action space for a specific player.

        Args:
            player: The player to set the action space for
            action_space: The action space to set
        """
        self.action_spaces[player.name] = action_space

    def render(self) -> Union[str, Dict[str, Any]]:
        """
        Render the current state of the environment.

        Returns:
            A dictionary representation of the current state
        """
        render_state = {
            "prompt": self.state["prompt"],
            "success": self.state["success"],
            "missing_words": self.state["missing_words"],
            "aborted": self.state["aborted"],
        }
        logger.debug(f"[render] Current state: \n{format_json(render_state)}")
        return render_state
