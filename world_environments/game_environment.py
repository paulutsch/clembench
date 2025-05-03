"""
Base class for all clembench environments.

Environments:
- are self-contained systems that manage their own state
- include an action space of actions that can be taken within them to alter their state
- include an observation space of observations that can be made of the state of the environment
- include a termination condition that defines when the environment is finished
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import gymnasium as gym
from clemcore.clemgame.player import Player
from gymnasium import spaces

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

logger = logging.getLogger(__name__)


class GameEnvironment(gym.Env, Generic[ObsType, ActType], ABC):
    """
    Base class for game environments in Clem.

    This class follows both the Gymnasium interface and the clembench framework.
    """

    def __init__(
        self,
        action_spaces: Dict[str, spaces.Space[ActType]],
        observation_spaces: Dict[str, spaces.Space[ObsType]],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a game environment.

        Args:
            config: Optional configuration parameters for the environment
        """
        super().__init__()
        logger.info(
            f"Initializing game environment with action spaces: {action_spaces} and observation spaces: {observation_spaces}"
        )

        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces

        # environment state
        self.state: Dict[str, Any] = {}
        self.terminated: bool = False
        self.truncated: bool = False
        self.info: Dict[str, Any] = {}

        # environment config
        self.config: Dict[str, Any] = config or {}

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
        self, player: Player, action: ActType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment using the provided action.

        Args:
            action: The action to take in the environment
            player: The player taking the action

        Returns:
            Tuple containing:
                - Next observation
                - Reward value
                - Whether the episode is terminated
                - Whether the episode is truncated
                - Information dictionary
                    - response_score: The score for the response
                    - response_feedback: The feedback for the response
                    - episode_score: The score for the episode
        """
        raise NotImplementedError

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
            True if the environment is terminated or truncated, False otherwise
        """
        return self.terminated or self.truncated

    def get_info(self) -> Dict[str, Any]:
        """
        Get the current information dictionary.

        Returns:
            The current information dictionary
        """
        return self.info

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        """
        return self.config

    def update_config(self, config: Dict[str, Any]):
        """
        Update the configuration.

        Args:
            config: The configuration to update
        """
        self.config.update(config)

    def validate_action(self, player: Player, action: ActType) -> bool:
        """
        Validate if an action is legal in the current state.

        Args:
            action: The action to validate

        Returns:
            True if the action is valid, False otherwise
        """
        if self.action_spaces[player.name] is None:
            return True
        return self.action_spaces[player.name].contains(action)

    @abstractmethod
    def legal_actions(self, player: Player) -> List[ActType]:
        """
        Get a list of legal actions in the current state.

        Returns:
            A list of legal actions
        """
        raise NotImplementedError

    @abstractmethod
    def set_observation_for(self, player: Player, observation: ObsType):
        """
        Set the observation for a player.
        """
        raise NotImplementedError

    @abstractmethod
    def observe_for(self, player: Player) -> ObsType:
        """
        Get the observation for a player.

        Args:
            player: The player to get the observation for

        Returns:
            The observation for the player
        """
        raise NotImplementedError

    @abstractmethod
    def render(self) -> Union[str, Dict[str, Any]]:
        """
        Render the current state of the environment.

        Returns:
            A string or dictionary representation of the current state
        """
        raise NotImplementedError
