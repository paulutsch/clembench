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
from gymnasium import spaces

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

logger = logging.getLogger(__name__)


class GameEnvironment(gym.Env, Generic[ObsType, ActType], ABC):
    """
    Base class for game environments in Clem.

    This class follows both the Gymnasium interface and the clembench framework.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a game environment.

        Args:
            config: Optional configuration parameters for the environment
        """
        super().__init__()
        logger.info(f"Initializing game environment with config: {config}")
        self.config = config if config is not None else {}

        # set the subclasses' action and observation spaces
        self.action_space: spaces.Space[ActType]
        self.observation_space: spaces.Space[ObsType]

        # set the environment state
        self.state: Dict[str, Any] = {}
        self.terminated: bool = False
        self.truncated: bool = False
        self.info: Dict[str, Any] = {}

    @abstractmethod
    def reset(self) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Returns:
            Tuple containing:
                - Initial observation
                - Information dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, action: ActType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment using the provided action.

        Args:
            action: The action to take in the environment

        Returns:
            Tuple containing:
                - Next observation
                - Reward value
                - Whether the episode is terminated
                - Whether the episode is truncated
                - Information dictionary
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

    def validate_action(self, action: ActType) -> bool:
        """
        Validate if an action is legal in the current state.

        Args:
            action: The action to validate

        Returns:
            True if the action is valid, False otherwise
        """
        if self.action_space is None:
            return True
        return self.action_space.contains(action)

    @abstractmethod
    def legal_actions(self) -> List[ActType]:
        """
        Get a list of legal actions in the current state.

        Returns:
            A list of legal actions
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
