"""
Hello Game Environment - implements the GameEnvironment interface for the Hello Game.
"""

import json
import logging
import string
from typing import Any, Dict, List, Optional, Tuple, Union

from clemcore.clemgame.player import Player
from gymnasium import spaces

from world_environments.game_environment import GameEnvironment

logger = logging.getLogger(__name__)


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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Hello Game environment.

        Args:
            config: Configuration dictionary used to reset the environment.
            Can include:
                - target_name: The name of the person to be greeted
                - language: The language for the greeting
                - prompt: The prompt for the greeter
        """
        # Create these dictionaries before calling super().__init__
        self.player_observations = {}

        # Define action and observation spaces
        # Important: We're initializing with empty dictionaries to avoid type checking issues
        super().__init__({}, {}, config)

        # Define default action space
        action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(1),  # Only one action type: greet
                "text": spaces.Text(1000),  # Text with max 1000 chars
            }
        )

        # Define default observation space
        observation_space = spaces.Dict(
            {
                "context": spaces.Text(5000),
                "success": spaces.Discrete(2),  # 0: False, 1: True
                "round": spaces.Discrete(10),  # Max 10 rounds
            }
        )

        # Set spaces after initialization to avoid type errors
        self.action_spaces = {}
        self.action_spaces["default"] = action_space

        self.observation_spaces = {}
        self.observation_spaces["default"] = observation_space

        # Set empty state
        self.state = {}

        logger.info("[_init] HelloGameEnvironment initialized successfully")

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to its initial state.

        Returns:
            Tuple containing:
                - Initial observation dictionary
                - Information dictionary
        """
        logger.info("[reset] Resetting environment")

        target_name = self.config.get("target_name", "User")
        language = self.config.get("language", "en")
        prompt = self.config.get("prompt", "")

        required_words = ["welcome", "hello", target_name.lower()]

        self.state = {
            "target_name": target_name,
            "language": language,
            "prompt": prompt,
            "required_words": required_words,
            "missing_words": [],
            "success": False,
            "aborted": False,
            "round": 0,
            "current_context": prompt,
        }
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[reset] Reset state: \n{format_json(self.state)}")

        self.player_observations = {}
        logger.debug("[reset] Reset player observations")

        self.terminated = False
        self.truncated = False
        self.info = {"message": "Environment reset"}
        logger.debug("[reset] Reset state flags and info")

        logger.info("[reset] Environment reset complete")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[reset] Initial info: \n{format_json(self.info)}")

        return self.info

    def step(
        self, player: Player, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
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
                - Whether the episode is truncated
                - Information dictionary
        """
        logger.info(
            f"[step] Environment step with player: {player.name if hasattr(player, 'name') else 'unknown'}"
        )
        logger.debug(f"[step] Action: {action}")

        if self.terminated or self.truncated:
            logger.warning("[step] Environment already terminated, ignoring action")
            self.info = {"warning": "Environment already terminated"}
            observation = {
                "context": self.state["current_context"],
                "success": self.state["success"],
                "round": self.state["round"],
            }
            return observation, 0.0, self.terminated, self.truncated, self.info

        # Get the text response from the action
        text_response = action.get("text", "")
        logger.debug(f"[step] Text response: {text_response}")
        self.state["current_context"] = text_response

        # Process the greeting - validate it meets requirements
        logger.debug("[step] Validating greeting")
        self._validate_greeting(text_response)

        # Update round counter
        self.state["round"] += 1
        logger.debug(f"[step] Round updated to: {self.state['round']}")

        # Check if game should terminate
        self.terminated = True  # Hello game only has one round
        logger.debug(f"[step] Game terminated: {self.terminated}")

        # Prepare reward
        reward = 1.0 if self.state["success"] else 0.0
        logger.debug(f"[step] Reward: {reward}")

        # Prepare information
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

        # Comprehensive info dictionary for evaluation and logging
        self.info = {
            "message": result_message,
            "success": self.state["success"],
            "aborted": self.state["aborted"],
            "missing_words": self.state["missing_words"],
            # Required scores for the evaluation system
            "response_score": reward,
            "response_feedback": result_message,
            "episode_score": reward,  # In this simple game, episode score equals response score
        }
        logger.debug(f"[step] Info dictionary: \n{format_json(self.info)}")

        # Prepare observation
        observation = {
            "context": self.state["current_context"],
            "success": self.state["success"],
            "round": self.state["round"],
        }
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[step] Observation: \n{format_json(observation)}")

        # Update the observation for the player
        self.set_observation_for(player, observation["context"])
        logger.debug(
            f"[step] Updated observation for player: {player.name if hasattr(player, 'name') else 'unknown'}"
        )

        return observation, reward, self.terminated, self.truncated, self.info

    def _validate_greeting(self, greeting: str) -> None:
        """
        Validate if the greeting meets the requirements.

        Args:
            greeting: The greeting text to validate
        """
        logger.debug(f"[_validate_greeting] Validating greeting: {greeting}")

        # Check rule: utterance starts with key word
        if not greeting.startswith("GREET:"):
            logger.warning(
                f"[_validate_greeting] Greeting doesn't start with 'GREET:': {greeting}"
            )
            self.state["aborted"] = True
            self.state["success"] = False
            return
        else:
            self.state["aborted"] = False
            self.state["success"] = True

        # Check rule: required words are included
        greeting_lower = greeting.lower()
        # Remove punctuation
        greeting_clean = greeting_lower.translate(
            str.maketrans("", "", string.punctuation)
        )
        logger.debug(f"[_validate_greeting] Cleaned greeting: {greeting_clean}")

        missing_words = []
        for required_word in self.state["required_words"]:
            if required_word not in greeting_clean:
                logger.debug(
                    f"[_validate_greeting] Missing required word: {required_word}"
                )
                self.state["success"] = False
                missing_words.append(required_word)

        self.state["missing_words"] = missing_words

        if not missing_words:
            logger.info("[_validate_greeting] All required words found in greeting")
        else:
            logger.warning(
                f"[_validate_greeting] Missing words in greeting: {missing_words}"
            )

    def set_observation_for(self, player: Player, content: str, **extras) -> None:
        """
        Set the observation context for a specific player.

        Args:
            player: The player to set the observation for
            content: The text content for the observation
            extras: Additional observation parameters
        """
        player_name = player.name if hasattr(player, "name") else str(player)

        # Create observation with role and content structure
        observation = {"role": "user", "content": content, **extras}

        # Store in player_observations dictionary
        self.player_observations[player_name] = observation

        logger.info(f"[set_observation_for] Set observation for player: {player_name}")

    def observe_for(self, player: Player) -> Dict[str, Any]:
        """
        Get the current observation for a specific player.

        Args:
            player: The player to get the observation for

        Returns:
            The observation for the player
        """
        player_name = player.name if hasattr(player, "name") else str(player)
        logger.debug(f"[observe_for] Getting observation for player: {player_name}")

        if player_name not in self.player_observations:
            # If no observation is set, create a default one
            logger.warning(
                f"[observe_for] No observation found for player: {player_name}. Creating default."
            )
            self.player_observations[player_name] = {
                "role": "user",
                "content": self.state.get("prompt", "No prompt available."),
            }

        observation = self.player_observations[player_name]
        logger.debug(f"[observe_for] Observation for {player_name}: {observation}")
        return observation

    def render(self) -> Dict[str, Any]:
        """
        Render the current state of the environment.

        Returns:
            A dictionary representation of the current state
        """
        render_state = {
            "target_name": self.state["target_name"],
            "prompt": self.state["prompt"],
            "success": self.state["success"],
            "missing_words": self.state["missing_words"],
            "aborted": self.state["aborted"],
            "round": self.state["round"],
        }
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[render] Current state: \n{format_json(render_state)}")
        return render_state

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current environment state.

        Returns:
            The current state dictionary
        """
        return self.state

    def legal_actions(self, player: Player) -> List[Dict[str, Any]]:
        """
        Get a list of legal actions in the current state.

        Args:
            player: The player to get legal actions for

        Returns:
            A list containing a single example action (since any greeting text is allowed)
        """
        logger.debug(
            f"[legal_actions] Getting legal actions for player: {player.name if hasattr(player, 'name') else 'unknown'}"
        )

        if self.terminated or self.truncated:
            logger.debug(
                "[legal_actions] Environment is terminated, no legal actions available"
            )
            return []

        # For HelloGame, any text response is valid
        actions = [{"action_type": 0, "text": "GREET: Hello, welcome!"}]
        logger.debug(f"[legal_actions] Legal actions: {actions}")
        return actions

    def get_info(self) -> Dict[str, Any]:
        """
        Get the current info dictionary.

        Returns:
            The current info dictionary
        """
        return self.info
