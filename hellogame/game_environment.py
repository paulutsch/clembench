"""
Hello Game Environment - implements the GameEnvironment interface for the Hello Game.
"""

import logging
import string
from typing import Any, Dict, List, Optional, Tuple, Union

from gymnasium import spaces

from world_environments.game_environment import GameEnvironment

logger = logging.getLogger(__name__)

# Configure logger to print to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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
            config: Configuration dictionary that can include:
                - target_name: The name of the person to be greeted
                - language: The language for the greeting
                - prompt: The prompt for the greeter
        """
        super().__init__(config)
        logger.info("HelloGameEnvironment initialized successfully")
        # Define simple action and observation spaces
        # For Hello Game, both are dictionaries
        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(1),  # Only one action type: greet
                "text": spaces.Text(1000),  # Text with max 1000 chars
            }
        )

        self.observation_space = spaces.Dict(
            {
                "context": spaces.Text(5000),
                "success": spaces.Discrete(2),  # 0: False, 1: True
                "round": spaces.Discrete(10),  # Max 10 rounds (though usually just 1)
            }
        )

        # Set initial state
        self.reset()

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Returns:
            Tuple containing:
                - Initial observation dictionary
                - Information dictionary
        """
        # Get configuration values or use defaults
        target_name = self.config.get("target_name", "User")
        language = self.config.get("language", "en")
        prompt = self.config.get("prompt", "")

        # Set the required words for a successful greeting
        required_words = ["welcome", "hello", target_name.lower()]

        # Initialize state
        self.state = {
            "target_name": target_name,
            "language": language,
            "prompt": prompt,
            "required_words": required_words,
            "missing_words": [],
            "success": True,  # Start assuming success
            "aborted": False,
            "round": 0,
            "current_context": prompt,
        }

        self.terminated = False
        self.truncated = False
        self.info = {"message": "Environment reset"}

        # Return initial observation and info
        observation = {
            "context": prompt,
            "success": 1,  # 1 for True, 0 for False
            "round": 0,
        }

        return observation, self.info

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment using the provided action.

        Args:
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
        if self.terminated or self.truncated:
            self.info = {"warning": "Environment already terminated"}
            observation = {
                "context": self.state["current_context"],
                "success": 1 if self.state["success"] else 0,
                "round": self.state["round"],
            }
            return observation, 0.0, self.terminated, self.truncated, self.info

        # Get the text response from the action
        text_response = action.get("text", "")
        self.state["current_context"] = text_response

        # Process the greeting - validate it meets requirements
        self._validate_greeting(text_response)

        # Update round counter
        self.state["round"] += 1

        # Check if game should terminate
        self.terminated = True  # Hello game only has one round

        # Prepare reward
        reward = 1.0 if self.state["success"] else 0.0

        # Prepare information
        if self.state["aborted"]:
            result_message = "invalid format: abort game"
        elif self.state["success"]:
            result_message = "greeting successful: end game"
        else:
            missing = ",".join(self.state["missing_words"])
            result_message = f"greeting failed: missing words=[{missing}]"

        self.info = {
            "message": result_message,
            "success": self.state["success"],
            "aborted": self.state["aborted"],
            "missing_words": self.state["missing_words"],
        }

        # Prepare observation
        observation = {
            "context": self.state["current_context"],
            "success": 1 if self.state["success"] else 0,
            "round": self.state["round"],
        }

        return observation, reward, self.terminated, self.truncated, self.info

    def _validate_greeting(self, greeting: str) -> None:
        """
        Validate if the greeting meets the requirements.

        Args:
            greeting: The greeting text to validate
        """
        # Check rule: utterance starts with key word
        if not greeting.startswith("GREET:"):
            self.state["aborted"] = True
            self.state["success"] = False
            return

        # Check rule: required words are included
        greeting_lower = greeting.lower()
        # Remove punctuation
        greeting_clean = greeting_lower.translate(
            str.maketrans("", "", string.punctuation)
        )

        missing_words = []
        for required_word in self.state["required_words"]:
            if required_word not in greeting_clean:
                self.state["success"] = False
                missing_words.append(required_word)

        self.state["missing_words"] = missing_words

    def legal_actions(self) -> List[Dict[str, Any]]:
        """
        Get a list of legal actions in the current state.

        Returns:
            A list containing a single example action (since any greeting text is allowed)
        """
        if self.terminated or self.truncated:
            return []

        # For HelloGame, any text response is valid
        return [{"action_type": 0, "text": "GREET: Hello, welcome!"}]

    def render(self) -> Dict[str, Any]:
        """
        Render the current state of the environment.

        Returns:
            A dictionary representation of the current state
        """
        return {
            "target_name": self.state["target_name"],
            "prompt": self.state["prompt"],
            "success": self.state["success"],
            "missing_words": self.state["missing_words"],
            "aborted": self.state["aborted"],
            "round": self.state["round"],
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current environment state.

        Returns:
            The current state dictionary
        """
        return self.state
