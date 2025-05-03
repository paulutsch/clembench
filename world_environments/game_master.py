"""
GameMaster is a class that implements the game master for a game.

It is almost an exact copy of the GameMaster class in clemcore.

The essential difference is that it includes and interacts with a GameEnvironment as an attribute.

Eventually, these local GameMaster and DialogueGameMaster could replace the ones in clemcore, but for now we will keep them separate to avoid breaking changes.
"""

"""
Steps taken when "playing a game":

1. GameMaster.setup()
    1. GameMaster._on_setup()
2. GameMaster.play()
    1. GameMaster._play()
    2. GameMaster._on_valid_player_response()
"""

import collections
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

from clemcore import backends
from clemcore.clemgame import GameResourceLocator
from clemcore.clemgame.player import Player
from clemcore.clemgame.recorder import NoopGameRecorder

# Import the base environment class
from .game_environment import GameEnvironment

logger = logging.getLogger(__name__)

# Define type aliases for action and observation types
# These will be used in place of concrete classes
GameObservation = TypeVar("GameObservation")
GameAction = TypeVar("GameAction")


# changes in the new verion of GameMaster:
# - added a new attribute game_environment
# - added a new method create_action_from_response
class GameMaster(ABC):
    """Base class to contain game-specific functionality.

    A GameMaster (sub-)class

    - prepares a concrete game instance
    - plays an episode of a game instance
    - records a game episode
    - evaluates the game episode records
    - builds the interaction transcripts

    This new version of the GameMaster delegates state management to the GameEnvironment.
    """

    # edited this from original GameMaster class, such that it includes a game_environment
    def __init__(
        self,
        name: str,
        path: str,
        experiment: Dict,
        player_models: List[backends.Model],
        game_environment: GameEnvironment,
    ):
        """
        Args:
            name: The name of the game (as specified in game_registry).
            path: Path to the game (as specified in game_registry).
            experiment: The parameter of the experiment, that is, parameters that are the same for all game instances.
            player_models: Player models to use for one or two players.
            game_environment: The game environment that contains and manages the game state.
        """
        self.game_name = name
        self.experiment: Dict = experiment
        self.player_models: List[backends.Model] = player_models
        self._game_recorder = NoopGameRecorder()
        self.game_resources = GameResourceLocator(
            name, path
        )  # could be obsolete, when all info is in the instances
        self.game_environment = game_environment

    # took this without change from original GameMaster class
    @property
    def game_recorder(self):
        return self._game_recorder

    # took this without change from original GameMaster class
    @game_recorder.setter
    def game_recorder(self, game_recorder):
        self._game_recorder = game_recorder

    # this is a new method
    @abstractmethod
    def create_action_from_response(
        self, response: str, action_type: str = "text"
    ) -> Any:
        """Create an action from a player's response.

        Args:
            response: The textual response from the player
            action_type: The type of action to create

        Returns:
            An action object suitable for the environment
        """
        raise NotImplementedError("Subclasses must implement this method")

    # took this without change from original GameMaster class
    def load_json(self, file_path: Union[str, Path]):
        return self.game_resources.load_json(file_path)

    # took this without change from original GameMaster class
    def load_template(self, file_path: Union[str, Path]):
        return self.game_resources.load_template(file_path)

    # took this without change from original GameMaster class
    def log_to_self(self, type_: str, value: Any):
        """Logs an action of the passed type from GM to GM.
        This is a logging method, and will not add anything to the conversation history.
        Args:
            type_: The type of the action to be logged.
            value: The content value of the action to be logged. Must be JSON serializable.
        """
        self._game_recorder.log_event("GM", "GM", {"type": type_, "content": value})

    # took this without change from original GameMaster class
    def log_key(self, key: str, value: Any):
        self._game_recorder.log_key(key, value)

    # took this without change from original GameMaster class
    def log_players(self, players_dict):
        self._game_recorder.log_players(players_dict)

    # took this without change from original GameMaster class
    def log_next_round(self):
        self._game_recorder.log_next_round()

    # took this without change from original GameMaster class
    def log_event(self, from_, to, action):
        self._game_recorder.log_event(from_, to, action)

    # took this without change from original GameMaster class
    def store_records(self, results_root, dialogue_pair_desc, game_record_dir):
        self._game_recorder.store_records(
            results_root, dialogue_pair_desc, game_record_dir
        )

    # took this without change from original GameMaster class
    @abstractmethod
    def setup(self, **kwargs):
        """Load resources and prepare everything to play the game.
        Needs to log the players dictionary via self.log_players(players_dict).
        Called by the game's GameBenchmark run method for each game instance.
        Args:
            kwargs: Keyword arguments used to set up the GameMaster instance.
        """
        pass

    # took this without change from original GameMaster class
    @abstractmethod
    def play(self) -> None:
        """Play the game (multiple turns of a specific game instance)."""
        pass


class DialogueGameMaster(GameMaster):
    """Extended GameMaster, implementing turns as described in the clembench paper.
    Has most logging and gameplay procedures implemented, including convenient logging methods.

    This version integrates a GameEnvironment as self-contained object for state management.
    """

    # edited this from original GameMaster class, such that it includes a game_environment, removes the context_for_player attribute (delegating different player contexts to game_environment)
    def __init__(
        self,
        name: str,
        path: str,
        experiment: dict,
        player_models: List[backends.Model],
        game_environment: GameEnvironment,
    ):
        """
        Args:
            name: The name of the game (as specified in game_registry).
            path: Path to the game (as specified in game_registry).
            experiment: The experiment (set of instances) to use.
            player_models: Player models to use for one or two players.
            game_environment: The environment that maintains the game state.
        """
        super().__init__(name, path, experiment, player_models, game_environment)

        # set players
        self.players_by_names: Dict[str, Player] = collections.OrderedDict()

        # TODO: remove this attribute and delegate context management to game_environment
        self.context_for_player: Dict[str, Dict] = (
            dict()
        )  # context entries look like {"role":"user", "content": ...}
        self.current_player: Optional[Player] = None
        self.current_player_idx: int = 0

        self.current_round: int = 0
        self.info = {}

    # took this without change from original GameMaster class
    def __setstate__(self, state):
        self.__dict__.update(state)
        for (
            player
        ) in (
            self.players_by_names.values()
        ):  # sync game recorders (not copied in Player)
            player.game_recorder = self.game_recorder

    # took this without change from original GameMaster class
    def get_players(self) -> List[Player]:
        """Get a list of the players.
        Returns:
            List of Player instances in the order they are added.
        """
        return list(self.players_by_names.values())

    # edited this from original GameMaster class, such that it uses the game_environment for state management
    def add_player(
        self,
        player: Player,
        initial_prompt: Optional[Union[str, Dict]] = None,
        initial_context: Optional[Union[str, Dict]] = None,
    ):
        """Add a player to the game. The same player cannot be added twice.
        The player identity is determined by the player's name.

        Important: During gameplay, the players will be called in the same order as added to the game master!

        Args:
            player: The player to be added to the game. The player's name must be unique.
            initial_prompt: The initial prompt given to the player (optional). See Player for more details.
            initial_context: A context to be immediately set for the player (optional). This is useful for initial
                            prompts that are supposed to be handled as the first context, for example, when adding
                            the other player's response to the prompt is not necessary, but the player is supposed
                            to directly react to the initial prompt. Alternatively, overwrite on_before_game() and
                            use set_context_for(player) to set the player context.
        """
        player.game_recorder = (
            self.game_recorder
        )  # player should record to the same interaction log

        # set the initial prompt on the player if provided
        if initial_prompt is not None:
            setattr(player, "_initial_prompt", initial_prompt)

        player.name = (
            f"Player {len(self.players_by_names) + 1} ({player.__class__.__name__})"
        )
        if player.name in self.players_by_names:
            raise ValueError(
                f"Player names must be unique, "
                f"but there is already a player registered with name '{player.name}'."
            )
        self.players_by_names[player.name] = player
        if initial_context is not None:
            assert isinstance(
                initial_context, (str, dict)
            ), f"The initial context must be a str or dict, but is {type(initial_context)}"
            if isinstance(initial_context, dict):
                assert (
                    "content" in initial_context
                ), "The initial context requires a content entry"
                extras = {
                    k: v
                    for k, v in initial_context.items()
                    if k not in ["role", "content"]
                }
                self.game_environment.set_observation_for(
                    player, initial_context["content"], **extras
                )
            else:
                self.game_environment.set_observation_for(player, initial_context)

    # edited this from original GameMaster class, such that it resets the game_environment
    def setup(self, **kwargs):
        """Load resources and prepare everything to play the game.
        Needs to log the players dictionary via self.log_players(players_dict).
        Intended to be left as-is by inheriting classes. Implement game-specific setup functionality in the _on_setup
        method.
        Called by the game's GameBenchmark run method for each game instance.
        Args:
            kwargs: Keyword arguments used to set up the GameMaster instance. This is usually a game instance object
                read from the game's instances.json.
        """
        self._on_setup(**kwargs)
        # log players
        players_descriptions = collections.OrderedDict(
            GM=f"Game master for {self.game_name}"
        )
        for name, player in self.players_by_names.items():
            players_descriptions[name] = player.get_description()
        self.log_players(players_descriptions)

        # Initialize the current player
        if self.players_by_names:
            self.current_player = self.get_players()[self.current_player_idx]

        # call game hooks
        self._on_before_game()
        self._on_before_round()

    # took this without change from original GameMaster class
    @abstractmethod
    def _on_setup(self, **kwargs):
        """Method executed at the start of the default setup method.
        Template method: Must be implemented!
        Use add_player() here to add the players.
        Args:
            kwargs: Keyword arguments of the game instance. This is usually a game instance object
                read from the game's instances.json.
        """
        pass

    # edited this from original GameMaster class, such that it returns the game_environment's state
    def get_game_state(self):
        """Get the current game state from the environment."""
        return self.game_environment.get_state()

    # took this without change from original GameMaster class
    def get_current_player(self) -> Optional[Player]:
        return self.current_player

    # took this without change from original GameMaster class
    # def set_context_for(self, player: Player, content: str, **extras):
    #     """
    #     Set the context for the specified Player. The player will be prompted with the context on its next turn.

    #     The context always has a 'role' and 'content' entry where the 'role' is always set to 'user'.
    #     Args:
    #         player: The player to set the context for.
    #         content: The text content to be added to the context.
    #         extras: Additional content to be merged into the context e.g. information about images
    #     """
    #     message = {"role": "user", "content": content}
    #     context = {**extras, **message}
    #     self.context_for_player[player.name] = context

    # took this without change from original GameMaster class
    # def get_context_for(self, player: Player) -> Dict:
    #     """Get the context for a player.

    #     Args:
    #         player: The player to get the context for

    #     Returns:
    #         The context for the player

    #     Raises:
    #         AssertionError: If the player is None or has no context set
    #     """
    #     assert (
    #         player.name in self.context_for_player
    #     ), f"No context set for {player.name}"
    #     context = self.context_for_player[player.name]
    #     assert "role" in context, f"Player context must have a 'role' entry"
    #     assert context["role"] == "user", f"Role of player context must be 'user'"
    #     assert "content" in context, f"Player context must have a 'content' entry"
    #     return context

    # edited this from original GameMaster class, such that it uses the game_environment for state management
    def play(self) -> None:
        """
        Main play loop method. This method is called to run the game for benchmarking.
        This implementation uses the game environment for state management.
        """
        if self.current_player is None:
            logger.warning("No current player set, ending game.")
            return

        done = False
        while not done:
            self._on_before_round()
            observation = self.game_environment.observe_for(self.current_player)

            # Get context from the environment
            context = {
                "role": "user",
                "content": observation["content"],
            }

            # Get player's response
            response = self.current_player(context)

            # Validate and parse the response
            if not self._validate_player_response(self.current_player, response):
                # Handle invalid responses based on game rules
                done = not self._does_game_proceed()
                if done:
                    self._on_after_game()
                break

            # Parse valid response
            parsed_response = self._parse_response(self.current_player, response)

            # Custom game logic for processing the response
            self._on_valid_player_response(self.current_player, parsed_response)

            # Create action from response and step the environment
            action = self.create_action_from_response(parsed_response)
            next_observation, reward, terminated, truncated, info = (
                self.game_environment.step(self.current_player, action)
            )

            # Update game info
            self.info.update(info)

            # Check if the game should continue
            done = terminated or truncated or not self._does_game_proceed()

            # Handle turn passing and round transitions
            if not done and self._should_pass_turn():
                self.current_player = self._next_player()
                if self._start_next_round():
                    self._on_after_round()
                    self.current_round += 1
                    self.log_next_round()

                # Get observation for the next player
                observation = self.game_environment.observe_for(self.current_player)
            else:
                observation = next_observation

            # Handle game end
            if done:
                self._on_after_game()
                info["episode_score"] = self.compute_episode_score()
                self.info.update(info)
                self.log_key("episode_score", info["episode_score"])

    def _process_step_result(
        self, response: str, context: Any, terminated: bool, truncated: bool
    ) -> Tuple[bool, Dict]:
        """
        This method is deprecated and will be removed in a future version.
        Game state processing now happens directly in the play() method.

        Returns a default result for compatibility.
        """
        logger.warning(
            "_process_step_result is deprecated, logic moved to play() method"
        )
        done = terminated or truncated or not self._does_game_proceed()
        return done, {}

    def _next_player(self) -> Player:
        """
        Subclasses can overwrite this method to determine the next player after a player's turn has been passed.

        Default: The gamer master passes the turn to the next player in the player list (order as added).
        Starting again with the first player, when all players have had their turn(s).

        :return: the next (current) player
        """
        players = self.get_players()
        if not players:
            raise ValueError("No players have been added to the game")

        self.current_player_idx = (self.current_player_idx + 1) % len(players)
        return players[self.current_player_idx]

    def _start_next_round(self) -> bool:
        """
        Subclasses can overwrite this method to specify when a next round should start after a player's turn is passed.

        Default: Start next round when we cycled through the whole list i.e. it is again the first player's turn.

        :return: True, when to start a new round
        """
        return self.current_player_idx == 0

    def __prepare_next_round(self):
        self.log_next_round()  # add record entry for player turns
        self._on_before_round()

    def get_response_feedback(self, response: str, context: Dict):
        """
        Optional.
        :param response: The response of the current player.
        :param context: The context given to the current player to generate the response for.
        :return: a verbal feedback about the player's response given the context
        """
        return None

    def compute_response_score(self, response: str, context: Dict):
        """
        Mandatory.
        :param response: The response of the current player.
        :param context: The context given to the current player to generate the response for.
        :return: the performance score for a player's response given the context
        """
        return 0

    def compute_episode_score(self):
        """
        Mandatory.
        :return: the performance of the agent over the whole episode
        """
        return 0

    def _should_pass_turn(self):
        """
        Whether to pass the turn to the next player. Otherwise, the current player keeps playing
        based on the context set via set_player_context(player, content).
        """
        return True

    @abstractmethod
    def _on_valid_player_response(self, player: Player, parsed_response: str):
        """
        Method executed after a player response has been parsed and validated.

        Set the response as the context for the other player (if necessary).

        You could also set a new context for the current player and give the player
        another turn by letting _should_pass_turn() return False.

        To do this use the method set_context_for(player, response).
        Args:
            player: The Player instance that produced the response (or has been modified by the GM).
            parsed_response: The parsed and valid response of the current player.
        """
        pass

    @abstractmethod
    def _validate_player_response(self, player: Player, response: str) -> bool:
        """
        Decide if a player response is valid. An invalid response breaks the game rules and might end the game.

        Note: If the response is not valid, then _parse_response() and on_valid_player_response() will not be called.

        However, game developers can decide to give the player another turn by letting _should_pass_turn() return False.

        Args:
            player: The player that gave the response.
            response: The response of the current player.
        Returns:
            True, if the response is fine. Otherwise, False.
        """
        raise NotImplementedError

    def _parse_response(self, player: Player, response: str) -> str:
        """Decide if a response utterance should be modified and apply modifications.

        Hook: Modify this method for game-specific functionality.

        Args:
            player: The Player instance that produced the response. Intended to allow for individual handling of
                different players.
            response: The response of the current player.
        Returns:
            The parsed response
        """
        return response

    @abstractmethod
    def _does_game_proceed(self) -> bool:
        """Check if game should proceed.

        Template method: Must be implemented!

        This method is used to determine if a game should continue or be stopped. Both successful completion of the game
        and game-ending failures should lead to this method returning False.
        Returns:
            A bool, True if game continues, False if game should stop.
        """
        raise NotImplementedError

    def _on_before_round(self):
        """Executed in the play loop before a new round of gameplay starts.

        Hook: Modify this method for game-specific functionality.
        """
        pass

    def _on_after_round(self):
        """Executed in the play loop after a round of gameply finished i.e. _start_next_round() resolves to True.

        Hook: Modify this method for game-specific functionality.
        """
        pass

    def _on_before_game(self):
        """Executed once at the start, before entering the play loop.

        Hook: Modify this method for game-specific functionality.

        Adding the initial prompt to the dialogue history with this method is recommended.
        """
        logger.info("[_on_before_game] Game starting")
        if self.current_player is None:
            raise ValueError("No current player set, ending game.")

        # Get the prompt from the environment state
        self.game_environment.reset()
        prompt = self.game_environment.state["prompt"]
        logger.debug(f"[_on_before_game] Initial prompt from environment: {prompt}")

        # Set the initial observation for the greeter player using the game environment
        self.game_environment.set_observation_for(self.current_player, prompt)
        logger.debug(f"[_on_before_game] Set initial observation for greeter player")

    def _on_after_game(self):
        """Executed once at the end, after exiting the play loop.

        Hook: Modify this method for game-specific functionality.

        This method is useful to process and log/record overall game results.
        """
        pass
