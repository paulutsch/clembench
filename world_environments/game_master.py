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
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from clemcore import backends
from clemcore.clemgame import GameResourceLocator
from clemcore.clemgame.player import Player
from clemcore.clemgame.recorder import NoopGameRecorder

from _logger import format_json, setup_logger

from .game_environment import GameEnvironment

logger = setup_logger(__name__)


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
    def create_action_from_response(self, response: str) -> Dict[str, Any]:
        """Create an action from a player's response.

        Default: return action

        Args:
            response: The textual response from the player
            action_type: The type of action to create

        Returns:
            The action to be taken, in the form of a dictionary with (at least) the following keys:
                - action_type: The type of action to create
                - body: The body of the action
        """
        raise NotImplementedError

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
        raise NotImplementedError

    # took this without change from original GameMaster class
    @abstractmethod
    def play(self) -> None:
        """Play the game (multiple turns of a specific game instance)."""
        raise NotImplementedError


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

        self.current_player: Optional[Player] = None
        self.current_player_idx: int = 0

        self.current_round: int = 0

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
        # initial_content: str = "",
    ):
        """Add a player to the game. The same player cannot be added twice.
        The player identity is determined by the player's name.

        Important: During gameplay, the players will be called in the same order as added to the game master!

        Args:
            player: The player to be added to the game. The player's name must be unique.
        """
        player.game_recorder = (
            self.game_recorder
        )  # player should record to the same interaction log

        player.name = (
            f"Player {len(self.players_by_names) + 1} ({player.__class__.__name__})"
        )
        if player.name in self.players_by_names:
            raise ValueError(
                f"Player names must be unique, "
                f"but there is already a player registered with name '{player.name}'."
            )
        self.players_by_names[player.name] = player

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
        players_descriptions = collections.OrderedDict(
            GM=f"Game master for {self.game_name}"
        )
        for name, player in self.players_by_names.items():
            players_descriptions[name] = player.get_description()
        self.log_players(players_descriptions)

        if self.players_by_names:
            self.current_player = self.get_players()[self.current_player_idx]

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
        raise NotImplementedError

    # edited this from original GameMaster class, such that it returns the game_environment's state
    def get_environment_state(self):
        """Get the current game state from the environment."""
        return self.game_environment.state

    # took this without change from original GameMaster class
    def get_current_player(self) -> Optional[Player]:
        return self.current_player

    # edited this from original GameMaster class, such that it uses the game_environment for state management
    def play(self) -> None:
        """
        Main play loop method. This method is called to run the game for benchmarking.
        This implementation uses the game environment for state management.
        """
        logger.debug(
            f"[_play] Starting game with current player: {self.current_player}"
        )
        if self.current_player is None:
            logger.warning("No current player set, ending game.")
            return

        self.game_environment.reset()

        terminated = False
        while not terminated:
            self._on_before_round()
            observation = self.game_environment.get_observation_space(
                self.current_player
            )
            logger.debug(f"[_play] Observation: {observation}")
            context = {
                "role": "user",
                "content": observation["content"],
            }

            response = self.current_player(context)

            # TODO: now that we have _validate_action in the game_environment, do we still need this?
            if not self._validate_player_response(self.current_player, response):
                logger.warning(
                    f"[_play] Player {self.current_player.name} response is invalid"
                )
                terminated = self._should_terminate_on_invalid_response()
                if terminated:
                    self._on_after_game()
                    break

            parsed_response = self._parse_response(self.current_player, response)
            logger.debug(f"[_play] Parsed response: {parsed_response}")
            self._on_valid_player_response(self.current_player, parsed_response)

            action = self.create_action_from_response(parsed_response)
            logger.debug(f"[_play] Action: {action}")
            next_observation, score, terminated = self.game_environment.step(
                self.current_player, action
            )

            if terminated:
                self._on_after_game()

            if not terminated and self._should_pass_turn():
                # next player should now play
                self.current_player = self._next_player()
                # if the current player is the first player, we are at the end of a round (default behavior in _start_next_round)
                if self._start_next_round():
                    self._on_after_round()
                    self.current_round += 1
                    self.log_next_round()

                observation = self.game_environment.get_observation_space(
                    self.current_player
                )
            else:
                observation = next_observation

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

    @abstractmethod
    def compute_response_score(self, response: str, context: Dict):
        """
        Mandatory.
        :param response: The response of the current player.
        :param context: The context given to the current player to generate the response for.
        :return: the performance score for a player's response given the context
        """
        raise NotImplementedError

    @abstractmethod
    def compute_episode_score(self):
        """
        Mandatory.
        :return: the performance of the agent over the whole episode
        """
        raise NotImplementedError

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
        raise NotImplementedError

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

    def create_action_from_response(self, response: str) -> Dict[str, Any]:
        """Create an action from a player's response.

        Default: return action

        Args:
            response: The textual response from the player
            action_type: The type of action to create

        Returns:
            {"action_type": "verbal_response", "body": response}
        """
        return {"action_type": "verbal_response", "body": response}

    def _should_terminate_on_invalid_response(self) -> bool:
        """
        Decide if the game should terminate on an invalid response.

        Default: False
        """
        return False

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
        """
        pass

    def _on_after_game(self):
        """Executed once at the end, after exiting the play loop.

        Hook: Modify this method for game-specific functionality.
        """
        self._add_logs_to_episode_scores()

    def _add_logs_to_episode_scores(self):
        """Executed once at the end, after exiting the play loop.

        Hook: Modify this method for game-specific functionality.

        This method is useful to process and log/record overall game results.
        """
        logger.info("[_on_after_game] Game completed, processing final state")

        final_state = self.game_environment.state

        logger.debug(f"Final game state: \n{format_json(final_state)}")

        for key, value in final_state.items():
            self.log_key(key, value)

        self.log_key("episode_score", self.compute_episode_score())

        logger.info(f"[_on_after_game] Game completed")
