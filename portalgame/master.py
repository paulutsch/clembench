import random
import re
from typing import Dict, List

import numpy as np
from clemcore.backends import Model
from clemcore.clemgame import EnvGameMaster, GameBenchmark, GameScorer, GameSpec, Player
from clemcore.clemgame.metrics import (
    BENCH_SCORE,
    METRIC_ABORTED,
    METRIC_LOSE,
    METRIC_SUCCESS,
)

from portalgame.game_environment import PortalAction, PortalGameEnvironment


class PortalPlayer(Player):
    """Player for the Portal game."""

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, context: Dict) -> str:
        """Custom response for the player."""
        direction = random.choice(["n", "s", "e", "w"])
        return f"DIRECTION: {direction}"


class PortalGame(EnvGameMaster):
    """Game master for the Portal game."""

    def __init__(
        self,
        game_spec: GameSpec,
        experiment: Dict,
        player_models: List[Model],
    ):
        super().__init__(game_spec, experiment, player_models)

    def _on_setup(self, **game_instance):
        """
        Called during game setup. Configure game parameters and initialize players.

        Args:
            game_instance: Game instance parameters from instances.json
        """
        self.game_environment = PortalGameEnvironment(config=game_instance)

        for player in self.player_models:
            self.add_player(PortalPlayer(player))

        self.game_environment.reset()

    def _response_valid(
        self, player: Player, utterance: str
    ) -> bool:
        """
        Validate the player's response.

        Args:
            player: The player making the response
            utterance: The player's response

        Returns:
            bool: Whether the response is valid
        """
        try:
            # look for 'DIRECTION:' and extract the last non-whitespace char after it

            match = re.search(
                r"DIRECTION:\s*([nsewNSEW])", utterance, re.IGNORECASE | re.DOTALL
            )
            if match:
                direction = match.group(1).lower()
            else:
                # fallback: last non-whitespace char
                direction = utterance.strip()[-1].lower()
            return direction in ["n", "s", "e", "w"]
        except Exception:
            return False

    def _parse_action_from_response(self, response: str) -> PortalAction:
        """Create an action from a player's response.

        Args:
            response: The textual response from the player

        Returns:
            PortalAction: The parsed action
        """
        match = re.search(
            r"DIRECTION:\s*([nsewNSEW])", response, re.IGNORECASE | re.DOTALL
        )
        if match:
            direction = match.group(1).lower()
        else:
            direction = response.strip()[-1].lower()
        action: PortalAction = {
            "action_type": "default",
            "direction": direction,
        }
        return action


class PortalGameScorer(GameScorer):
    """Scorer for the Portal game."""

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_round_score(self, round_idx, round_events):
        self.log_round_score(
            round_idx,
            "Reward",
            round_events[-1]["action"]["content"],
        )

    def compute_episode_scores(self, interactions: Dict) -> None:
        """Compute episode-level scores for the Portal game.

        Args:
            interactions: Dict containing the logged episode's interactions.
        """
        aborted = interactions.get(METRIC_ABORTED, False)
        success = interactions.get(METRIC_SUCCESS, False)

        shortest_path = self.game_instance["shortest_path"]
        moves = sum(interactions["Request Count"])
        efficiency = shortest_path / moves

        if aborted:
            bench_score = np.nan
        elif not success:
            bench_score = 0.0
        else:
            bench_score = 100.0

            # for example: the more efficient the player is, the higher the bench score
            bench_score = bench_score - (50 * (1 - efficiency))

        self.log_episode_score("Efficiency", efficiency)
        self.log_episode_score(BENCH_SCORE, bench_score)


class PortalGameBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> EnvGameMaster:
        return PortalGame(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return PortalGameScorer(self.game_name, experiment, game_instance)
