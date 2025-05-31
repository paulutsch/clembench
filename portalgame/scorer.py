from typing import Any, Dict

from clemcore.clemgame import GameScorer
from clemcore.clemgame.metrics import BENCH_SCORE
from clemcore.utils.logger import setup_logger

logger = setup_logger(__name__)


class PortalGameScorer(GameScorer):
    """Scorer for the Portal game."""

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        """Compute scores for the episode."""
        # Get the final state from the last turn
        last_turn = episode_interactions["turns"][-1]
        for event in last_turn:
            if event["type"] == "state":
                state = event["content"]
                break

        # Compute success rate
        success_rate = 1.0 if state["success"] else 0.0

        # Compute average moves
        avg_moves = state["moves"]

        # Store scores
        self.scores[BENCH_SCORE] = success_rate
        self.scores["avg_moves"] = avg_moves
