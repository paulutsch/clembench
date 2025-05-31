"""
Generate instances for the game.

Creates files in ./in
"""

import logging
import os

from clemcore.clemgame import GameInstanceGenerator
from tqdm import tqdm

logger = logging.getLogger(__name__)

LANGUAGE = "en"


class PortalGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self):
        experiment = self.add_experiment(f"greet_{LANGUAGE}")
        experiment["language"] = LANGUAGE  # experiment parameters

        names = self.load_file("resources/names", file_ending=".txt").split("\n")

        prompt = self.load_template("resources/initial_prompts/prompt")

        for game_id in tqdm(range(len(names))):
            target_name = names[game_id]

            instance_prompt = prompt.replace("$NAME$", target_name)

            game_instance = self.add_game_instance(experiment, game_id)
            game_instance["prompt"] = instance_prompt  # game parameters
            game_instance["target_name"] = target_name  # game parameters


if __name__ == "__main__":
    PortalGameInstanceGenerator().generate()
