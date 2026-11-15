from connect4.agents import RandomAgent, MCTSAgent
from connect4.experiments import play_game

print(play_game(RandomAgent(42), MCTSAgent(500)))