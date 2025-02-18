import math
import random
from Breakout.mcts_node_class import MCTSNode

class MCTS:
    def __init__(self, root):
        self.root = root

    def search(self, iterations=1000):
        for _ in range(iterations):
            node = self.tree_policy()
            reward = self.default_policy(node.state)
            self.backup(node, reward)
        return self.root.best_child(c_param=0.0).state

    def tree_policy(self):
        node = self.root
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child()
        return node

    def default_policy(self, state):
        while not state.is_terminal():
            state = state.take_action(random.choice(state.get_legal_actions()))
        return state.get_reward()

    def backup(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent