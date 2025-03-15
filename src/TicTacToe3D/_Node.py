import numpy as np
import math

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # État du jeu
        self.parent = parent
        self.action = action  # Action menant à cet état
        self.children = []
        self.visits = 0
        self.value = 0  # Moyenne des récompenses

    def is_fully_expanded(self):
        return len(self.children) == len(self.get_possible_actions())

    def best_child(self, exploration_weight=1.0):
        return max(self.children, key=lambda child: (child.value / (child.visits + 1e-6)) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6)))

    def get_possible_actions(self):
        # J'ai cassé un truc ici, plus aucune action n'est possible et je peux plus ctrl+z jusqu'a la version anterieure...
        print(f"État actuel du jeu dans get_possible_actions: {self.state}")
        empty_cells = np.argwhere(self.state == 0)
        print(f"Cases vides détectées: {empty_cells}")  # Afficher les cases vides trouvées
        possible_actions = [tuple(cell) for cell in empty_cells]
        print(f"Actions possibles: {possible_actions}")
        return possible_actions