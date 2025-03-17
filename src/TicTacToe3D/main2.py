import gymnasium as gym
import ale_py
import math
import random
import copy
import numpy as np

# Noeud du MCTS
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visits = 0
        self.value = 0.0    # Valeur moyenne estimee (somme recompenses / nb visites)

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, possible_actions):
        for action in possible_actions:
            if action not in self.children:
                self.children[action] = MCTSNode(state=None, parent=self)

    def best_child(self, c_param=1.4):
        best_score = float('-inf')
        best_action = None
        best_node = None

        for action, child in self.children.items():
            # Valeur moyenne du noeud
            q_value = child.value
            # Bonus d'exploration
            u_value = c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-9))
            
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_node = child
                best_action = action

        return best_action, best_node

    def update(self, reward):
        self.visits += 1
        # MAJ vers la moyenne
        self.value += (reward - self.value) / self.visits

# Structure principale du MCTS
def mcts_search(root, env, n_simulations=1000):
    # Lance un cycle de MCTS à partir d'un noeud racine.
    for _ in range(n_simulations):
        # Copier l'environnement actuel pour explorer
        env.unwrapped.restore_state(root.state)
        node_to_expand = expand(root, env)
        reward = simulate(env)
        #print("xxxxxxxxxxxxxxxxxxxxxx reward :", reward) # pb avec le reward qui reste a 0
        backpropagate(node_to_expand, reward)

    # Selectionner l'action la plus visitee
    best_action = None
    best_visits = -1
    for action, child in root.children.items():
        if child.visits > best_visits:
            best_visits = child.visits
            best_action = action #reste "0" ??? (et pas NONE, donc on selectionne bien un noeud...)
            #print("----------------------- best_action:", best_action)
    return best_action

def expand(node, env):
    #Parcours l'arbre, on s'arrête dès qu'on arrive sur une feuille et on l'expand.
    current_node = node
    done = False

    while True:
        # Si le noeud courant est une feuille : on l'expand puis on s'arrête.
        if current_node.is_leaf():
            possible_actions = env.action_space.available_actions() if hasattr(env.action_space, "available_actions") else range(env.action_space.n)
            current_node.expand(possible_actions)
            return current_node

        # Sinon : on descend dans l'arbre
        best_action, best_child_node = current_node.best_child(c_param=1.4)
        _, reward, done, _, _ = env.step(best_action)

        # MAJ de l'etat du noeud enfant
        best_child_node.state = env.unwrapped.clone_state() if hasattr(env.unwrapped, 'clone_state') else None

        if done:
            # Etat terminal pendant la descente => on renvoie quand même le noeud
            return best_child_node

        current_node = best_child_node

def simulate(env): # PB AVEC LE REWARD --------------------------------------------------------------------------------------
    # Effectue une simulation random jusqu'a la fin de partie
    done = False
    total_reward = 0.0
    while not done:
        possible_actions = env.action_space.available_actions() if hasattr(env.action_space, "available_actions") else range(env.action_space.n)
        action = random.choice(list(possible_actions))
        _, reward, done, _, _ = env.step(action)
        total_reward += reward
    total_reward += 1 #maj manuel quand la partie est finie, ne marche pas ??? total_reward reste a 0, mais "total_reward = x" donne bien x...
    return total_reward

def backpropagate(node, reward):
    # Retro-propagation de la recompense jusqu'a la racine
    current_node = node
    while current_node is not None:
        current_node.update(reward)
        current_node = current_node.parent

# Le Jeu (TicTacToe3D)
def main():
    # Initialisation de l'environnement reel
    env = gym.make("ALE/TicTacToe3D-v5", render_mode="human")
    observation, info = env.reset()

    # Initialisation de l'environnement d'exploration
    #env_sim = gym.make("ALE/TicTacToe3D-v5", render_mode="human") #pour visualiser l'exploration
    env_sim = gym.make("ALE/TicTacToe3D-v5")
    env_sim.reset()

    # Creation de la racine MCTS
    root = MCTSNode(state=env.unwrapped.clone_state())

    done = False
    tour = 0
    while not done:
        tour+=1
        print(tour)

        # Recherche avec le MCTS sur une COPIE de l'env
        #mettre plus que 5, mon pc est juste faible donc je met un nb faible pour tester
        #il est possible que le mouvement soit toujours "0" car il n'y a pas assez d'iteration pour en trouver un meilleur...
        action = mcts_search(root, env_sim, n_simulations=5)
        print("Action chosen:", action)
        
        # Execution de l'action dans l'environnement reel
        old_obs = observation
        observation, reward, done, _, info = env.step(action)
        print("Observation changed?", np.any(old_obs != observation))
        print("Reward:", reward)

        # MAJ de la racine (nouvel etat après action reelle)
        new_root = MCTSNode(state=env.unwrapped.clone_state(), parent=None)
        root = new_root

    print("Partie terminee")
    env.close()

if __name__ == "__main__":
    main()