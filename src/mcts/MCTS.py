import numpy as np
from collections import defaultdict
import math

from NODE import Node

class MCTS:
    def __init__(self, env, exploration_weight=math.sqrt(2)):
        self.env = env
        self.exploration_weight = exploration_weight
        
    def uct_select_child(self, node):
        """Select a child node using the UCT formula"""
        def uct_value(n):
            if n.visits == 0:
                return float('inf')
            return (n.wins / n.visits) + self.exploration_weight * math.sqrt(
                math.log(node.visits) / n.visits)
            
        return max(node.children, key=uct_value)
    
    def expand(self, node):
        """Expand the node by adding a child node"""
        if node.untried_actions is None:
            node.untried_actions = list(range(self.env.action_space.n))
            
        action = node.untried_actions.pop()
        
        # Clone environment state
        env_state = self.env.unwrapped.clone_state()
        observation, reward, terminated, truncated, _ = self.env.step(action)
        
        child_node = Node(
            state=observation,
            parent=node,
            action=action
        )
        node.children.append(child_node)
        
        return child_node


    def rollout(self, node):
        """Simulate a random game from the node's state"""
        current_state = node.state
        done = False
        total_reward = 0
        
        while not done:
            action = self.env.action_space.sample()  # Random policy for rollout
            observation, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            current_state = observation
            
        return total_reward


    def backpropagate(self, node, reward):
        """Update node statistics going up the tree"""
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent
            
    def get_best_action(self, root_state, num_simulations=1000):
        """Get the best action by running MCTS simulations"""
        root = Node(state=root_state)
        
        for _ in range(num_simulations):
            node = root
            
            # Selection
            while node.untried_actions is None or not node.untried_actions:
                if not node.children:
                    break
                node = self.uct_select_child(node)
                
            # Expansion
            if node.untried_actions:
                node = self.expand(node)
                
            # Rollout
            reward = self.rollout(node)
            
            # Backpropagation
            self.backpropagate(node, reward)
            
        # Return the action that led to the most visited child
        return max(root.children, key=lambda c: c.visits).action

