import random
import numpy as np
import torch
import torch.nn as nn
from treys import Card, Deck, Evaluator  
import sys# For hand evaluation
sys.path.append("../src")
  # Define the Poker environment


  # Define the DQN neural network
class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.out(x)

class PokerEnvCLI:
      def __init__(self):
          self.reset()
          
      def reset(self):
          self.deck = Deck()
          self.evaluator = Evaluator()
          # Reset game state
          self.player_hand = [self.deck.draw(1)[0], self.deck.draw(1)[0]]
          self.bot_hand = [self.deck.draw(1)[0], self.deck.draw(1)[0]]
          self.community_cards = []
          self.pot = 0
          self.player_stack = 1000
          self.bot_stack = 1000
          self.current_bet = 0
          self.betting_round = 0  # 0: pre-flop, 1: flop, 2: turn, 3: river
          self.done = False
          self.bet_history = []
          self.last_action = None
          self.player_current_bet = 0
          self.bot_current_bet = 0
          self.reward = 0  # Initialize reward
          return self.get_state()
      
      def step(self, player_action):
          if self.done:
              raise Exception("Game is over. Call reset().")
          
          # Initialize reward
          self.reward = 0

          # Process player's action
          self.process_player_action(player_action)
          if self.done:
              return self.get_state(), self.reward, self.done, {}
          
          # Bot's turn
          bot_action = self.bot_decision()
          bot_action_str = self.action_mapping(bot_action)
          print(f"Bot action: {bot_action_str}")
          self.process_bot_action(bot_action)
          if self.done:
              return self.get_state(), self.reward, self.done, {}
          
          # Check for game end conditions
          if self.betting_round > 3:
              # Showdown
              self.resolve_showdown()
              self.done = True
              return self.get_state(), self.reward, self.done, {}
          
          return self.get_state(), self.reward, self.done, {}
      
      def process_player_action(self, action):
          action = action.lower()
          if action == 'fold':
              self.bot_stack += self.pot
              print("You folded. Bot wins the pot.")
              self.reward = -1
              self.done = True
          elif action == 'check' or action == 'call' or action == 'check/call':
              call_amount = self.current_bet - self.player_current_bet
              self.player_stack -= call_amount
              self.pot += call_amount
              self.player_current_bet += call_amount
              if self.player_current_bet == self.current_bet and self.bot_current_bet == self.current_bet:
                  self.advance_round()
          elif action.startswith('bet') or action.startswith('raise'):
              try:
                  amount = int(action.split()[1])
                  self.player_stack -= amount
                  self.pot += amount
                  self.player_current_bet += amount
                  self.current_bet = self.player_current_bet
              except:
                  print("Invalid bet amount.")
                  return
          else:
              print("Invalid action.")
              return
      
      def process_bot_action(self, action_idx):
          action = self.action_mapping(action_idx)
          if action == 'fold':
              self.player_stack += self.pot
              print("Bot folded. You win the pot!")
              self.reward = 1
              self.done = True
          elif action == 'check/call':
              call_amount = self.current_bet - self.bot_current_bet
              self.bot_stack -= call_amount
              self.pot += call_amount
              self.bot_current_bet += call_amount
              if self.player_current_bet == self.current_bet and self.bot_current_bet == self.current_bet:
                  self.advance_round()
          elif action.startswith('raise'):
              amount = self.get_bot_raise_amount(action)
              self.bot_stack -= amount
              self.pot += amount
              self.bot_current_bet += amount
              self.current_bet = self.bot_current_bet
          else:
              print("Bot made an invalid action.")
              self.reward = 1  # You win if the bot makes an invalid action
              self.done = True
      
      def advance_round(self):
          self.betting_round += 1
          self.player_current_bet = 0
          self.bot_current_bet = 0
          self.current_bet = 0
          if self.betting_round == 1:
              # Flop
              self.community_cards.extend(self.deck.draw(3))
          elif self.betting_round == 2:
              # Turn
              self.community_cards.append(self.deck.draw(1)[0])
          elif self.betting_round == 3:
              # River
              self.community_cards.append(self.deck.draw(1)[0])
          else:
              # Showdown
              pass
      
      def resolve_showdown(self):
          player_score = self.evaluator.evaluate(self.player_hand, self.community_cards)
          bot_score = self.evaluator.evaluate(self.bot_hand, self.community_cards)
          if player_score < bot_score:
              self.player_stack += self.pot
              print("You win the showdown!")
              self.reward = 1
          elif player_score > bot_score:
              self.bot_stack += self.pot
              print("Bot wins the showdown.")
              self.reward = -1
          else:
              # Split pot
              self.player_stack += self.pot / 2
              self.bot_stack += self.pot / 2
              print("It's a tie at showdown.")
              self.reward = 0
          self.done = True
          # Reveal hands
          print(f"Your hand: {Card.print_pretty_cards(self.player_hand)}")
          print(f"Bot's hand: {Card.print_pretty_cards(self.bot_hand)}")
          print(f"Community cards: {Card.print_pretty_cards(self.community_cards)}")
      
      def bot_decision(self):
          # Get the state
          state = self.get_bot_state()
          state_tensor = torch.tensor([state], dtype=torch.float32)
          with torch.no_grad():
              q_values = self.policy_net(state_tensor)
          action_idx = q_values.argmax().item()
          return action_idx
      
      def action_mapping(self, action_idx):
          action_map = {
              0: 'fold',
              1: 'check/call',
              2: 'raise_small',
              3: 'raise_medium',
              4: 'raise_large'
          }
          return action_map[action_idx]
      
      def get_bot_raise_amount(self, action_str):
          if action_str == 'raise_small':
              return 10
          elif action_str == 'raise_medium':
              return 50
          elif action_str == 'raise_large':
              return 100
      
      def get_state(self):
          # For the player
          # Return any state information if needed
          return None
      
      def get_bot_state(self):
          # Encode bot's hand and community cards
          hand_cards = self.bot_hand + self.community_cards
          hand_encoded = self.encode_cards(hand_cards)
          # Normalize stacks and pot
          normalized_bot_stack = self.bot_stack / 1000
          normalized_player_stack = self.player_stack / 1000
          normalized_pot = self.pot / 2000  # Max possible pot
          # Encode betting round
          round_encoded = [0] * 5
          round_encoded[min(self.betting_round, 4)] = 1
          # Combine all features
          state = np.concatenate([
              hand_encoded,
              [normalized_bot_stack],
              [normalized_player_stack],
              [normalized_pot],
              round_encoded
          ])
          return state.astype(np.float32)
      
      def encode_cards(self, cards):
          # One-hot encode the cards
          card_vector = np.zeros(52)
          suit_map = {1: 0, 2: 1, 4: 2, 8: 3}  # Map suit bitmasks to indices 0..3
          for card in cards:
              rank = Card.get_rank_int(card)  # Returns 2..14
              suit_bitmask = Card.get_suit_int(card)  # Returns 1, 2, 4, or 8
              suit = suit_map.get(suit_bitmask)
              if suit is None:
                  raise ValueError(f"Unknown suit bitmask: {suit_bitmask}")
              idx = (rank - 2) + suit * 13  # Adjust rank to 0..12
              if idx < 0 or idx >= 52:
                  raise ValueError(f"Calculated index {idx} is out of bounds")
              card_vector[idx] = 1
          return card_vector

      def load_bot_model(self, model_path):
          # Initialize the bot's policy network
          state_dim = len(self.get_bot_state())
          action_dim = 5  # Number of actions (adjusted to match the trained model)
          self.policy_net = DQN(state_dim, action_dim)
          self.policy_net.load_state_dict(torch.load(model_path))
          self.policy_net.eval()  # Set network to evaluation mode

      def print_game_state(self):
          print("\n--- Game State ---")
          print(f"Your stack: {self.player_stack}")
          print(f"Bot's stack: {self.bot_stack}")
          print(f"Pot: {self.pot}")
          print(f"Your hand: {Card.print_pretty_cards(self.player_hand)}")
          if self.community_cards:
              print(f"Community cards: {Card.print_pretty_cards(self.community_cards)}")
          else:
              print("Community cards: None")
          print("-------------------\n")
