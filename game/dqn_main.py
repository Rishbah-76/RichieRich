import pygame
import os
import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append("../src")

from dqn_env import PokerEnvCLI
from treys import Card, Deck, Evaluator
print(os.getcwd())
# Initialize Pygame
pygame.init()
pygame.font.init()
pygame.mixer.init()

# Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CARD_WIDTH = 100
CARD_HEIGHT = 145
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (25, 25, 25)
GREEN = (34, 139, 34)
RED = (220, 20, 60)
BLUE = (0, 0, 139)

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

class PokerGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Poker Game")
        self.clock = pygame.time.Clock()
        self.load_assets()
        self.setup_game()
        self.setup_ui()

    def load_assets(self):
        self.card_images = {}
        self.card_back = pygame.image.load(os.path.join("assets\cards", "card_back.png"))
        self.card_back = pygame.transform.scale(self.card_back, (CARD_WIDTH, CARD_HEIGHT))
        
        # Load card images with correct naming convention
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = ['02', '03', '04', '05', '06', '07', '08', '09', '10', 'J', 'Q', 'K', 'A']
        
        for suit in suits:
            for rank in ranks:
                filename = f"card_{suit}_{rank}.png"
                try:
                    img = pygame.image.load(os.path.join("assets\cards", filename))
                    img = pygame.transform.scale(img, (CARD_WIDTH, CARD_HEIGHT))
                    self.card_images[f"{suit}_{rank}"] = img
                except pygame.error as e:
                    print(f"Error loading card image: {filename}")
                    raise e

        # Load table background
        self.table_bg = pygame.image.load(os.path.join("assets", "poker-table.png"))
        self.table_bg = pygame.transform.scale(self.table_bg, (WINDOW_WIDTH, WINDOW_HEIGHT))

    def setup_game(self):
        self.env = PokerEnvCLI()
        self.env.load_bot_model('poker_dqn_model.pth')
        self.game_state = self.env.reset()
        
    def setup_ui(self):
        self.font_large = pygame.font.SysFont("Arial", 32)
        self.font_medium = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 18)

        # Setup buttons
        button_height = 40
        button_y = WINDOW_HEIGHT - 80
        
        self.buttons = {
            'fold': pygame.Rect(800, button_y, 80, button_height),
            'call': pygame.Rect(890, button_y, 80, button_height),
            'raise': pygame.Rect(980, button_y, 80, button_height),
            'bet_input': pygame.Rect(1070, button_y, 140, button_height)
        }
        
        self.input_active = False
        self.input_text = ""
        self.warning_text = ""

    def get_card_image(self, card):
        rank = Card.get_rank_int(card)
        suit = Card.get_suit_int(card)
        
        # Convert to proper naming format
        suit_name = {
            1: 'hearts',
            2: 'spades',
            4: 'diamonds',
            8: 'clubs'
        }[suit]
        
        rank_name = {
            14: 'A',
            13: 'K',
            12: 'Q',
            11: 'J',
            10: '10'
        }.get(rank, str(rank))
        print(f"herheehehehe : {suit_name}_{rank_name}")
        return self.card_images[f"{suit_name}_{rank_name}"]

    def draw_game_state(self):
        # Draw background
        self.screen.blit(self.table_bg, (0, 0))

        # Draw cards
        self.draw_community_cards()
        self.draw_player_cards()
        self.draw_opponent_cards()

        # Draw game info
        self.draw_game_info()
        
        # Draw UI elements
        self.draw_buttons()
        
        pygame.display.flip()

    def draw_community_cards(self):
        x = WINDOW_WIDTH // 2 - (CARD_WIDTH * 2.5)
        y = WINDOW_HEIGHT // 2 - CARD_HEIGHT // 2
        
        for card in self.env.community_cards:
            self.screen.blit(self.get_card_image(card), (x, y))
            x += CARD_WIDTH + 10

    def draw_player_cards(self):
        x = WINDOW_WIDTH // 2 - CARD_WIDTH - 10
        y = WINDOW_HEIGHT - CARD_HEIGHT - 20
        
        for card in self.env.player_hand:
            self.screen.blit(self.get_card_image(card), (x, y))
            x += CARD_WIDTH + 10

    def draw_opponent_cards(self):
        x = WINDOW_WIDTH // 2 - CARD_WIDTH - 10
        y = 20
        
        if self.env.done:
            for card in self.env.bot_hand:
                self.screen.blit(self.get_card_image(card), (x, y))
                x += CARD_WIDTH + 10
        else:
            for _ in range(2):
                self.screen.blit(self.card_back, (x, y))
                x += CARD_WIDTH + 10

    def draw_game_info(self):
        # Draw pot
        pot_text = self.font_large.render(f"Pot: ${self.env.pot}", True, WHITE)
        self.screen.blit(pot_text, (WINDOW_WIDTH//2 - pot_text.get_width()//2, WINDOW_HEIGHT//2 - 100))

        # Draw stacks
        player_text = self.font_medium.render(f"Your Stack: ${self.env.player_stack}", True, WHITE)
        bot_text = self.font_medium.render(f"Bot Stack: ${self.env.bot_stack}", True, WHITE)
        
        self.screen.blit(player_text, (20, WINDOW_HEIGHT - 40))
        self.screen.blit(bot_text, (20, 20))

        # Draw current bets
        if self.env.player_current_bet > 0:
            bet_text = self.font_small.render(f"Your Bet: ${self.env.player_current_bet}", True, WHITE)
            self.screen.blit(bet_text, (WINDOW_WIDTH//2 - 60, WINDOW_HEIGHT - 180))

        if self.env.bot_current_bet > 0:
            bet_text = self.font_small.render(f"Bot Bet: ${self.env.bot_current_bet}", True, WHITE)
            self.screen.blit(bet_text, (WINDOW_WIDTH//2 - 60, 180))

    def draw_buttons(self):
        for text, rect in self.buttons.items():
            pygame.draw.rect(self.screen, BLUE, rect)
            button_text = self.font_small.render(text.upper(), True, WHITE)
            self.screen.blit(button_text, 
                           (rect.centerx - button_text.get_width()//2,
                            rect.centery - button_text.get_height()//2))

        if self.input_active:
            pygame.draw.rect(self.screen, WHITE, self.buttons['bet_input'])
            text_surface = self.font_small.render(self.input_text, True, BLACK)
            self.screen.blit(text_surface, (self.buttons['bet_input'].x + 5, self.buttons['bet_input'].y + 10))

        if self.warning_text:
            warning_surface = self.font_small.render(self.warning_text, True, RED)
            self.screen.blit(warning_surface, (WINDOW_WIDTH - 300, WINDOW_HEIGHT - 120))

    def handle_input(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            
            for button, rect in self.buttons.items():
                if rect.collidepoint(mouse_pos):
                    if button == 'fold':
                        self.env.step('fold')
                    elif button == 'call':
                        self.env.step('check/call')
                    elif button == 'raise':
                        if self.input_text.isdigit():
                            self.env.step(f'raise {self.input_text}')
                            self.input_text = ""
                    elif button == 'bet_input':
                        self.input_active = True
                    break
            else:
                self.input_active = False

        elif event.type == pygame.KEYDOWN and self.input_active:
            if event.key == pygame.K_RETURN:
                self.input_active = False
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            elif event.unicode.isdigit():
                self.input_text += event.unicode

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                self.handle_input(event)
            
            self.draw_game_state()
            
            if self.env.done:
                pygame.time.wait(2000)
                self.env.reset()

        pygame.quit()

if __name__ == "__main__":
    game = PokerGame()
    game.run()