import pygame
import sys
import h5py
import copy
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import joblib

class Connect4:




    def evaluate_board(self):
        player_score = self.evaluate_board_for_player('Red')
        opponent_score = self.evaluate_board_for_player('Yellow')
        return player_score - opponent_score

    def evaluate_board_for_player(self, player):
        score = 0
        # Directions: Horizontal, Vertical, Diagonal /, Diagonal \
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if self.board[row][col] == player:
                    for direction in directions:
                        score += self.count_potentials(self.board, row, col, direction, player)
        return score

    def count_potentials(self, board, row, col, direction, player):
        consecutive_count = 1
        for i in range(1, 4):  # Check up to three pieces in each direction
            new_row = row + direction[0] * i
            new_col = col + direction[1] * i
            if 0 <= new_row < len(board) and 0 <= new_col < len(board[0]):
                if board[new_row][new_col] == player:
                    consecutive_count += 1
                else:
                    break
            else:
                break
        # Score based on the number of pieces in line
        if consecutive_count == 2:
            return 10  # Two in a row scores 10
        elif consecutive_count == 3:
            return 100  # Three in a row scores 100
        elif consecutive_count == 4:
            return 1000  # Four in a row scores 1000 (or could be treated as a win condition)
        return 0

    def get_valid_moves(self):
        return [col for col in range(7) if self.is_valid_move(col)]

    def minimax(self, depth, alpha, beta, maximizing_player, move_list:list = []):
        valid_moves = self.get_valid_moves()

        self.print_board(self.board)


        if depth == 0 or self.check_win() or self.is_tie():
                return move_list, self.evaluate_board()

        if maximizing_player:
            value = -float('inf')
            best_seq = move_list
            for move in valid_moves:
                self.drop_piece(move)
                new_seq , new_score = self.minimax(depth - 1, alpha, beta, False, move_list + [move])
                self.undo_move(move)
                if new_score > value:
                    value = new_score
                    best_seq = new_seq
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best_seq, value
        else:
            value = float('inf')
            best_seq = move_list
            for move in valid_moves:
                self.drop_piece(move)
                new_seq, new_score = self.minimax(depth - 1, alpha, beta, True, move_list + [move])
                self.undo_move(move)
                if new_score < value:
                    value = new_score
                    best_seq = new_seq
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return best_seq, value

    def __init__(self):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.players = ['Red', 'Yellow']
        self.current_player = 0
        self.game_over = False

    def is_valid_move(self, column):
        return 0 <= column < 7 and self.board[0][column] == ' '


    # change from range(5, -1, -1)
    def drop_piece(self, column):
        for row in range(5, -1, -1):
            if self.board[row][column] == ' ':
                self.board[row][column] = self.players[self.current_player]
                break

    def undo_move(self, column):
        for row in range(6):
            if self.board[row][column] != ' ':
                self.board[row][column] = ' '
                break

    def check_win(self):
        for row in range(6):
            for col in range(7):
                if self.board[row][col] != ' ':
                    if col + 3 < 7 and self.board[row][col] == self.board[row][col + 1] == \
                            self.board[row][col + 2] == self.board[row][col + 3]:
                        return True
                    if row + 3 < 6 and self.board[row][col] == self.board[row + 1][col] == \
                            self.board[row + 2][col] == self.board[row + 3][col]:
                        return True
                    if col + 3 < 7 and row + 3 < 6 and self.board[row][col] == self.board[row + 1][col + 1] == \
                            self.board[row + 2][col + 2] == self.board[row + 3][col + 3]:
                        return True
                    if col - 3 >= 0 and row + 3 < 6 and self.board[row][col] == self.board[row + 1][col - 1] == \
                            self.board[row + 2][col - 2] == self.board[row + 3][col - 3]:
                        return True
        return False

    def is_tie(self):
        return all(self.board[i][j] != ' ' for i in range(6) for j in range(7))

    def reset_game(self):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.current_player = 0
        self.game_over = False

    def make_move(self, column):
        if self.is_valid_move(column):
            self.drop_piece(column)
            if self.check_win():
                self.game_over = True
            elif self.is_tie():
                self.game_over = True
            else:
                self.current_player = (self.current_player + 1) % 2

    def get_state(self):
        state = np.zeros((6, 7, 3), dtype=int)
        for i in range(6):
            for j in range(7):
                if self.board[i][j] == 'Red':
                    state[i, j, 0] = 1
                elif self.board[i][j] == 'Yellow':
                    state[i, j, 1] = 1
                else:
                    state[i, j, 2] = 1
        return state

    def get_valid_moves(self):
        return [col for col in range(7) if self.is_valid_move(col)]

    def generate_game_states(self, num_games):
        data_features = []
        data_labels = []

        for _ in range(num_games):
            self.reset_game()
            while not self.game_over:
                state = self.get_state()
                valid_moves = self.get_valid_moves()
                move = np.random.choice(valid_moves)
                self.make_move(move)

                data_features.append(state.flatten())
                data_labels.append(move)

                if self.check_win() or self.is_tie():
                    break

        return np.array(data_features), np.array(data_labels)
    
    def print_board(self, board):
        # Header for column numbers
        print("  ".join([str(i) for i in range(len(board[0]))]))
        
        # Print each row of the board
        for row in board:
            # Create a formatted string for each row where each cell is separated by '|'
            formatted_row = " | ".join(["R" if cell == "Red" else "Y" if cell == "Yellow" else " " for cell in row])
            print("| " + formatted_row + " |")
            
        # Print a bottom border
        print("-" * (len(board[0]) * 4 + 1))
class Game:
    def __init__(self, model):
        pygame.display.init()  # Initialize only the display module
        pygame.font.init()  # Initialize the font module

        self.screen_width = 639
        self.screen_height = 553

       
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Connect Four")

        self.background_image = pygame.image.load("Connect4Board.png").convert()
        self.background_image = pygame.transform.scale(self.background_image, (self.screen_width, self.screen_height))

        self.clock = pygame.time.Clock()
        self.connect4_game = Connect4()
        self.game_over = False
        self.winner = None
        self.game_over_time = None

        self.dataset_features = []
        self.dataset_labels = []
        self.model = model  # Store the model

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if not self.game_over and event.type == pygame.MOUSEBUTTONDOWN:
                if 0 <= event.pos[0] <= self.screen_width:
                    column = event.pos[0] // 91
                    if self.connect4_game.is_valid_move(column):
                        self.connect4_game.drop_piece(column)
                        if self.connect4_game.check_win():
                            self.game_over = True
                            self.winner = self.connect4_game.players[self.connect4_game.current_player]
                            self.game_over_time = pygame.time.get_ticks()
                        elif self.connect4_game.is_tie():
                            self.game_over = True
                            self.winner = "Tie"
                            self.game_over_time = pygame.time.get_ticks()
                        else:
                            encoded_board = self.encode_board(self.connect4_game.board).flatten()
                            self.dataset_features.append(encoded_board)
                            self.dataset_labels.append(column)
                            self.connect4_game.current_player = (self.connect4_game.current_player + 1) % 2
                            if self.connect4_game.current_player == 1:  # AI's turn
                                self.ai_move()

            if not self.game_over and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:
                    print(self.connect4_game.evaluate_board())
                    print(self.connect4_game.minimax(4, -float('inf'), float('inf'), True))

            if self.game_over and event.type == pygame.MOUSEBUTTONDOWN:
                if hasattr(self, 'play_again_button') and self.play_again_button.collidepoint(event.pos):
                    self.reset_game()
                    self.game_over = False
                    self.winner = None
                    self.game_over_time = None



        return False

    def encode_board(self, board):
        encoded_board = np.zeros((6, 7, 3))
        for i in range(6):
            for j in range(7):
                if board[i][j] == 'Red':
                    encoded_board[i, j, 0] = 1
                elif board[i][j] == 'Yellow':
                    encoded_board[i, j, 1] = 1
                else:
                    encoded_board[i, j, 2] = 1
        return encoded_board

    def ai_move(self):
        encoded_board = self.encode_board(self.connect4_game.board).flatten().reshape(1, -1)
        best_move = self.model.predict(encoded_board)[0]
        best_move = int(best_move)  # Ensure best_move is an integer
        if self.connect4_game.is_valid_move(best_move):
            self.connect4_game.drop_piece(best_move)
            if self.connect4_game.check_win():
                self.game_over = True
                self.winner = self.connect4_game.players[self.connect4_game.current_player]
                self.game_over_time = pygame.time.get_ticks()
            elif self.connect4_game.is_tie():
                self.game_over = True
                self.winner = "Tie"
                self.game_over_time = pygame.time.get_ticks()
            else:
                self.connect4_game.current_player = (self.connect4_game.current_player + 1) % 2

    def draw(self):
      self.draw_board()  # Remove self.screen argument here
      if self.game_over:
        self.draw_winner_screen()
      pygame.display.update()


    def draw_board(self):
    # Use the background image directly as the screen
       self.screen.blit(self.background_image, (0, 0))

    # Draw circles based on the game board state
       for row in range(6):
         for col in range(7):
            color = (255, 255, 255)  # Default color for empty slot
            if self.connect4_game.board[row][col] == 'Red':
                color = (255, 0, 0)  # Red color for player 1
            elif self.connect4_game.board[row][col] == 'Yellow':
                color = (255, 255, 0)  # Yellow color for player 2

            # Calculate the center of each slot
            circle_center = (col * 91 + 45, row * 91 + 45)  # Adjusted center calculation
            if color != (255, 255, 255):
                pygame.draw.circle(self.screen, color, circle_center, 40)


    
    def draw_winner_screen(self):
        custom_font_path = "CustomFont2.ttf"
        font_size = 25
        custom_font = pygame.font.Font(custom_font_path, font_size)
        if self.winner == "Tie":
            text = custom_font.render("It's a tie!", True, (0, 0, 0))
        else:
            text = custom_font.render(f"The {self.winner} Player Has Won! Congrats!", True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(text, text_rect)
        
        if self.game_over_time is not None and pygame.time.get_ticks() - self.game_over_time >= 5000:
            self.draw_play_again_button()

    def draw_play_again_button(self):
        custom_font_path = "CustomFont2.ttf"
        font_size = 25
        custom_font = pygame.font.Font(custom_font_path, font_size)
        play_again_text = custom_font.render("Play Again", True, (255, 255, 255))
        play_again_rect = play_again_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 85))
        self.play_again_button = play_again_rect
        self.screen.blit(play_again_text, play_again_rect)

    def reset_game(self):
        self.connect4_game = Connect4()
        self.dataset_features = []
        self.dataset_labels = []

    def run(self):
        quit_game = False
        while not quit_game:
            quit_game = self.handle_events()
            self.draw()
            self.clock.tick(30)
        pygame.quit()

    def save_dataset(self):
        dataset_features = np.array(self.dataset_features)
        dataset_labels = np.array(self.dataset_labels).reshape(-1, 1)

        with h5py.File('connect4_dataset.h5', 'w') as hf:
            hf.create_dataset('features', data=dataset_features)
            hf.create_dataset('labels', data=dataset_labels)

def load_or_train_model():
    if os.path.exists('connect4_model.pkl'):
        model = joblib.load('connect4_model.pkl')
        print("Model loaded from file.")
    else:
        from train_model import train_model  # Import train_model function
        model = train_model()
    return model


if __name__ == "__main__":
    model = load_or_train_model()
    
    game = Game(model)
    game.run()
    game.save_dataset()
    print("Dataset saved successfully.")



