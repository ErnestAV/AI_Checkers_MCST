import numpy as np
import tkinter as tk
from tkinter import messagebox
import numpy as np
import random
import time
import copy
import threading
from math import log, sqrt

class CheckersGameGUI:
    def __init__(self, master, player1, player2):
        self.master = master
        self.game = CheckersGame()
        self.players = [player1, player2]
        self.current_player = 1
        self.master.title("Checkers Game")
        self.canvas = tk.Canvas(master, width=400, height=400)
        self.canvas.pack()
        self.selected_piece = None
        self.draw_board()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def draw_board(self):
        self.canvas.delete("all")
        color = "white"
        for i in range(8):
            color = "gray" if color == "white" else "white"
            for j in range(8):
                x0, y0, x1, y1 = j * 50, i * 50, (j + 1) * 50, (i + 1) * 50
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
                piece = self.game.board[i, j]
                if piece in [1, 3]:
                    fill = "red"
                elif piece in [2, 4]:
                    fill = "black"
                else:
                    continue
                if self.selected_piece == (i, j):
                    fill = "green"

                if piece in [3, 4]:
                    self.draw_star(x0, y0, x1, y1, fill)
                else:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill=fill)

    def draw_star(self, x0, y0, x1, y1, fill):
        points = [ 
            x0 + (x1 - x0) / 2, y0,
            x0 + (3 * (x1 - x0) / 4), y1 - (y1 - y0) / 4,
            x1, y1 - (y1 - y0) / 4,
            x0 + (x1 - x0) / 2, y1,
            x0, y1 - (y1 - y0) / 4,
            x0 + (x1 - x0) / 4, y1 - (y1 - y0) / 4
        ]
        self.canvas.create_polygon(points, fill=fill, outline="black")

    def on_canvas_click(self, event):
        current_player = self.players[self.current_player - 1]
        if current_player.player_type == 'human':
            col = event.x // 50
            row = event.y // 50
            piece = self.game.board[row, col]

            if self.selected_piece and (row, col) != self.selected_piece:
                move = [self.selected_piece, (row, col)]
                current_player.set_human_move(move)
                self.move_piece(move)
                self.post_move_processing()
            elif piece in [self.current_player, self.current_player + 2]:
                self.selected_piece = (row, col)
                self.draw_board()

    def post_move_processing(self):
        print("Post move processing called.")
        self.current_player = 3 - self.current_player
        self.game.current_player = self.current_player
        print(f"Next player: Player {self.current_player}")

        if self.players[self.current_player - 1].player_type == 'AI':
            print("Starting AI thread.")
            self.execute_ai_move()
    
    def execute_ai_move(self):
        print("AI move function called.")
        ai_move = self.players[self.current_player - 1].get_ai_move(self.game)

        if ai_move:
            print(f"Executing AI move: {ai_move}")
            self.move_piece(ai_move)
            self.update_board()
            self.post_move_processing()
        else:
            print("No valid AI move found.")

    def move_piece(self, move):
        current_player = self.players[self.current_player - 1]
        player_move = current_player.get_move(self.game)

        player_move = [list(move) for move in player_move]

        if player_move and player_move in self.game.get_possible_moves(self.game.current_player):
            self.game.make_move(player_move, self.game.current_player)
            self.selected_piece = None
            self.draw_board()

            if self.game.is_game_over():
                winner = self.game.get_winner()
                messagebox.showinfo("Game Over", f"Player {winner} wins!")
        else:
            self.selected_piece = None
            self.draw_board()

        self.update_board()
    
    def update_board(self):
        self.draw_board()
        self.master.update()

    def on_human_move(self, move):
        self.move_piece(move)
        self.update_board()
        if self.players[self.current_player - 1].player_type == 'AI':
            self.make_ai_move()

    def make_ai_move(self):
        print("AI move function called.")
        ai_move = self.players[self.current_player - 1].get_ai_move(self.game)

        if ai_move:
            print(f"Executing AI move: {ai_move}")
            self.move_piece(ai_move)
            self.update_board()
            self.post_move_processing()
        else:
            print("No valid AI move found.")

    def _ai_move_thread(self):
        print("AI making a move in a separate thread...")
        ai_move = self.players[self.game.current_player - 1].get_ai_move(self.game)
        if ai_move:
            self.canvas.after(0, self._execute_ai_move, ai_move)

    def _execute_ai_move(self, ai_move):
        self.game.make_move(ai_move, self.game.current_player)
        self.game.current_player = 3 - self.game.current_player
        self.draw_board()
        if self.game.is_game_over():
            winner = self.game.get_winner()
            messagebox.showinfo("Game Over", f"Player {winner} wins!")

class CheckersGame:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.initialize_board()
        self.piece_count = {1: 12, 2: 12}
        self.current_player = 1
        self.move_history = []
        self.no_progress_count = 0

    def initialize_board(self):
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    if row < 3:
                        self.board[row, col] = 1
                    elif row > 4:
                        self.board[row, col] = 2

    def get_possible_moves(self, player_num):
        moves = []
        for row in range(8):
            for col in range(8):
                if self.board[row, col] in [player_num, player_num + 2]:
                    is_king = self.board[row, col] == player_num + 2
                    moves.extend(self.get_moves_for_piece(row, col, player_num, is_king))
        return moves

    def get_moves_for_piece(self, row, col, player_num, is_king):
        moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if not is_king:
            directions = directions[2:] if player_num == 1 else directions[:2]

        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == 0:
                moves.append([[row, col], [r, c]])

        captures = self.get_captures_for_piece(row, col, player_num, is_king)
        if captures:
            moves.extend(captures)

        return moves
    
    def get_captures_for_piece(self, row, col, player_num, is_king, path=[]):
        captures = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if is_king else \
                    ([(1, -1), (1, 1)] if player_num == 1 else [(-1, -1), (-1, 1)])

        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == 3 - player_num:
                r_jump, c_jump = r + dr, c + dc
                if 0 <= r_jump < 8 and 0 <= c_jump < 8 and self.board[r_jump, c_jump] == 0:
                    self.board[row, col], self.board[r, c], self.board[r_jump, c_jump] = 0, 0, player_num
                    next_jumps = self.get_captures_for_piece(r_jump, c_jump, player_num, is_king, path + [[row, col], [r_jump, c_jump]])
                    captures.extend(next_jumps if next_jumps else [path + [[row, col], [r_jump, c_jump]]])
                    self.board[row, col], self.board[r, c], self.board[r_jump, c_jump] = player_num, 3 - player_num, 0

        return captures

    
    def make_move(self, move, player_num):
        self.board[move[1][0], move[1][1]] = player_num
        self.board[move[0][0], move[0][1]] = 0

        if abs(move[1][0] - move[0][0]) == 2:
            self.board[int((move[1][0] + move[0][0]) / 2), int((move[1][1] + move[0][1]) / 2)] = 0
            opponent_num = 3 - player_num
            self.piece_count[opponent_num] -= 1

        # Kinging logic
        if (player_num == 1 and move[1][0] == 7) or (player_num == 2 and move[1][0] == 0):
            self.board[move[1][0], move[1][1]] = player_num + 2

        self.move_history.append(self.board.copy())

        if self.move_is_progress(move):
            self.no_progress_count = 0
        else:
            self.no_progress_count += 1


    def is_game_over(self):
        return self.piece_count[1] == 0 or self.piece_count[2] == 0 or \
               len(self.get_possible_moves(1)) == 0 or len(self.get_possible_moves(2)) == 0 or \
               self.is_draw()
    
    def is_draw(self):
        last_state_count = sum(np.array_equal(self.move_history[-1], past_state) for past_state in self.move_history)
        if last_state_count >= 3:
            return True
        if self.no_progress_count >= 50: 
            return True
        return False
    
    
    def get_winner(self):
        if len(self.get_possible_moves(1)) == 0:
            return 2
        elif len(self.get_possible_moves(2)) == 0:
            return 1
        return 0
    
    def move_is_progress(self, move):
        print(f"Move received in move_is_progress: {move}")
        start_pos, end_pos = move[0], move[-1]
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        if abs(start_row - end_row) == 2 and abs(start_col - end_col) == 2:
            return True

        if (self.current_player == 1 and end_row == 7) or (self.current_player == 2 and end_row == 0):
            if self.board[start_row, start_col] in [1, 2]:
                return True

        return False

class Node:
    def __init__(self, game_state, player_num, move=None, parent=None):
        self.game_state = game_state
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = game_state.get_possible_moves(player_num)

    def is_terminal_node(self):
        return self.game_state.is_game_over()

    def expand(self):
        if not self.untried_moves:
            raise Exception("No more moves to expand")
        move = self.untried_moves.pop()
        new_game_state = copy.deepcopy(self.game_state)
        current_player_num = new_game_state.current_player
        new_game_state.make_move(move, current_player_num)
        child_node = Node(new_game_state, move=move, parent=self, player_num=current_player_num)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

    def uct_select_child(self):
        best_value, best_node = float('-inf'), None
        for child in self.children:
            if child.visits == 0:
                uct_value = float('inf')
            else:
                uct_value = (child.wins / child.visits) + sqrt(2 * log(self.visits) / child.visits)
            if uct_value > best_value:
                best_value, best_node = uct_value, child
        return best_node
    
class CheckersPlayer:
    def __init__(self, player_number, player_type='human', mcts_time_limit=1.0):
        self.player_number = player_number
        self.player_type = player_type
        self.mcts_time_limit = mcts_time_limit
        self.move_ready = threading.Event()

    def get_move(self, game):
        print(f"Player {self.player_number} ({self.player_type}) to move.")
        if self.player_type == 'human':
            return self.get_human_move(game)
        elif self.player_type == 'AI':
            return self.get_ai_move(game)

    def set_human_move(self, move):
        self.human_move = move
        self.move_ready.set()

    def get_human_move(self, game):
        print("Waiting for human move...")
        self.move_ready.wait()
        move = self.human_move
        self.human_move = None
        self.move_ready.clear()
        print("Move received:", move)
        return move

    def get_ai_move(self, game):
        root_node = Node(game_state=copy.deepcopy(game), player_num=self.player_number)
        end_time = time.time() + self.mcts_time_limit

        while time.time() < end_time:
            node = root_node
            while not node.is_terminal_node():
                if node.untried_moves:
                    node = node.expand()
                else:
                    node = node.uct_select_child()

            simulation_result = self.simulate_game(node.game_state)

            while node:
                node.update(simulation_result)
                node = node.parent

        if root_node.children:
            for child in root_node.children:
                print(f"Move: {child.move}, Visits: {child.visits}, Wins: {child.wins}")
            ai_move = sorted(root_node.children, key=lambda c: c.visits, reverse=True)[0].move
            print(f"AI move selected: {ai_move}")
            return ai_move
        else:
            print("No valid AI moves available.")
            return None

    def simulate_game(self, game_state):
        while not game_state.is_game_over():
            print("In simulate_game loop, game over status:", game_state.is_game_over())
            possible_moves = game_state.get_possible_moves(game_state.current_player)
            game_state.make_move(random.choice(possible_moves), game_state.current_player)
            game_state.current_player = 3 - game_state.current_player
        winner = game_state.get_winner()
        return 1 if winner == self.player_number else 0

root = tk.Tk()
player1 = CheckersPlayer(1, 'human')
player2 = CheckersPlayer(2, 'AI')
game_gui = CheckersGameGUI(root, player1, player2)
root.mainloop()