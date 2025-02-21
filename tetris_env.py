import gym
import numpy as np
import pygame


class TetrisEnv(gym.Env):
    """
    A modular, Gym-compatible Tetris environment for DQN.

    - Observation: A dictionary with:
         "board": the board state with only placed pieces.
         "active_tetromino_mask": a binary mask of the current piece at its spawn position.
    - Action: A 2D vector [column, rotation] indicating the desired drop column and rotation.
    - Reward: Configurable via reward_params.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        board_width=10,
        board_height=20,
        state_encoding="binary",  # "binary" or "piece"
        reward_params=None,  # e.g., {"line_clear": 40, "step": 1, "game_over": -10}
        piece_set=None,  # Custom set of pieces; default uses standard 7 Tetriminos
        render_mode=None,  # None, "human", or "rgb_array"
        **kwargs
    ):
        super().__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.state_encoding = state_encoding
        self.render_mode = render_mode

        # Reward configuration with defaults
        default_rewards = {
            "line_clear": 40,
            "step": 1,
            "game_over": -10,
            "invalid_action": -50,
        }
        if reward_params is not None:
            default_rewards.update(reward_params)
        self.reward_params = default_rewards

        # Action space: [column, rotation]
        self.action_space = gym.spaces.MultiDiscrete(board_width * 4)

        # Observation space: For simplicity, we assume both board and active piece are represented as arrays
        # with the same shape. (Note: When using a dictionary observation space, you may need to define a Dict space.)
        board_space = gym.spaces.Box(
            low=0, high=7, shape=(board_height, board_width), dtype=np.int8
        )
        self.observation_space = gym.spaces.Dict(
            {"board": board_space, "active_tetromino_mask": board_space}
        )

        # Initialize board and piece variables
        self.board = np.zeros((board_height, board_width), dtype=np.int8)
        if piece_set is None:
            self.piece_set = self.get_default_pieces()
        else:
            self.piece_set = piece_set

        self.current_piece = None
        self.current_piece_type = None
        self.current_piece_shape = None  # Will hold the rotated piece shape
        self.current_piece_pos = None  # [row, col]

        # Pygame initialization for human rendering
        self.screen = None
        if self.render_mode == "human":
            pygame.init()
            self.block_size = 30  # pixels per block
            self.screen = pygame.display.set_mode(
                (
                    self.board_width * self.block_size,
                    self.board_height * self.block_size,
                )
            )
            pygame.display.set_caption("Tetris Environment")

    def get_default_pieces(self):
        """
        Returns the standard 7 Tetriminos.
        Each piece is assigned a unique id (1 to 7) and defined as a numpy array.
        """
        pieces = {}
        # I piece (id=1)
        pieces[1] = np.array([[1, 1, 1, 1]])
        # O piece (id=2)
        pieces[2] = np.array([[2, 2], [2, 2]])
        # T piece (id=3)
        pieces[3] = np.array([[0, 3, 0], [3, 3, 3]])
        # S piece (id=4)
        pieces[4] = np.array([[0, 4, 4], [4, 4, 0]])
        # Z piece (id=5)
        pieces[5] = np.array([[5, 5, 0], [0, 5, 5]])
        # J piece (id=6)
        pieces[6] = np.array([[6, 0, 0], [6, 6, 6]])
        # L piece (id=7)
        pieces[7] = np.array([[0, 0, 7], [7, 7, 7]])
        return pieces

    def reset(self, seed=None, options=None):
        """
        Resets the environment:
          - Clears the board.
          - Spawns a new piece.
        Returns a dictionary observation and info.
        """
        super().reset(seed=seed)
        self.board = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        self.spawn_new_piece()
        obs = self.get_observation()
        return obs, {}

    def spawn_new_piece(self):
        """
        Selects a new piece at random from the piece_set and sets its spawn position.
        For this version, we place the new piece at the top center.
        """
        self.current_piece_type = np.random.choice(list(self.piece_set.keys()))
        self.current_piece = self.piece_set[self.current_piece_type].copy()
        self.current_piece_shape = self.current_piece.copy()
        piece_height, piece_width = self.current_piece_shape.shape
        spawn_col = (self.board_width - piece_width) // 2
        spawn_row = 0  # Top row
        self.current_piece_pos = [spawn_row, spawn_col]

    def get_observation(self):
        """
        Returns a dictionary observation containing:
         - "board": the board state (with placed pieces only).
         - "active_tetromino_mask": the active piece drawn at its current (spawn) position.
        """
        # Board observation
        if self.state_encoding == "binary":
            board_obs = (self.board > 0).astype(np.int8)
        elif self.state_encoding == "piece":
            board_obs = self.board.copy()
        else:
            board_obs = self.board.copy()

        # Build the active tetromino mask (an empty board with the active piece overlaid)
        active_mask = np.zeros_like(self.board, dtype=np.int8)
        piece = self.current_piece_shape
        row, col = self.current_piece_pos
        piece_height, piece_width = piece.shape
        # For simplicity, assume the piece fits entirely within the board.
        active_mask[row : row + piece_height, col : col + piece_width] = (
            piece > 0
        ).astype(np.int8)

        return {"board": board_obs, "active_tetromino_mask": active_mask}

    def step(self, action):
        """
        Executes one time-step:
         - Rotates the current piece based on action.
         - Determines the drop row for the given column.
         - If placement is valid, places the piece and clears lines.
         - Returns the observation as a dictionary.
        """
        # Unpack action parameters
        column, rotation = action // 4, action % 4  # rotation: number of 90° rotations

        # Apply rotation to the current piece
        self.current_piece_shape = np.rot90(self.current_piece, k=rotation)
        piece_height, piece_width = self.current_piece_shape.shape

        # Check horizontal bounds
        if column < 0 or column + piece_width > self.board_width:
            penalty = self.reward_params.get("invalid_action", -50)
            info = {"reason": "Invalid column placement"}
            self.spawn_new_piece()
            return self.get_observation(), penalty, False, False, info

        # Determine the drop row for the piece
        row = 0
        while True:
            if not self.check_valid_position(self.current_piece_shape, row, column):
                break
            row += 1
            if row + piece_height > self.board_height:
                break
        drop_row = row - 1

        # Check if drop position is valid
        if drop_row < 0 or not self.check_valid_position(
            self.current_piece_shape, drop_row, column
        ):
            penalty = self.reward_params.get("invalid_action", -50)
            info = {"reason": "Invalid drop placement"}
            self.spawn_new_piece()
            return self.get_observation(), penalty, False, False, info

        # Place the piece on the board at the computed drop position.
        self.place_piece(self.current_piece_shape, drop_row, column)

        # Clear any full lines.
        lines_cleared = self.clear_full_lines()

        reward = (
            self.reward_params["step"]
            + lines_cleared * self.reward_params["line_clear"]
        )

        # Check for game over (if any block is in the top row)
        if np.any(self.board[0] > 0):
            reward += self.reward_params["game_over"]
            terminated = True
        else:
            terminated = False

        # Spawn a new piece (which will appear at the spawn position)
        self.spawn_new_piece()

        obs = self.get_observation()
        info = {"lines_cleared": lines_cleared}
        return obs, reward, terminated, False, info

    def check_valid_position(self, piece, row, col):
        """
        Checks if the piece can be placed at (row, col) without collision or going out-of-bounds.
        """
        piece_height, piece_width = piece.shape
        if (
            row + piece_height > self.board_height
            or col < 0
            or col + piece_width > self.board_width
        ):
            return False

        board_slice = self.board[row : row + piece_height, col : col + piece_width]
        overlap = (board_slice > 0) & (piece > 0)
        return not np.any(overlap)

    def place_piece(self, piece, row, col):
        """
        Places the piece on the board at (row, col).
        """
        piece_height, piece_width = piece.shape
        for i in range(piece_height):
            for j in range(piece_width):
                if piece[i, j] > 0:
                    self.board[row + i, col + j] = piece[i, j]

    def clear_full_lines(self):
        """
        Clears full lines from the board.
        """
        full_lines = [i for i in range(self.board_height) if np.all(self.board[i] > 0)]
        lines_cleared = len(full_lines)
        if lines_cleared > 0:
            new_board = np.zeros_like(self.board)
            new_row = self.board_height - 1
            for i in range(self.board_height - 1, -1, -1):
                if i not in full_lines:
                    new_board[new_row] = self.board[i]
                    new_row -= 1
            self.board = new_board
        return lines_cleared

    def render(self, mode="human"):
        """
        Renders the current board state.
        """
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))
            for i in range(self.board_height):
                for j in range(self.board_width):
                    # Render placed pieces from self.board
                    cell_value = self.board[i, j]
                    if cell_value > 0:
                        color = self.get_color(cell_value)
                        rect = pygame.Rect(
                            j * self.block_size,
                            i * self.block_size,
                            self.block_size,
                            self.block_size,
                        )
                        pygame.draw.rect(self.screen, color, rect)
                        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
                    # Render the active piece in a different shade (overlay)
                    # Note: For simplicity, we do not blend colors here.
                    active_mask = self.get_observation()["active_tetromino_mask"]
                    if active_mask[i, j] > 0:
                        pygame.draw.rect(
                            self.screen,
                            (255, 255, 255),
                            pygame.Rect(
                                j * self.block_size,
                                i * self.block_size,
                                self.block_size,
                                self.block_size,
                            ),
                            2,
                        )
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            rgb_array = np.zeros(
                (
                    self.board_height * self.block_size,
                    self.board_width * self.block_size,
                    3,
                ),
                dtype=np.uint8,
            )
            for i in range(self.board_height):
                for j in range(self.board_width):
                    cell_value = self.board[i, j]
                    color = self.get_color(cell_value) if cell_value > 0 else (0, 0, 0)
                    rgb_array[
                        i * self.block_size : (i + 1) * self.block_size,
                        j * self.block_size : (j + 1) * self.block_size,
                    ] = color
            return rgb_array

    def get_color(self, piece_id):
        """
        Maps a piece id to an RGB color.
        """
        color_map = {
            1: (0, 255, 255),  # I: Cyan
            2: (255, 255, 0),  # O: Yellow
            3: (128, 0, 128),  # T: Purple
            4: (0, 255, 0),  # S: Green
            5: (255, 0, 0),  # Z: Red
            6: (0, 0, 255),  # J: Blue
            7: (255, 165, 0),  # L: Orange
        }
        return color_map.get(piece_id, (255, 255, 255))

    def close(self):
        """
        Closes the pygame window.
        """
        if self.screen:
            pygame.quit()
            self.screen = None


# Example test run
if __name__ == "__main__":
    import gymnasium as gym

    env = TetrisEnv(render_mode="human")
    obs, _ = env.reset()
    print("Observation keys:", obs.keys())
    print("Board shape:", obs["board"].shape)
    print("Active Tetromino Mask shape:", obs["active_tetromino_mask"].shape)
