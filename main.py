import numpy as np

def solve_lights_out(board):
    """
    Solves a Lights Out puzzle using linear algebra over GF(2).
    
    Args:
        board: 2D list/array where 1 = light on, 0 = light off
    
    Returns:
        solution: 2D array showing which buttons to press (1 = press, 0 = don't press)
        or None if no solution exists
    """
    rows = len(board)
    cols = len(board[0])
    n = rows * cols
    
    # Create the coefficient matrix A
    A = np.zeros((n, n), dtype=int)
    
    for i in range(rows):
        for j in range(cols):
            button_idx = i * cols + j
            
            # Pressing a button affects itself
            light_idx = i * cols + j
            A[light_idx][button_idx] = 1
            
            # And its neighbors
            if i > 0:
                light_idx = (i-1) * cols + j
                A[light_idx][button_idx] = 1
            if i < rows - 1:
                light_idx = (i+1) * cols + j
                A[light_idx][button_idx] = 1
            if j > 0:
                light_idx = i * cols + (j-1)
                A[light_idx][button_idx] = 1
            if j < cols - 1:
                light_idx = i * cols + (j+1)
                A[light_idx][button_idx] = 1
    
    # Convert board to vector
    b = np.array([board[i][j] for i in range(rows) for j in range(cols)], dtype=int)
    
    # Solve Ax = b over GF(2)
    solution_vector = gaussian_elimination_gf2(A, b)
    
    if solution_vector is None:
        return None
    
    # Convert solution vector back to 2D grid
    solution = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(solution_vector[i * cols + j])
        solution.append(row)
    
    return solution


def gaussian_elimination_gf2(A, b):
    """Solves Ax = b over GF(2) using Gaussian elimination."""
    n = len(b)
    m = len(A[0])
    
    aug = np.column_stack([A.copy(), b.copy()])
    
    pivot_row = 0
    for col in range(m):
        found_pivot = False
        for row in range(pivot_row, n):
            if aug[row][col] == 1:
                aug[[pivot_row, row]] = aug[[row, pivot_row]]
                found_pivot = True
                break
        
        if not found_pivot:
            continue
        
        for row in range(n):
            if row != pivot_row and aug[row][col] == 1:
                aug[row] = (aug[row] + aug[pivot_row]) % 2
        
        pivot_row += 1
    
    x = np.zeros(m, dtype=int)
    
    for row in range(min(pivot_row, n) - 1, -1, -1):
        leading_col = -1
        for col in range(m):
            if aug[row][col] == 1:
                leading_col = col
                break
        
        if leading_col == -1:
            if aug[row][m] == 1:
                return None
            continue
        
        val = aug[row][m]
        for col in range(leading_col + 1, m):
            val = (val + aug[row][col] * x[col]) % 2
        x[leading_col] = val
    
    return x


def verify_solution(board, solution):
    """Verify that the solution actually solves the puzzle."""
    rows = len(board)
    cols = len(board[0])
    
    # Create a copy of the board
    result = [row[:] for row in board]
    
    # Apply each button press
    for i in range(rows):
        for j in range(cols):
            if solution[i][j] == 1:
                # Toggle this position
                result[i][j] = 1 - result[i][j]
                # Toggle neighbors
                if i > 0:
                    result[i-1][j] = 1 - result[i-1][j]
                if i < rows - 1:
                    result[i+1][j] = 1 - result[i+1][j]
                if j > 0:
                    result[i][j-1] = 1 - result[i][j-1]
                if j < cols - 1:
                    result[i][j+1] = 1 - result[i][j+1]
    
    # Check if all lights are off
    all_off = all(result[i][j] == 0 for i in range(rows) for j in range(cols))
    
    return all_off, result


def print_board(board, label="Board"):
    """Pretty print the board."""
    print(f"\n{label}:")
    for row in board:
        print(" ".join(str(cell) for cell in row))


def print_solution(solution):
    """Print which buttons to press."""
    print("\n" + "="*50)
    print("SOLUTION:")
    print("="*50)
    print("\nButtons to press (X = press, . = don't press):")
    for row in solution:
        print(" ".join("X" if cell else "." for cell in row))
    
    print("\nPress these positions (row, col) - using 1-based indexing:")
    positions = []
    for i in range(len(solution)):
        for j in range(len(solution[0])):
            if solution[i][j] == 1:
                positions.append(f"({i+1},{j+1})")
    
    if positions:
        print(", ".join(positions))
    else:
        print("No buttons need to be pressed (puzzle already solved)")


def solve_from_string(board_string):
    """
    Parse a board from a string format.
    Example: "1 0 1\\n0 0 1\\n1 1 1"
    """
    lines = board_string.strip().split('\n')
    board = []
    for line in lines:
        row = [int(x) for x in line.strip().split()]
        board.append(row)
    return board


# Main interactive section
if __name__ == "__main__":
    print("="*60)
    print("LIGHTS OUT PUZZLE SOLVER")
    print("="*60)
    print("\nThis solver uses linear algebra over GF(2) to find the")
    print("optimal solution for any Lights Out puzzle.")
    print("\nHow it works:")
    print("1. Models each button as a binary variable (press/don't press)")
    print("2. Creates equations for each light (on/off)")
    print("3. Solves the system using Gaussian elimination in GF(2)")
    print("="*60)
    
    # Example 1: 3x3
    print("\n\nEXAMPLE 1: 3x3 Puzzle")
    board_3x3 = [
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    print_board(board_3x3, "Initial Board")
    solution = solve_lights_out(board_3x3)
    
    if solution:
        print_solution(solution)
        verified, final_board = verify_solution(board_3x3, solution)
        print(f"\nVerification: {'✓ CORRECT' if verified else '✗ FAILED'}")
        if not verified:
            print_board(final_board, "Final Board (should be all 0s)")
    
    # Example 2: 5x5
    print("\n\nEXAMPLE 2: 5x5 Puzzle")
    board_5x5 = [
        [1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1]
    ]
    print_board(board_5x5, "Initial Board")
    solution = solve_lights_out(board_5x5)
    
    if solution:
        print_solution(solution)
        verified, final_board = verify_solution(board_5x5, solution)
        print(f"\nVerification: {'✓ CORRECT' if verified else '✗ FAILED'}")
        if not verified:
            print_board(final_board, "Final Board (should be all 0s)")
    
    # Custom input example
    print("\n\n" + "="*60)
    print("TO SOLVE YOUR OWN PUZZLE:")
    print("="*60)
    print("In Python, use:")
    print("  board = [[1,0,1], [0,0,1], [1,1,1]]  # Your puzzle")
    print("  solution = solve_lights_out(board)")
    print("  print_solution(solution)")