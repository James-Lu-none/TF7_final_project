def check_win(board, row, col, last_move):
    status = 0
    # 檢查行
    for c in range(10):
        if board[row][c]%2 == board[row][c+1]%2 == board[row][c+2]%2 == board[row][c+3]%2 == board[row][c+4]%2 == board[row][c+5]%2 == last_move%2:
            if board[row][c] >= 0 and board[row][c+1] >= 0 and board[row][c+2] >= 0 and board[row][c+3] >= 0 and board[row][c+4] >= 0 and board[row][c+5] >= 0 :
                status = 1

    # 檢查列
    for r in range(10):
        if board[r][col]%2 == board[r+1][col]%2 == board[r+2][col]%2 == board[r+3][col]%2 == board[r+4][col]%2 == board[r+5][col]%2 == last_move%2:
            if board[r][col] >= 0 and board[r+1][col] >= 0 and board[r+2][col] >= 0 and board[r+3][col] >= 0 and board[r+4][col] >= 0 and board[r+5][col] >= 0 :
                status = 2

    # 檢查正對角線
    for i in range(10):
        if row - i >= 0 and col - i >= 0 and row - i + 5 < 15 and col - i + 5 < 15:
            if board[row-i][col-i]%2 == board[row-i+1][col-i+1]%2 == board[row-i+2][col-i+2]%2 == board[row-i+3][col-i+3]%2 == board[row-i+4][col-i+4]%2 == board[row-i+5][col-i+5]%2 == last_move%2:
                if board[row-i][col-i] >= 0 and board[row-i+1][col-i+1] >= 0 and board[row-i+2][col-i+2] >= 0 and board[row-i+3][col-i+3] >= 0 and board[row-i+4][col-i+4] >= 0 and board[row-i+5][col-i+5] >= 0 :
                    status = 3

    # 檢查負對角線
    for i in range(10):
        if row + i < 15 and col - i >= 0 and row + i - 5 >= 0 and col - i + 5 < 15:
            if board[row+i][col-i]%2 == board[row+i-1][col-i+1]%2 == board[row+i-2][col-i+2]%2 == board[row+i-3][col-i+3]%2 == board[row+i-4][col-i+4]%2 == board[row+i-5][col-i+5]%2 == last_move%2:
                if board[row+i][col-i] >= 0 and board[row+i-1][col-i+1] >= 0 and board[row+i-2][col-i+2] >= 0 and board[row+i-3][col-i+3] >= 0 and board[row+i-4][col-i+4] >= 0 and board[row+i-5][col-i+5] >= 0 :
                 status = 4

    # 檢查行
    for c in range(11):
        if board[row][c]%2 == board[row][c+1]%2 == board[row][c+2]%2 == board[row][c+3]%2 == board[row][c+4]%2 == last_move%2 and status != 1:
            if board[row][c] >= 0 and board[row][c+1] >= 0 and board[row][c+2] >= 0 and board[row][c+3] >= 0 and board[row][c+4] >= 0 and last_move >= 0 :
                return True

    # 檢查列
    for r in range(11):
        if board[r][col]%2 == board[r+1][col]%2 == board[r+2][col]%2 == board[r+3][col]%2 == board[r+4][col]%2 == last_move%2 and status != 2:
            if board[r][col] >= 0 and board[r+1][col] >= 0 and board[r+2][col] >= 0 and board[r+3][col] >= 0 and board[r+4][col] >= 0 :
                return True

    # 檢查正對角線
    for i in range(11):
        if row - i >= 0 and col - i >= 0 and row - i + 4 < 15 and col - i + 4 < 15:
            if board[row-i][col-i]%2 == board[row-i+1][col-i+1]%2 == board[row-i+2][col-i+2]%2 == board[row-i+3][col-i+3]%2 == board[row-i+4][col-i+4]%2 == last_move%2 and status != 3:
                if board[row-i][col-i] >= 0 and board[row-i+1][col-i+1] >= 0 and board[row-i+2][col-i+2] >= 0 and board[row-i+3][col-i+3] >= 0 and board[row-i+4][col-i+4] >= 0 :
                    return True

    # 檢查負對角線
    for i in range(11):
        if row + i < 15 and col - i >= 0 and row + i - 4 >= 0 and col - i + 4 < 15:
            if board[row+i][col-i]%2 == board[row+i-1][col-i+1]%2 == board[row+i-2][col-i+2]%2 == board[row+i-3][col-i+3]%2 == board[row+i-4][col-i+4]%2 == last_move%2 and status != 4:
                if board[row+i][col-i] >= 0 and board[row+i-1][col-i+1] >= 0 and board[row+i-2][col-i+2] >= 0 and board[row+i-3][col-i+3] >= 0 and board[row+i-4][col-i+4] >= 0 :
                    return True

    return False