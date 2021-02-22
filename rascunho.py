# %%

def set_goals(board):
    flat = sum(board, []) 
    return [flat[0:3],flat[3:6],flat[6:9],flat[0:9:3],flat[1:9:3],flat[2:9:3],flat[0:9:4],flat[2:7:2]]


def draw(posicoes):

    print("---------")
    print("| " + " ".join(posicoes[0]) + " |")
    print("| " + " ".join(posicoes[1]) + " |")
    print("| " + " ".join(posicoes[2]) + " |")
    print("---------")


def fill_grid(arr, entrada):

    ii = 0
    for i in range(3):
        arr[i][0] = entrada[ii]
        arr[i][1] = entrada[ii+1]
        arr[i][2] = entrada[ii+2]
        ii += 3

def enter_cells():
    
    ent = input("Enter cells:").replace('_',' ')
    l = abs(9 - len(ent))
    if l > 0:
        ent += (l*" ")
    return ent

def winner(goals):
    winners = []
    for i in goals:
        if i == ['X','X','X'] or i == ['O','O','O']:
            winners.append(i)
    if len(winners) >= 2: 
        print('Impossible')
        return True
    elif len(winners) == 0:
        if " " in goals: print('Game not finished')
        else: 
            print('Draw')
            return True
    else: 
        print(winners[0][0], 'wins')
        return True

def read_player(board):
    
    while True:
        try:
            
            num = [int(x)-1 for x in input('Enter the coordinates:').split()]

            if min(num) >= 0 and max(num) <= 2:
                dim = board[num[0]][num[1]]
                if dim != " ":
                    print('This cell is occupied! Choose another one!')
                else:
                    board[num[0]][num[1]] = 'X'
                    goals = set_goals(board)
                    draw(board)
                    if winner(goals): break
            else:
                print("Coordinates should be from 1 to 3!")
        except ValueError:
            print("You should enter numbers!")
        
        
# %%
board = [[' ', ' ', ' '] for _ in range(3)]
entrada = enter_cells()
fill_grid(board,entrada)
draw(board)
read_player(board)

# %%
import random
n = int(input())
random.seed(n)
print(random.choice("Voldemort"))
# %%
n = int(input())
random.seed(n)
print(random.uniform(0,1))
# %%

random.seed(3)
# call the function here
print(random.betavariate(.9, .1))
# %%
Class House():
    construction = "building"
    elevator = True

# object of the class House
new_house = House()
# %%
