import numpy as np
from collections import defaultdict

MAP = ["CCC",
       " ==",
       "CCC",
       " ==",
       "CCC"]

class BirdEnv:
    """Bird looks for a worm"""
    metadata = {'render.modes': ['human']}

    def __init__(self, rng):
        self.nA = 5 #number of actions: right, left, down, up, eat the worm
        self.nC = 9 #number of cells in 3x3 grid
        self.nS = self.nC**2 # state = (position of bird, position of worm)
        self.ncol = 3 #number of columns in 3x3 grid
        self.rand_generator = rng
        #transitions[c][a] == [(probability, nextcell),..]
        self.transitions = {c : {} for c in range(self.nC)}
        def move(i, j, inc):
            cell_i = max(min(i + inc[0], 4), 0)
            cell_j = max(min(j + inc[1], 2), 0)
            #move according to action, if you can
            if MAP[cell_i][cell_j] == "=":
                cell_i = i
            elif MAP[cell_i][cell_j] == " ":
                cell_i += inc[0]
            cell = 3 * (cell_i // 2) + cell_j
            return cell
        for i, row in enumerate(MAP):
            for j, char in enumerate(row):
                if char == "C":
                    d = defaultdict(lambda:0)
                    for inc in [(0,1), (0, -1), (1, 0), (-1,0)]:
                        cell = move(i,j,inc)
                        d[cell] += 0.025
                    for action, inc in enumerate([(0,1), (0, -1), (1, 0), (-1,0)]):
                        cell = move(i,j,inc)
                        trans = d.copy()
                        trans[cell] += 0.9                     
                        self.transitions[3*(i//2)+j][action] = [(prob, nextcell) for nextcell, prob in trans.items()]
        #initial cell distribution (always start in the upper left corner)
        self.icd = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cell = self.rand_generator.choice(np.arange(self.nC), p=self.icd)
        #initial worm distribution: in one of the three right-most locations at the end of each corridor
        self.iwd = [0, 0, 1./3, 0, 0, 1./3, 0, 0, 1./3]
        self.worm = self.rand_generator.choice(np.arange(self.nC), p=self.iwd)
        self.lastaction = 4
    
    def state(self):
        return self.nC * self.cell + self.worm
    
    def step(self, action):
        """Execute one time step within the environment"""
        reward = 0
        if action == 4:
            #try eating the worm
            if self.cell == self.worm:
                #move worm into one of the empty cells on the right
                self.worm = self.rand_generator.choice([(self.worm + 3) % self.nC, (self.worm + 6) % self.nC])
                reward = 1
        else:
            transitions = self.transitions[self.cell][action]
            i = self.rand_generator.choice(np.arange(len(transitions)), p=[t[0] for t in transitions])
            _, cell = transitions[i]
            self.cell = cell
        self.lastaction = action
        state = self.state()
        return state, reward

    def reset(self):
    # Reset the state of the environment to an initial state
        self.cell = self.rand_generator.choice(np.arange(self.nC), p=self.icd)
        self.worm = self.rand_generator.choice(np.arange(self.nC), p=self.iwd)
        
    def render(self, mode='human', close=False):
    # Render the environment to the screen
        outfile = sys.stdout
        desc = [["C", "C", "C"], ["C", "C", "C"], ["C", "C", "C"]]
        row, col = self.cell // self.ncol, self.cell % self.ncol
        desc[row][col] = "B"
        row, col = self.worm // self.ncol, self.worm % self.ncol
        desc[row][col] = "W"
        
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Right", "Left", "Down", "Up", "Eat"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        
