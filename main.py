import numpy as np
import random
import seaborn as sns
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import imageio
import copy
import argparse
import csv

parser = argparse.ArgumentParser(
    prog = "main.py",
    description= "Code for demonstrating QLearning, First Visit Monte Carlo and SARSA"
)

# General arguments
parser.add_argument("-saveDir", type = str, default = "", help = "Directory to save training output data to (relative path)")
parser.add_argument("-numIters", type = int, default = 200000, help = "Number of training iterations to run")
parser.add_argument("-policy", type = str, default = "episilonGreedy", choices= ["episilonGreedy", "greedy"], help = "Choice of policy")
parser.add_argument("-epsilon", type = float, default = 0.1, help = "Value of epsilon in the epsilon greedy policy")
parser.add_argument("-algorithm", type = str, default = "SARSA", choices= ["SARSA", "Q", "FVMC"])
parser.add_argument("-gamma", type = float, default = 0.95, help = "Discounting rate of rewards. Discounting is enabled by default. Set to 1 if theres no discounting")

# Environment arguments
parser.add_argument("-width", type = int, default = 0, help = "Width of the grid. Not setting width and height results in the basic 4x4 grid initialization")
parser.add_argument("-height", type = int, default = 0, help = "Width of the grid. Not setting width and height results in the basic 4x4 grid initialization")
parser.add_argument("-saveGridFile", type = str, default = "", help = "File to save the generated environment, so the environment can be reused")
parser.add_argument("-gridFile", type = str, default = "", help = "gridFile to load and use as the environment")

# Additional general arguments
parser.add_argument("-taperEpsilon", type = bool, default = False, help = "Setting this to true will reduce episilon logarithmically as training progresses")
parser.add_argument("-halfEvery", type = int, default = 50000, help = "Sets the rate at which epsilon decreases, if taperEpsilon is set to true")
parser.add_argument("-noUselessMoves", type = bool, default = False, 
                    help ="Setting this to true will ensure that the robot checks if a move does not move it out of the grid before allowing it to \
                            be considered")
parser.add_argument("-penalizeBackTracking", type = bool, default = False,
                    help = "Setting this to true will penalize doing moves that undo the previous move")

# First Visit Monte Carlo arguments
parser.add_argument("-prune", type = bool, default = False, 
                    help = "Set to true if we want to stop updating the Q table beyond a certain number of iterations. \
                            This prevents the very small updates that are given when the episode has too many steps and discounting is enabled")
parser.add_argument("-batchSize", type = int, default = 2, help = "Size of each training batch")
parser.add_argument("-adjustmentRate", type = float, default = 1.0, 
                    help = "Modifies the FVMC update rule to step towards the new QTable value instead of directly replacing the QTable value with \
                            the new QTable value observed in a batch of episodes")

# Q Learning and SARSA arguments
parser.add_argument("-alpha", type = float, default = 0.0005, help = "Learning rate")

# Training data tracking arguments
parser.add_argument("-dataLogInterval", type = int, default = 1, help = "Rate at which to log rewards and number of steps per episode")
parser.add_argument("-QTableLogInterval", type = int, default = 10000, help = "Rate at which to log Q Table progress")

# Grid class handles the state of the environment
class Grid:
    def __init__(self):
        self.initRewards()

    THINICEGRIDVALUE = -1
    GOALGRIDVALUE = 1
    EMPTYGRIDVALUE = 0

    def initGridBasic(self):
        self.grid = np.zeros(shape = (4, 4), dtype=int)
        # Add thin ice zones
        self.grid[3][0] = self.THINICEGRIDVALUE
        self.grid[1][1] = self.THINICEGRIDVALUE
        self.grid[1][3] = self.THINICEGRIDVALUE
        self.grid[2][3] = self.THINICEGRIDVALUE
        # Add goal
        self.grid[3][3] = self.GOALGRIDVALUE

    # Simple BFS to check if goal is reachable. Used in generating a random test environment for the robot
    def checkIfGoalIsReachable(self, startPosition, goalPosition):
        frontier = [startPosition]
        visited = []
        
        while frontier:
            node = frontier.pop()
            visited.append(node)
            potentialFrontierNodes = []
            if node[0] > 0:
                potentialFrontierNodes.append((node[0] - 1, node[1]))
            if node[0] < (self.grid.shape[0] - 1):
                potentialFrontierNodes.append((node[0] + 1, node[1]))
            if node[1] > 0:
                potentialFrontierNodes.append((node[0], node[1] - 1))
            if node[1] < (self.grid.shape[1] - 1):
                potentialFrontierNodes.append((node[0], node[1] + 1))
            for potentialFrontierNode in potentialFrontierNodes:
                if self.grid[potentialFrontierNode[0]][potentialFrontierNode[1]] == self.THINICEGRIDVALUE:
                    continue
                if potentialFrontierNode == goalPosition:
                    return True
                if potentialFrontierNode not in visited:
                    frontier.append(potentialFrontierNode)

        return False

    # Initializes a random environment for the robot based on part 2 of the assignment
    def initGridExtended(
        self, startPosition: Tuple[int, int], 
        numRows: int = 10, 
        numCols: int = 10,
        goalPosition: Tuple[int, int] = None
    ):
        self.grid = np.zeros(shape = (numRows, numCols), dtype=int)

        # Randomly generate goal position
        if not goalPosition:
            goalPosition = (random.randrange(0, numRows), random.randrange(0, numCols))
            while goalPosition == startPosition:
                goalPosition = (random.randrange(0, numRows), random.randrange(0, numCols))
        self.grid[goalPosition[0]][goalPosition[1]] = self.GOALGRIDVALUE

        # Initialize the thin ice regions randomly, and discard them if they block access to the goal
        numPos = numRows * numCols
        numThinIce = int(numPos/4)
        numThinIcePlaced = 0
        while numThinIcePlaced < numThinIce:
            newThinIcePosition = (random.randrange(0, numRows), random.randrange(0, numCols))
            while newThinIcePosition == goalPosition or newThinIcePosition == startPosition:
                newThinIcePosition = (random.randrange(0, numRows), random.randrange(0, numCols))
            self.grid[newThinIcePosition[0]][newThinIcePosition[1]] = self.THINICEGRIDVALUE
            if self.checkIfGoalIsReachable(startPosition, goalPosition):
                numThinIcePlaced += 1
            else:
                self.grid[newThinIcePosition[0]][newThinIcePosition[1]] = self.EMPTYGRIDVALUE


    def initGridFromFile(self, fileName: str):
        print(f"Loading grid from file {fileName}...")
        with open(fileName,'r') as f:
            reader = csv.reader(f, delimiter=',')
            shape = next(reader)
            height = int(shape[0])
            width = int(shape[1])
            self.grid = np.zeros(shape = (height, width), dtype=int)

            for row in reader:
                coords_y = int(row[1][1])
                coords_x = int(row[1][4])
                if row[0] == 'x':
                    self.grid[coords_y][coords_x] = self.THINICEGRIDVALUE
                elif row[0] == 'g':
                    self.grid[coords_y][coords_x] = self.GOALGRIDVALUE

        print("Successfully loaded!") 
        self.printGrid()

    def saveGridToFile(self, fileName: str):
        data = []
        data.append((self.grid.shape[0], self.grid.shape[1]))

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i][j] == self.THINICEGRIDVALUE:
                    data.append(('x', (i, j)))
                elif self.grid[i][j] == self.GOALGRIDVALUE:
                    data.append(('g', (i, j)))

        with open(fileName, 'w') as f:
            csv.writer(f).writerows(data)

    # Sets the reward values for the different cell types
    def initRewards(self):
        self.rewardValues = {
            self.THINICEGRIDVALUE: -1,
            self.EMPTYGRIDVALUE: 0,
            self.GOALGRIDVALUE: 1
        }

    def queryRewardValue(self, state: Tuple[int, int]):
        gridValue = self.grid[state[0]][state[1]]
        return self.rewardValues[gridValue]

    def isStateTerminal(self, state: Tuple[int, int]):
        return self.grid[state[0]][state[1]] != 0

    def getGridSize(self):
        return self.grid.shape

    def printGrid(self):
        print("Grid layout")
        shape = self.grid.shape
        for i in range(shape[0]):
            toPrint = "|"
            for j in range(shape[1]):
                gridValue = self.grid[i][j]
                if gridValue == self.EMPTYGRIDVALUE:
                    toPrint += " |"
                elif gridValue == self.GOALGRIDVALUE:
                    toPrint += "G|"
                else:
                    toPrint += "X|"
            print(toPrint)

# QTable implementation
class QTable:
    def __init__(self, environment: Grid, actionsPerCoord = 4,initializationType: int = 0):
        self.gridSize = environment.getGridSize()
        self.environment = environment
        dims = (self.gridSize[0], self.gridSize[1], actionsPerCoord)
        if initializationType == 0:
            self.QTable = np.zeros(shape = dims, dtype=float)
        elif initializationType == 1:
            self.QTable = np.random.rand(self.gridSize[0], self.gridSize[1], actionsPerCoord)

    def getShape(self):
        return self.QTable.shape

    def printQTable(self):
        for i in range(self.gridSize[0]):
            toPrint = "|"
            for j in range(self.gridSize[1]):
                bestAction = self.QTable[i][j].argmax()
                if self.environment.grid[i][j] == self.environment.THINICEGRIDVALUE:
                    toPrint += "X|"
                    continue
                if self.environment.grid[i][j] == self.environment.GOALGRIDVALUE:
                    toPrint += "G|"
                    continue
                if bestAction == 0:
                    toPrint += "^|"
                elif bestAction == 1:
                    toPrint += "v|"
                elif bestAction == 2:
                    toPrint += "<|"
                else:
                    toPrint += ">|"
            print(toPrint)

# Robot class handles the state transitions
class Robot:
    def __init__(
        self, 
        startPosition: Tuple[int, int],
        environment: Grid,
    ):
        self.startPosition = startPosition
        self.position = startPosition
        self.gridShape = environment.getGridSize()
        self.environment = environment
        self.path = []
        self.numStepsTaken = 0
        self.numPossibleActions = 4

    UPMOVEMENTCODE = 0 
    DOWNMOVEMENTCODE = 1
    LEFTMOVEMENTCODE = 2
    RIGHTMOVEMENTCODE = 3

    def moveUp(self):
        if self.position[0] > 0:
            self.position = (self.position[0] - 1, self.position[1])
            
    def moveDown(self):
        if self.position[0] < (self.gridShape[0] - 1):
            self.position = (self.position[0] + 1, self.position[1])

    def moveLeft(self):
        if self.position[1] > 0:
            self.position = (self.position[0], self.position[1] - 1)

    def moveRight(self):
        if self.position[1] < (self.gridShape[1] - 1):
            self.position = (self.position[0], self.position[1] + 1)

    def move(self, movementCode: int):
        if movementCode == self.UPMOVEMENTCODE:
            self.moveUp()
        elif movementCode == self.DOWNMOVEMENTCODE:
            self.moveDown()
        elif movementCode == self.LEFTMOVEMENTCODE:
            self.moveLeft()
        else:
            self.moveRight()
        self.path.append(movementCode)
        self.numStepsTaken += 1

    def reset(self):
        self.position = self.startPosition
        self.path = []
        self.numStepsTaken = 0

    def getUselessMoves(self):
        uselessMoves = []

        if self.position[0] == 0:
            uselessMoves.append(self.UPMOVEMENTCODE)
        if self.position[0] == (self.gridShape[0] - 1):
            uselessMoves.append(self.DOWNMOVEMENTCODE)
        if self.position[1] == 0:
            uselessMoves.append(self.LEFTMOVEMENTCODE)
        if self.position[1] == (self.gridShape[1] - 1):
            uselessMoves.append(self.RIGHTMOVEMENTCODE)

        return uselessMoves

# Simple tracker that tracks a single metric
class Single1DMetricTracker:
    def __init__(self, metricName: str, xAxisName: str):
        self.metricName = metricName
        self.xAxisName = xAxisName
        self.metricValues = []
        self.x = []

    def update(self, metricValue: float, x: float):
        self.metricValues.append(metricValue)
        self.x.append(x)

    def getPlot(self):
        return self.x, self.metricValues

    def getRollingAverageMetricValues(self, windowSize: int = 10):
        metricRollingAverage = []
        numValues = len(self.x)
        for i in range(numValues):
            windowStartIndex = max(0, int(i - windowSize/2))
            windowEndIndex = min(int(i + windowSize/2), numValues)
            metricRollingAverage.append(np.mean(self.metricValues[windowStartIndex: windowEndIndex]))

        return metricRollingAverage


    def getPlotRollingAverage(self, windowSize: int = 10):
        return self.x, self.getRollingAverageMetricValues(windowSize)

    def savePlotToFolder(self, folderPath: str):
        x, y = self.getPlot()
        figureName = self.metricName + "V" + self.xAxisName
        self.saveDataToFolder(x, y, folderPath, figureName)

    def saveRollingAveragePlotToFolder(self, folderPath: str, windowSize: int = 10):
        x, y = self.getPlotRollingAverage(windowSize)
        figureName = self.metricName + "RollingAverageV" + self.xAxisName
        self.saveDataToFolder(x, y, folderPath, figureName)

    def saveDataToFolder(self, x: np.array, y: np.array, folderPath: str, figureName: str):
        os.makedirs(folderPath, exist_ok = True)
        savePath = folderPath + "/" + figureName + ".png"
        plt.plot(x, y)
        plt.xlabel(self.xAxisName)
        plt.ylabel(self.metricName)
        plt.savefig(savePath)
        plt.clf()

class QTableTracker:
    def __init__(self, environment: Grid, startPoint: Tuple[int, int]):
        self.QTableSnaphots = []
        self.iterationCounts = []
        self.environment = environment
        self.startPoint = startPoint

    def update(self, QTable: QTable, iterationCount: int):
        self.QTableSnaphots.append(copy.deepcopy(QTable.QTable))
        self.iterationCounts.append(iterationCount)

    def getCellValueHeatMap(self, QTable: np.ndarray):
        return np.max(QTable, axis=2)

    def saveSingleHeatMap(self, QTable: np.ndarray, plotTitle: str, savePath: str):
        heatMap = self.getCellValueHeatMap(QTable)
        QTableShape = QTable.shape
        for i in range(QTableShape[0]):
            for j in range(QTableShape[1]):
                if self.environment.grid[i][j] == self.environment.GOALGRIDVALUE:
                    heatMap[i, j] = 2
                elif self.environment.grid[i][j] == self.environment.THINICEGRIDVALUE:
                    heatMap[i, j] = -1
        ax = sns.heatmap(heatMap, linewidth=0.5, cmap="rocket")

        arrowLength = 0.4
        plt.scatter(self.startPoint[0] + 0.5, self.startPoint[1] + 0.5, s = 400, marker = "o", color = "yellow")
        for i in range(QTableShape[0]):
            for j in range(QTableShape[1]):
                if self.environment.grid[i][j] == self.environment.GOALGRIDVALUE:
                    plt.scatter(j + 0.5, i + 0.5, s = 400, marker = "*", color = "yellow")
                elif self.environment.grid[i][j] == self.environment.THINICEGRIDVALUE:
                    plt.scatter(j + 0.5, i + 0.5, s = 400, marker = "X", color = "red")
                else:
                    bestAction = QTable[i][j].argmax()
                    x, y, dx, dy = i + 0.5 , j + 0.5, 0, 0
                    if bestAction == 0:
                        dy = -arrowLength
                        x += arrowLength/2
                    elif bestAction == 1:
                        dy = arrowLength
                        x -= arrowLength/2
                    elif bestAction == 2:
                        dx = -arrowLength
                        y += arrowLength/2
                    else:
                        dx = arrowLength
                        y -= arrowLength/2
                    plt.arrow(y, x, dx, dy, width = 0.05)

        plt.title(plotTitle)
        plt.savefig(savePath)
        plt.clf()

    # Saves all the accumulated Q Tables as images, and compiles them into a gif
    def saveHeatMaps(self, saveFolder: str, generateGIF: bool = True, GIFDurationSec = 5):
        os.makedirs(saveFolder, exist_ok = True)
        fileList = []
        for i in range(len(self.QTableSnaphots)):
            fileName = f"{saveFolder}/iteration{self.iterationCounts[i]}.png"
            if generateGIF:
                fileList.append(fileName)
            self.saveSingleHeatMap(self.QTableSnaphots[i], f"Iteration {self.iterationCounts[i]}",fileName)

        if generateGIF:
            images = []
            for filePath in fileList:
                images.append(imageio.imread(filePath))
            secondsPerFrame = GIFDurationSec / len(self.QTableSnaphots)
            gifFileName = f"{saveFolder}/trainingProgress.gif"
            imageio.mimsave(gifFileName, images, duration=secondsPerFrame)

class ProgressTracker:
    def __init__(
        self, 
        environment: Grid, 
        startPosition: Tuple[int, int]
    ):
        self.stepsPerEpisodeTracker = Single1DMetricTracker("StepsPerEpisode", "NumIterations")
        self.rewardPerEpisodeTracker = Single1DMetricTracker("RewardPerEpisode", "NumIterations")
        self.QTableTracker = QTableTracker(environment, startPosition)

    def saveProgressData(self, foldername: str):
        self.rewardPerEpisodeTracker.saveRollingAveragePlotToFolder(foldername, 100)
        self.stepsPerEpisodeTracker.savePlotToFolder(foldername)
        self.QTableTracker.saveHeatMaps(foldername)

class greedyPolicy:
    def chooseAction(self, actionList: np.array):
        return actionList.argmax()

class epsilonGreedyPolicy:
    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def chooseAction(self, actionList: np.array):
        if random.random() > self.epsilon:
            return actionList.argmax()
        else:
            return random.randint(0, actionList.size - 1)

class firstVisitMonteCarlo:
    SARTupleType = Tuple[Tuple[int, int], int, float]

    def __init__(
        self, 
        environment: Grid,
        startPosition: Tuple[int, int],
        policy,
        gamma = 0.9,
        logStepsAndRewardsInterval = 100,
        logQTableProgessInterval = 10000,
        noUselessMoves = False,
        penalizeBackTracking = False,
    ):
        self.robot = Robot(startPosition, environment)
        self.startPosition = startPosition
        self.QTable = QTable(environment, self.robot.numPossibleActions, initializationType= 0)
        self.environment = environment
        self.policy = policy
        self.gamma = gamma
        self.noUselessMoves = noUselessMoves
        self.penalizeBackTracking = penalizeBackTracking

        self.logStepsAndRewardsInterval = logStepsAndRewardsInterval
        self.logQTableProgessInterval = logQTableProgessInterval
        self.progressTracker = ProgressTracker(environment, startPosition)

    # Runs a single episode and returns the number of steps in the episode and a list of tuples of [State, action, reward]. 
    # If set, only tuples for episodes after {updateCutOffIndex} will be returned 
    def runEpisode(self, robot: Robot, updateCutOffIndex: int = 0):
        SARRecord = []
        prevAction = None
        while not self.environment.isStateTerminal(robot.position):
            position = robot.position
            actionValues = copy.deepcopy(self.QTable.QTable[position[0]][position[1]]) # We need to deepcopy this as we dont want to modify the QTable
            # Remove useless moves from consideration if noUselessMoves is set to True
            if self.noUselessMoves:
                uselessMoves = robot.getUselessMoves()
                for uselessMove in uselessMoves:
                    actionValues[uselessMove] = -100

            # Penalize moves that undo the previous move if penalizeBackTracking is set to True
            if prevAction and self.penalizeBackTracking:
                type = prevAction % 2
                if type == 0:
                    actionValues[prevAction + 1] -= 0.2
                if type == 1:
                    actionValues[prevAction - 1] -= 0.2
            
            action = self.policy.chooseAction(actionValues)
            robot.move(action)
            prevAction = action

            # Append (State, action, reward obtained by executing action at state) to SARRecord
            SARRecord.append((
                position,
                action,
                self.environment.queryRewardValue(robot.position)
            ))

        robot.reset()
        if updateCutOffIndex == 0:
            return len(SARRecord), SARRecord
        else:
            return len(SARRecord), SARRecord[updateCutOffIndex::1]

        

    def train(
        self, 
        batchSize: int = 25, 
        numIter: int = 10000, 
        adjustmentRate: float = 1,
        prune: bool = False,
        taper = False,
        halfEvery = 50000
    ):
        gridSize = self.environment.getGridSize()
        actionsPerCoord = self.robot.numPossibleActions
        returnTableShape = (gridSize[0], gridSize[1], actionsPerCoord, batchSize)
        self.progressTracker.QTableTracker.update(self.QTable, 1) # Save the first QTable
        
        # Set the updateCutOffIndex if pruning is set to True. Updates that have decayed to less than 0.01 of the 
        # reward provided in the terminal state are not used to update the Qtable

        updateCutOffIndex = 0
        if prune:
            updateCutOffIndex = -(int(np.log(0.01)/ np.log(self.gamma)) + 1)

        for iterationNum in tqdm(range(numIter)):
            # Reevaluate epsilon if taper is set to True, and we have hit the threshold number of iteration for halving
            if taper and (iterationNum + 1) % halfEvery == 0:
                self.policy.epsilon /= 2
                print(f"Reached iteration {i + 1}! Halving epsilon to {self.policy.epsilon}")

            # Run episodes
            returnTable = np.zeros(returnTableShape, dtype = float) # The return table keeps track of the value of each cell that is encountered in each batch of episodes
            encounteredTable = np.full(self.QTable.getShape(), False)

            totalReward = 0
            totalNumStepsTaken = 0
            for episodeNum in range(batchSize):
                numSteps, episodeRecord = self.runEpisode(self.robot, updateCutOffIndex)

                totalNumStepsTaken += numSteps
                totalReward += episodeRecord[-1][2]

                expectedFutureReward = 0 # In the terminal state, expectedFutureReward = 0
                for state, action, reward in episodeRecord[::-1]: # We start from the back
                    # The value of the state action pair is given by the reward obtained by that state action pair, and the gamma discounted
                    # expected future reward. Each value is logged in the return table, and will be used as the expected future reward
                    # for the state action pair that precedes it.
                    # Since we start from the back, the value of older steps will override the value of newer steps. This is consistent
                    # with the First Visit Monte Carlo (FVMC) algorithm
                    returnValue = reward + self.gamma * expectedFutureReward
                    expectedFutureReward = returnValue
                    returnTable[state[0]][state[1]][action][episodeNum] = returnValue
                    encounteredTable[state[0]][state[1]][action] = True

            # Update QTable
            differences = []
            for i in range(gridSize[0]):
                for j in range(gridSize[1]):
                    for k in range(self.robot.numPossibleActions):
                        if (encounteredTable[i][j][k]):
                            returns = returnTable[i][j][k] # A list of values across the batch for a particular state action pair
                            averageValue = np.sum(returns)/ np.count_nonzero(returns)
                            difference = averageValue - self.QTable.QTable[i][j][k]
                            differences.append(difference)
                            self.QTable.QTable[i][j][k] += difference * adjustmentRate 
                            # If adjustment rate is set to 1, we obtain the FVMC update rule presented in the notes. Setting it to a value between 0 and 1
                            # will result in a step towards the value as evaluated by this batch, instead of overriding the QTable entry with the value as
                            # evaluated by this batch

            if (iterationNum+1)%self.logStepsAndRewardsInterval == 0:
                self.progressTracker.stepsPerEpisodeTracker.update(totalNumStepsTaken/ batchSize, iterationNum+1)
                self.progressTracker.rewardPerEpisodeTracker.update(totalReward/ batchSize, iterationNum+1)
            
            if (iterationNum+1)%self.logQTableProgessInterval == 0:
                self.progressTracker.QTableTracker.update(self.QTable, iterationNum+1)
            
            # Calculate the L2 norm of the differences. If its less than the threshold, stop
            totalSquaredDifferences = 0
            for difference in differences:
                totalSquaredDifferences += difference**2
            L2Difference = totalSquaredDifferences**0.5
            # if (L2Difference < 0.000001):
            #     break
        
        self.QTable.printQTable()
                            
            
class SARSA:
    def __init__(
        self, 
        environment: Grid,
        startPosition: Tuple[int, int],
        policy, 
        gamma: float = 0.9,
        alpha: float = 0.2,
        logStepsAndRewardsInterval: int = 100,
        logQTableProgessInterval: int = 10000
    ):
        self.robot = Robot(startPosition, environment)
        self.startPosition = startPosition
        self.QTable = QTable(environment, self.robot.numPossibleActions, initializationType=1)
        self.environment = environment
        self.policy = policy
        self.gamma = gamma
        self.alpha = alpha

        self.logStepsAndRewardsInterval = logStepsAndRewardsInterval
        self.logQTableProgessInterval = logQTableProgessInterval
        self.progressTracker = ProgressTracker(environment, startPosition)
        
    def train(self, numIter: int = 500000, taper = False, halfEvery = 50000):
        robot = Robot(self.startPosition, self.environment)
        QTable = self.QTable.QTable
        self.progressTracker.QTableTracker.update(self.QTable, 1)
        for i in tqdm(range(numIter)):
            # Perform first step to prep SAR
            state = robot.position
            action = self.policy.chooseAction(QTable[state[0]][state[1]])
            robot.move(action)
            reward = self.environment.queryRewardValue(robot.position)

            prevState = state
            prevAction = action
            prevReward = reward

            # Reevaluate epsilon if taper is set to True, and we have hit the threshold number of iteration for halving
            if taper and (i + 1) % halfEvery == 0:
                self.policy.epsilon /= 2
                print(f"Reached iteration {i}! Halving epsilon to {self.policy.epsilon}")
            # Continue simulation if the robot is not yet in a terminal state
            while not self.environment.isStateTerminal(prevState):
                state = robot.position 
                action = self.policy.chooseAction(QTable[state[0]][state[1]])
                prevStateActionPairValue = QTable[prevState[0]][prevState[1]][prevAction]
                # In the SARSA update step, we estimate the value of a state action pair using the sum of the rewards obtained by excuting that 
                # state action pair, and the gamma discounted QTable value of the next state action pair
                newStateActionPairValue = prevReward + self.gamma * QTable[state[0]][state[1]][action]
                # The QTable entry then steps towards this new value estimate by alpha * (the difference between the previous and new value etimate)
                QTable[prevState[0]][prevState[1]][prevAction] += self.alpha * (newStateActionPairValue - prevStateActionPairValue)

                robot.move(action) # We then move the robot,
                prevState = state # set the previous state,
                prevAction = action # set the previous action,
                prevReward = self.environment.queryRewardValue(robot.position) # and set the reward obtained by the previous state action pair

            if (i+1)%self.logStepsAndRewardsInterval == 0:
                self.progressTracker.stepsPerEpisodeTracker.update(robot.numStepsTaken, i+1)
                self.progressTracker.rewardPerEpisodeTracker.update(self.environment.queryRewardValue(prevState), i+1)
            
            if (i+1)%self.logQTableProgessInterval == 0:
                self.progressTracker.QTableTracker.update(self.QTable, i+1)
            

            robot.reset()

        self.QTable.printQTable()

class QLearning:
    def __init__(
        self, 
        environment: Grid,
        startPosition: Tuple[int, int],
        policy, 
        gamma: float = 0.9,
        alpha: float = 0.2,
        logStepsAndRewardsInterval: int = 100,
        logQTableProgessInterval: int = 10000,
    ):
        self.robot = Robot(startPosition, environment)
        self.startPosition = startPosition
        self.QTable = QTable(environment, self.robot.numPossibleActions, initializationType=1)
        self.environment = environment
        self.policy = policy
        self.gamma = gamma
        self.alpha = alpha

        self.logStepsAndRewardsInterval = logStepsAndRewardsInterval
        self.logQTableProgessInterval = logQTableProgessInterval
        self.progressTracker = ProgressTracker(environment, startPosition)

    def train(self, numIter: int = 500000, taper = True, halfEvery = 50000):
        robot = self.robot
        QTable = self.QTable.QTable
        self.progressTracker.QTableTracker.update(self.QTable, 1)
        for i in tqdm(range(numIter)):
            # Perform first step to prep SAR
            state = robot.position
            action = self.policy.chooseAction(QTable[state[0]][state[1]])
            robot.move(action)
            reward = self.environment.queryRewardValue(robot.position)

            prevState = state
            prevAction = action
            prevReward = reward

            if taper and (i + 1) % halfEvery == 0:
                print(f"Reached iteration {i + 1}! Halving epsilon to {self.policy.epsilon}")
                self.policy.epsilon /= 2
            while not self.environment.isStateTerminal(prevState):
                state = robot.position
                prevStateActionPairValue = QTable[prevState[0]][prevState[1]][prevAction]
                # In the Q Learning update step, we estimate the value of a state action pair using the sum of the rewards obtained by excuting that 
                # state action pair, and the gamma discounted MAXIMUM QTable value of the actions available at the next state
                newStateActionPairValue = prevReward + self.gamma * np.max(QTable[state[0]][state[1]])
                # The QTable entry then steps towards this new value estimate by alpha * (the difference between the previous and new value etimate)
                QTable[prevState[0]][prevState[1]][prevAction] += self.alpha * (newStateActionPairValue - prevStateActionPairValue)

                action = self.policy.chooseAction(QTable[state[0]][state[1]])
                robot.move(action) # We then move the robot,
                prevState = state # set the previous state,
                prevAction = action # set the previous action,
                prevReward = self.environment.queryRewardValue(robot.position) # and set the reward obtained by the previous state action pair

            if (i + 1)%self.logStepsAndRewardsInterval == 0:
                self.progressTracker.stepsPerEpisodeTracker.update(robot.numStepsTaken, i + 1)
                self.progressTracker.rewardPerEpisodeTracker.update(self.environment.queryRewardValue(prevState), i + 1)
            
            if (i + 1)%self.logQTableProgessInterval == 0:
                self.progressTracker.QTableTracker.update(self.QTable, i + 1)
            
            robot.reset()

        self.QTable.printQTable()        

if __name__ == "__main__":
    args = parser.parse_args()

    # Initialize environment
    environment = Grid()
    startPosition = (0, 0)

    if args.gridFile != "":
        environment.initGridFromFile(args.gridFile)
    elif args.width == 0 or args.height == 0:
        print("No width or height specified. Defaulting to basic 4x4 environment...")
        environment.initGridBasic()
    else:
        environment.initGridExtended(startPosition, args.height, args.width)

    print("Environment: (X represents thin ice region; G represents goal)")
    environment.printGrid()
    print("-----------------------------") 

    # Initialize policy
    policy = None
    
    if args.policy == "episilonGreedy":
        policy = epsilonGreedyPolicy(args.epsilon)
    else:
        policy = greedyPolicy()

    # Carry out RL
    RLSolver = None
    if args.algorithm == "SARSA":
        RLSolver = SARSA(
            environment,
            startPosition,
            policy = policy,
            gamma = args.gamma,
            alpha = args.alpha,
            logStepsAndRewardsInterval = args.dataLogInterval,
            logQTableProgessInterval = args.QTableLogInterval
        )

        RLSolver.train(
            numIter = args.numIters,
            taper = args.taperEpsilon,
            halfEvery = args.halfEvery
        )
    elif args.algorithm == "Q":
        RLSolver = QLearning(
            environment,
            startPosition,
            policy = policy,
            gamma = args.gamma,
            alpha = args.alpha,
            logStepsAndRewardsInterval = args.dataLogInterval,
            logQTableProgessInterval = args.QTableLogInterval,
            noUselessMoves = args.noUselessMoves,
            penalizeBackTracking = args.penalizeBackTracking
        )

        RLSolver.train(
            numIter = args.numIters,
            taper = args.taperEpsilon,
            halfEvery = args.halfEvery
        )
    elif args.algorithm == "FVMC":
        RLSolver = firstVisitMonteCarlo(
            environment,
            startPosition,
            policy = policy,
            gamma = args.gamma,
            logStepsAndRewardsInterval = args.dataLogInterval,
            logQTableProgessInterval = args.QTableLogInterval,
            noUselessMoves = args.noUselessMoves,
            penalizeBackTracking = args.penalizeBackTracking
        )

        RLSolver.train(
            batchSize = args.batchSize, 
            numIter = args.numIters, 
            adjustmentRate = args.adjustmentRate,
            prune = args.prune,
            taper = args.taperEpsilon,
            halfEvery = args.halfEvery
        )

    # Save results
    saveDir = args.saveDir
    if saveDir == "":
        saveDir = args.algorithm

    print(f"Saving results to ./{saveDir}...")
    RLSolver.progressTracker.saveProgressData(saveDir)

    # Save grid file
    if args.saveGridFile != "":
        environment.saveGridToFile(args.saveGridFile)

    print("done!")