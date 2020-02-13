# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import copy
from util import nearestPoint
from game import Actions


#################
# Team creation #
#################

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='QLAgent', second='AstarAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

#using the skeleton me from Reinforcement learning
class QLAgent(CaptureAgent):

#for training process and give initial value to the vars

    def __init__(self, index):

        self.epsilon = 0.3
        self.alpha = 0.8
        self.discount = 0.7
        self.weights = util.Counter()

        self.index = index

        self.food = 0

        self.observationHistory = []

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).
        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)
        IMPORTANT: This method may run for at most 15 seconds.
        """
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)

        self.atePac = False
        self.chasePac= False

        self.actionChoice = util.Queue()
        self.QLT = {}

    """
    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        action = None
        return action
    """

    def chooseAction(self, gameState):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        actionList = ['North', 'South', 'West', 'East']
        goal = False
        currNodes = util.Queue()
        visitedNum = 0
        alpha = 0.5
        currNodes.push(self.getCurrentObservation())
        exploredNodes = []
        discount = 0.7

        #conditions of win
        if len(self.getFood(gameState).asList()) <= 2:
            self.chasePac = True

        while not currNodes.isEmpty() and not goal:

            currentState = currNodes.pop()
            visitedNum = visitedNum + 1
            currList = currentState.getAgentState(self.index).getPosition()
            exploredNodes.append(currList)
            qList = self.QLT
            actions = currentState.getLegalActions(self.index)

            if Directions.STOP in currentState.getLegalActions(self.index):
                actions.remove(Directions.STOP)
            for action in actions:
                self.actionChoice.push(action)
            #conditions of finishing exploration
            if visitedNum > 10:
                goal = True

            #
            if not (currList in qList.keys()):
                qList[currList] = [0, 0, 0, 0]

            while not self.actionChoice.isEmpty():
                action = self.actionChoice.pop()
                actionID = actionList.index(action)
                nextActions = currentState.generateSuccessor(self.index, action).getLegalActions(self.index)

                successor = currentState.generateSuccessor(self.index, action)

                currentq = max([self.getReward(successor, nextAct) for nextAct in nextActions])

                if not self.chasePac and \
                        len(self.getFood(currentState).asList()) - len(self.getFood(successor).asList()) == 1:
                    reward = 90
                else:
                    reward = 0

                qList[currList][actionID] = 0.5 * qList[currList][actionID] \
                                                       + alpha * (reward + discount * currentq)
                if currentState.generateSuccessor(self.index, action).getAgentPosition(self.index) not in exploredNodes:
                    currNodes.push(currentState.generateSuccessor(self.index, action))

        actionSet = gameState.getLegalActions(self.index)
        currList = gameState.getAgentState(self.index).getPosition()
        qList1 = self.QLT

        if random.uniform(0, 1) >= 0.01:
            maxValue = -999999

            for var in range(len(qList1[currList])):
                if qList1[currList][var] != 0 and qList1[currList][var] > maxValue:
                    maxValue = qList1[currList][var]
            bestAction = actionList[qList1[currList].index(maxValue)]
        else:
            bestAction = random.choice(actionSet)


        successor = gameState.generateSuccessor(self.index, bestAction)
        successorPos = successor.getAgentPosition(self.index)

        op = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        gh = [agent for agent in op
              if agent.getPosition() != None and (not agent.isPacman)]
        pos = successor.getAgentState(self.index).getPosition()

        if len(gh) > 0:
            dists = [self.getMazeDistance(pos, agent.getPosition()) for agent in gh]
            if not self.chasePac:
                if min(dists) > 5:
                    self.chasePac = False
                else:
                    if successor.getAgentState(self.index).isPacman:
                        self.chasePac = True
        else:
            self.chasePac = False

        if pos == self.start:
            self.atePac = True
            self.chasePac = False
            self.atePac = False

        if self.chasePac or successorPos in self.getFood(gameState).asList() \
                and successorPos not in self.getFood(successor).asList():
            self.QLT = {}

        return bestAction

    #train the value q
    def getQValue(self, state, action):

        stateFeatures = self.getFeatures(state, action)
        stateWeights = self.getWeights(state, action)
        acc = 0
        for feature in stateFeatures:
            acc +=  (stateFeatures[feature] * stateWeights[feature])
        return acc

    #used for training the model
    def computeValueFromQValues(self, state):
        checkActs = True
        bestValue = -999999
        for action in state.getLegalActions(self.index):
            checkActs = False
            value = self.getQValue(state, action)
            if value > bestValue:
                bestValue = value
        if checkActs:
            return 0
        return bestValue
    '''
    for training process
    def update(self, state, action, nextState, reward):

        #stateFeatures = self.getFeatures(state, action)
        #currentValues = self.getWeights(state, action)
        nextAction = self.computeActionFromQValues(nextState)
        if nextAction == None:
            successorQValue = 0
        else:
            successorQValue = self.getQValue(nextState, nextAction)
        difference = (reward + self.discount * successorQValue) - self.getQValue(state, action)
        for feature in stateFeatures.keys():
            currentValues[feature] = currentValues[feature] + (self.alpha * difference * stateFeatures[feature])
        self.weights = currentValues.copy()
    '''
    def computeActionFromQValues(self, state):
        bestActions = None
        bestValue = -999999
        for action in state.getLegalActions(self.index):
            value = self.getQValue(state, action)
            if value > bestValue and bestValue != 0:
                bestActions = [action]
                bestValue = value
            elif value == bestValue:
                bestActions.append(action)
        return bestActions

    def getFeatures(self, gameState, action):

        successor = self.getSuccessor(gameState, action)
        features = util.Counter()

        pos = self.getSuccessor(gameState, action).getAgentState(self.index).getPosition()
        dist = [self.getMazeDistance(pos, food)
                           for food in self.getFood(successor).asList()]
        features['distToFood'] = min(dist)

        currState = successor.getAgentState(self.index)
        gh = [agent for agent in [successor.getAgentState(i) for i in self.getOpponents(successor)]
                  if agent.getPosition() != None and not agent.isPacman]

        pos = currState.getPosition()
        if len(gh) == 0:
            features['distToOp'] = 100
        else:
            features['distToOp'] = min([self.getMazeDistance(pos, agent.getPosition()) for agent in gh])

        features['currTos'] = self.getMazeDistance(pos, self.start)
        return features

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getWeights(self, gameState, action):

        if not self.chasePac:
            return {'distToFood': -1,
                    'distToOp': 0,
                    'currTos': 0}
        #successor = self.getSuccessor(gameState, action)
        #features = util.Counter()
        else:
            return {'distToFood': 0,
                    'distToOp': 12,
                    'currTos': -2}

    def getReward(self, state, action):
        """
        Computes a reward given the features and weights
        """
        features = self.getFeatures(state, action)
        weights = self.getWeights(state, action)
        return features * weights



class AstarAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)

        #Food related variables
        self.distToLostFood = 0
        self.lostFood = False
        self.posFood = None

        #Opponents related variable
        self.detectOp = False
        self.posOp = None

        self.detectGh = False


    def ghostDetect(self, gameState, action):
        if self.getSuccessor(gameState, action).getAgentState(self.index).isPacman:
            self.detectGh = True


    def foodLost(self, gameState, DistanceToMisFood):

        if self.getPreviousObservation() != None:
            for food in self.getFoodYouAreDefending(self.getPreviousObservation()).asList():
                if food not in self.getFoodYouAreDefending(gameState).asList():
                    self.lostFood = True
                    self.posFood = food
                else:
                    if DistanceToMisFood == 1:
                        self.lostFood = False
                    elif DistanceToMisFood > 1:
                        self.lostFood = True
                return

    def aStar(self, gameState, goalPoint):

        currentPos = gameState.getAgentPosition(self.index)

        currentExp = util.PriorityQueue()
        explored = []
        currentExp.push([gameState, []], 0)

        while not currentExp.isEmpty():
            pElement = currentExp.pop()
            currentState = pElement[0]
            path = pElement[1]

            if currentState.getAgentPosition(self.index) == goalPoint:
                bestActionList = path
                break

            if currentState in explored:
                continue
            else:
                explored.append(currentState)
            for action in currentState.getLegalActions(self.index):

                successor = currentState.generateSuccessor(self.index, action)
                successorPos = successor.getAgentPosition(self.index)
                item = [successor, path + [action]]

                g = self.getMazeDistance(currentPos, successorPos)
                heur = self.getMazeDistance(successorPos, goalPoint)
                f = g + heur

                currentExp.push(item, f)

        return bestActionList

    def chooseAction(self, gameState):
        actionList = gameState.getLegalActions(self.index)

        if Directions.STOP in actionList:
            actionList.remove(Directions.STOP)

        self.foodLost(gameState, self.distToLostFood)

        if self.detectOp:

            self.opDetect(gameState, self.aStar(gameState, self.posOp)[0])

            if not self.detectOp:
                self.distToLostFood = 0
                self.lostFood = False

            return self.aStar(gameState, self.posOp)[0]

        elif self.lostFood:

            self.distToLostFood = len(self.aStar(gameState, self.posFood))

            self.opDetect(gameState, self.aStar(gameState, self.posFood)[0])
            return self.aStar(gameState, self.posFood)[0]

        else:
            if self.getPreviousObservation() == None:
                return random.choice(actionList)
            else:
                if not self.detectGh:
                    if len(self.aStar(gameState,
                                      self.getPreviousObservation().getAgentPosition(self.index))) == 1:
                        if self.aStar(gameState, self.getPreviousObservation().getAgentPosition(self.index))[0] \
                                in actionList and len(actionList) != 1:

                            actionList.remove(self.aStar(gameState,
                                                         self.getPreviousObservation().getAgentPosition(self.index))[0])

                    action = random.choice(actionList)
                    self.ghostDetect(gameState, action)

                    if self.detectGh:
                        actionList.remove(action)
                        self.detectGh = False

                        if not actionList:
                            action = Directions.STOP
                        else: #process the empty actionList
                            action = random.choice(actionList)

                    self.opDetect(gameState, action)
                    return action

    def opDetect(self, gameState, action):

        pos = self.getSuccessor(gameState, action).getAgentPosition(self.index)
        oPs = [self.getSuccessor(gameState, action).getAgentState(i)
                   for i in self.getOpponents(self.getSuccessor(gameState, action))]
        pacs = [a for a in oPs if a.getPosition() != None and a.isPacman]

        if not pacs:
            self.detectOp = False
        else:
            self.detectOp = True
            distanceList = [self.getMazeDistance(pos, a.getPosition()) for a in pacs]
            index = distanceList.index(min(distanceList))
            self.posOp = pacs[index].getPosition()



    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
