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
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

#The common actions for offensive agent and denfesive agent
class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.walls = gameState.getWalls()
  
  #Contain all acitons for agents except STOP in each positons
  def getPosActions(self, Position):
    x,y = Position
    possibleActions = []
    for action in [Directions.NORTH,Directions.SOUTH,Directions.EAST,Directions.WEST]:
      if action == Directions.NORTH:
        dx,dy = 0,1
      elif action == Directions.SOUTH:
        dx,dy = 0,-1
      elif action == Directions.EAST:
        dx,dy = 1,0
      elif action == Directions.WEST:
        dx,dy = -1,0
      nextx,nexty = int(x+dx), int(y+dy)
      if not self.walls[nextx][nexty]:
        possibleActions.append(action)
    return possibleActions

  #Obtain the distances between pacman and defensiver
  def distToDefensiver(self, gameState): 
    myPos = gameState.getAgentState(self.index).getPosition()
    if myPos != None:
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      defensivers = [d for d in enemies if not d.isPacman and d.getPosition() != None and d.scaredTimer <= 2]
      if len(defensivers) > 0:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defensivers]
        return min(dists)
    return 100
  
  #Obtain the distance between the defensiver and the nearest capsules of agent
  def distFromDefToCap(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    if myPos != None:
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      defensivers = [d for d in enemies if not d.isPacman and d.getPosition() != None and d.scaredTimer <= 2]
      if len(defensivers) > 0:
        capPoss = self.getCapsules(gameState)
        if len(capPoss) > 0:
          dists = [self.getMazeDistance(myPos, capPos) for capPos in capPoss]
          minValue = min(dists)
          caps = [c for c, d in zip(capPoss, dists) if d == minValue]
          dis = [self.getMazeDistance(d.getPosition(), caps[0]) for d in defensivers]
          return min(dis)
    return 0

  '''
  Decide if the current postion is the dead road.
  Pitfalls: 
  1. Only the dead road that the max deepth is 6. Otherwise, the time is overload.
  2. If three directions of the position that have four excutable actions except STOP are dead roads, this situation cannot be thought as dead road. Time will be over.
  3. The agent wrongly regards the defensive as a walls.
  '''
  def isDeadLine(self, beforeAction, gameState, i,j):
    i = i + 1
    if i > 6 or j > 3:
      return 0
    myPos = gameState.getAgentState(self.index).getPosition()
    if myPos != None:
      actions = self.getPosActions(myPos)
    else:
      return 0
    length = 0
    if len(actions) == 1:
      return i
    elif len(actions) == 2:
      if i == 1:
        nextAction1 = actions[0]
        nextAction2 = actions[1]
        successor1 = self.getSuccessor(gameState,nextAction1)
        if successor1.getAgentState(self.index).getPosition() != None:
          nextPos1 = self.isDeadLine(nextAction1,successor1,i,j)
        else: nextPos1 = 0
        successor2 = self.getSuccessor(gameState,nextAction2)
        if successor2.getAgentState(self.index).getPosition() != None:
          nextPos2 = self.isDeadLine(nextAction2,successor2,i,j)
        else: nextPos2 = 0
        if nextPos1 != 0: 
          length = nextPos1
        elif nextPos2 != 0:
          length = nextPos2
        else:
          length = 0
      else:
        if Directions.REVERSE[beforeAction] in actions:
          actions.remove(Directions.REVERSE[beforeAction])
        nextAction = actions[0]
        successor = self.getSuccessor(gameState,nextAction)
        if successor.getAgentState(self.index).getPosition() != None:
          nextPos = self.isDeadLine(nextAction,successor,i,j)
        else: nextPos = 0
        if nextPos != 0: 
          length = nextPos
        else:
          length = 0
      return length
    elif len(actions) == 3:
      j = j + 1
      if i == 1:
        nextAction1 = actions[0]
        nextAction2 = actions[1]
        nextAction3 = actions[2]
        successor1 = self.getSuccessor(gameState,nextAction1)
        if successor1.getAgentState(self.index).getPosition() != None:
          nextPos1 = self.isDeadLine(nextAction1,successor1,i,j)
        else: nextPos1 = 0
        successor2 = self.getSuccessor(gameState,nextAction2)
        if successor2.getAgentState(self.index).getPosition() != None:
          nextPos2 = self.isDeadLine(nextAction2,successor2,i,j)
        else: nextPos2 = 0
        successor3 = self.getSuccessor(gameState,nextAction3)
        if successor3.getAgentState(self.index).getPosition() != None:
          nextPos3 = self.isDeadLine(nextAction3,successor3,i,j)
        else: nextPos3 = 0
        if nextPos1 != 0 and nextPos2 != 0:
          length = max(nextPos1,nextPos2)
        elif nextPos2 != 0 and nextPos3 != 0:
          length = max(nextPos2,nextPos3)
        elif nextPos1 != 0 and nextPos3 != 0:
          length = max(nextPos1,nextPos3)
        else:
          length = 0
      else:
        if Directions.REVERSE[beforeAction] in actions:
          actions.remove(Directions.REVERSE[beforeAction])
        nextAction1 = actions[0]
        nextAction2 = actions[1]
        successor1 = self.getSuccessor(gameState,nextAction1)
        if successor1.getAgentState(self.index).getPosition() != None:
          nextPos1 = self.isDeadLine(nextAction1,successor1,i,j)
        else: nextPos1 = 0
        successor2 = self.getSuccessor(gameState,nextAction2)
        if successor2.getAgentState(self.index).getPosition() != None:
          nextPos2 = self.isDeadLine(nextAction2,successor2,i,j)
        else: nextPos2 = 0
        if nextPos1 != 0 and nextPos2 != 0: 
          length = max(nextPos1,nextPos2)
        else:
          length = 0
      return length
    else:
      return 0

  #Return the distance of the pacman who carry large number of foods
  def distToLargeCarry(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    if myPos != None:
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      largeCarry = [d for d in enemies if d.isPacman and d.getPosition() != None and d.numCarrying > 5]
      if len(largeCarry) > 0:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in largeCarry]
        return min(dists)
    return 0

  #Return the distance of the capsules which in the enemy range.
  def distToCapsule(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    if myPos != None:
      capPoss = self.getCapsules(gameState)
      if len(capPoss) > 0:
        dists = [self.getMazeDistance(myPos, capPos) for capPos in capPoss]
        return min(dists)
    return 0
  
  #Return the number of the rest of capsules in the enemy range.
  def numOfCapsule(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    if myPos != None:
      capPoss = self.getCapsules(gameState)
      return len(capPoss)
    return 0

  #Return the distance between the agent and the enemy offensiver. If it does not find the offensiver, return the distance between the agent and the defensive.
  def distToPacman(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    if myPos != None:
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      pacmanPos = [d for d in enemies if d.isPacman and d.getPosition() != None]
      if len(pacmanPos) > 0:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in pacmanPos]
        return min(dists)
      notPacmanPos = [d for d in enemies if not d.isPacman and d.getPosition() != None]
      if len(notPacmanPos) > 0:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in notPacmanPos]
        return min(dists)
    return 20

  #Return the distance of another partner.
  def distToPartner(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    if myPos != None:
      teams = self.getTeam(gameState)
      teams.remove(self.index)
      partner = teams[0]
      partnerPos = gameState.getAgentState(partner).getPosition()
      if partnerPos != None :
      #and not gameState.getAgentState(self.index).isPacman:
        dists = self.getMazeDistance(myPos, partnerPos)
        return dists
    return 0

  #Return the number of carring foods.
  def foodCarrying(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    if myPos != None:
        return gameState.getAgentState(self.index).numCarrying
    return 0

  #Returen the shorest distance to home.
  def distToHome(self, gameState):
    myIspacman = gameState.getAgentState(self.index).isPacman
    myPos = gameState.getAgentState(self.index).getPosition()
    if self.red:
        boundary = int((gameState.data.layout.width - 2) / 2)
    else:
        boundary = int(((gameState.data.layout.width - 2) / 2) + 1)
    boundaryList = []
    for i in range(1, gameState.data.layout.height - 1):
        if not gameState.hasWall(boundary,i):
            boundaryList.append((boundary, i))
    if myIspacman and myPos != None:
      homeDistance = [self.getMazeDistance(myPos, b) for b in boundaryList]
      return min(homeDistance)
    else:
      return 0

  #Return the position of the food which is the nearest to the boundary.
  def foodNearestBoundary(self, gameState):
    myFoodList = self.getFoodYouAreDefending(gameState).asList()
    if self.red:
        boundary = int(((gameState.data.layout.width - 2) / 2) + 1)
    else:
        boundary = int((gameState.data.layout.width - 2) / 2)
    foodPos = None
    boundaryNearestFood = 9999
    for i in range(1, gameState.data.layout.height - 1):
        if not gameState.hasWall(boundary,i):
            for food in myFoodList:
              dictToBoundary = self.getMazeDistance(food,(boundary,i))
              if dictToBoundary < boundaryNearestFood:
                boundaryNearestFood = dictToBoundary
                foodPos = food
    return foodPos

  #Return the sum of the distance of the own foods.
  def DistToAllFood(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    myFoodList = self.getFoodYouAreDefending(gameState).asList()
    dist = 0
    for food in myFoodList:
      dist = dist + self.getMazeDistance(food,myPos)
    return dist

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
    

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.walls = gameState.getWalls()

  #Obtain the values of each features.
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()   
    features['successorScore'] = -len(foodList)

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      if action == rev: features['reverse'] = 1
      else : features['reverse'] = 0
      features['distanceToFood'] = minDistance
      features['isDeadLine'] = self.isDeadLine(action,successor, 0, 0)
      features['distanceToHome'] = self.distToHome(successor)
      features['distanceToLargeCarry'] = self.distToLargeCarry(successor)
      features['foodCarry'] = self.foodCarrying(successor)
      features['distanceToDefensiver'] = self.distToDefensiver(successor)
      features['distanceToCapsule'] = self.distToCapsule(successor)
      features['numOfCapsule'] = self.numOfCapsule(successor)
      features['distanceToPartner'] = self.distToPartner(successor)
      features['distFromDefToCap'] = self.distFromDefToCap(successor)

    return features

  #Obtain the weights of each features
  def getWeights(self, gameState, action):
    
    weights = util.Counter()
    weights['successorScore'] = 1000
    weights['distanceToFood'] = -16
    weights['distanceToLargeCarry'] = -80
    weights['distanceToCapsule'] = -10
    weights['numOfCapsule'] = -20000
    weights['reverse'] = -50
    weights['distanceToPartner'] = 10

    return weights
  
  #Return the weights of each action
  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    nowFeatures = self.getFeatures(gameState, Directions.STOP)
    evaluation = 0
    if features['isDeadLine'] > 7:
      features['isDeadLine'] = 0
    for key, value in weights.items():
      evaluation = evaluation + value * features[key]
    #If the offensiver are carrying the large amount of foods, it is strongly willing to go back home.
    #5 is the result decided by adjustion.
    evaluation = evaluation - (features['distanceToHome'] * features['foodCarry'] * 5)
    #If the offensiver have obtained enough high score, it is strongly willing to go back home.
    if features['successorScore'] == -2 or (gameState.data.timeleft/4) < (nowFeatures['distanceToHome']+15):
      evaluation = evaluation + 500 - (features['distanceToHome'] * 50)
    #If the offensiver are around by the opponent defensiver
    if nowFeatures['distanceToDefensiver'] < 10 and not gameState.getAgentState(self.index).isPacman:
      evaluation = evaluation + (100 * features['distanceToDefensiver'])
    if features['distanceToDefensiver'] < 7:
      '''
      If the distance between the offensiver and the nearest capsule is less than the distance of the defensiver 
      and the nearest capsule of the offensiver, the offensiver will eat the capsule.
      '''
      if features['distanceToDefensiver'] == 1:
        evaluation = evaluation - 1000
      if nowFeatures['distanceToCapsule'] < nowFeatures['distFromDefToCap']:
        evaluation = evaluation - (700 * features['distanceToCapsule'])
      #If the defensiver is close to the offensiver, the offensiver will escape.  
      evaluation = evaluation - 3500 + (500 * features['distanceToDefensiver'])
      evaluation = evaluation - (20 * features['distanceToCapsule'])
      #If the current position and the next step all are the dead roads, the offensiver will escape.
      if features['isDeadLine'] != 0 and nowFeatures['isDeadLine'] != 0:
        evaluation = evaluation + 3500 * features['isDeadLine']
      #If the current position is dead road and the next step are not dead road, the offensiver will escape.
      if nowFeatures['isDeadLine'] != 0 and features['isDeadLine'] == 0:
        evaluation = evaluation + 20000
      #If the current position is not dead road and the next step are dead road, the offensiver will decide to go on or escape according to the possibility.
      if nowFeatures['isDeadLine'] == 0 and features['isDeadLine'] != 0:
        if features['distanceToDefensiver'] <= (2*features['isDeadLine']+1):
          evaluation = evaluation - 20000
    else:
      evaluation = evaluation - 285
    return evaluation

  #2-step temporal difference
  def simulation(self, depth, gameState, action, decay):
      if depth == 0:
        return self.evaluate(gameState, action)
      else:
        simuResult = []
        newState = self.getSuccessor(gameState,action)
        actions = newState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        for newaction in actions:
          simuResult.append(self.evaluate(gameState, action) + (decay * self.simulation(depth - 1, newState, newaction, decay)))
        return max(simuResult)

  #Choose the action with the highest weights.
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest weights.
    """
    actions = gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    values = [self.simulation(1,gameState, a, 0.001) for a in actions]
    maxResult = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxResult]
    chosenAction = random.choice(bestActions)

    return chosenAction

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def __init__(self, index):
    ReflexCaptureAgent.__init__(self, index)
    self.i = 0

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.walls = gameState.getWalls()

  #Return the values of the feature
  '''
  Benefits: 
  1. At the beginning, the defensiver walks around the boundary. 
  If it does not find the offensiver, it will go to the place where many foods exist.
  2. The defensiver will calculate the capsule which the offensiver mostly is willing to eat. 
  It will go and protect the foods.
  '''
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    nearestFood = self.foodNearestBoundary(successor)

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    myFoodList = self.getFoodYouAreDefending(gameState).asList()
    features['numInvaders'] = len(invaders)
    if myPos != None and nearestFood != None:
      features['nearestfoodDistance'] = self.getMazeDistance(nearestFood,myPos)
    if features['nearestfoodDistance'] < 2:
      self.i = self.i + 1
    if self.i > 10:
      features['nearestfoodDistance'] = 0
    foodPos = None
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
      enemyNearestFood = 9999
      for enemy in invaders:
        for food in myFoodList:
          distToFood = self.getMazeDistance(enemy.getPosition(),food)
          if distToFood < enemyNearestFood:
            enemyNearestFood = distToFood
            foodPos = food
      if myPos != None and foodPos != None:
        features['nearestfoodDistance'] = self.getMazeDistance(myPos,foodPos)
    else:
      features['invaderDistance'] = 100
    if action == Directions.STOP: features['stop'] = 1
    else : features['stop'] = 0
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    else : features['reverse'] = 0
    features['distToPacman'] = self.distToPacman(successor)
    features['isDeadLine'] = self.isDeadLine(action,successor, 0, 0)
    features['distanceToLargeCarry'] = self.distToLargeCarry(successor)
    features['distanceToAllFood'] = self.DistToAllFood(successor)

    return features

  #Return the weights of the features
  def getWeights(self, gameState, action):
    weights = util.Counter()
    weights['numInvaders'] = -3000
    weights['nearestfoodDistance'] = -20
    weights['onDefense'] = 2000
    weights['reverse'] = -40
    weights['distanceToLargeCarry'] = -25
    weights['stop'] = -40
    weights['invaderDistance'] = -40
    return weights
  
  #Return the weights of each action
  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    nowFeatures = self.getFeatures(gameState, Directions.STOP)
    evaluation = 0
    for key, value in weights.items():
      evaluation = evaluation + value * features[key]
   
    #If the defensiver does not find the offensiver, it will go around the foods.
    if features['invaderDistance'] == 100:
      evaluation = evaluation - (2*features['distanceToAllFood'])
    #If the defensiver is sacared, it will keep the distance with the offensive as 2.
      #If the defensiver is sacard and escaping, it will avoid to go in the dead road.
    if gameState.getAgentState(self.index).scaredTimer > 0 and features['isDeadLine'] != 0:  
      evaluation = evaluation - 2000
    elif gameState.getAgentState(self.index).scaredTimer > 0 and features['invaderDistance'] <= 2:
      evaluation = evaluation + (1500*features['distToPacman'])
    if nowFeatures['distToPacman'] < 8:
      #If the defensiver is not sacared, it will be willing to eat the pacman and avoid to go in the dead road.
      evaluation = evaluation - (40 * features['distToPacman'])

    if features['isDeadLine'] != 0:
      evaluation = evaluation - 30
    return evaluation

  #2-step temporal difference
  def simulation(self, depth, gameState, action, decay):
      if depth == 0:
        return self.evaluate(gameState, action)
      else:
        simuResult = []
        newState = self.getSuccessor(gameState,action)
        actions = newState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        for newaction in actions:
          simuResult.append(self.evaluate(gameState, action) + (decay * self.simulation(depth - 1, newState, newaction, decay)))
        return max(simuResult)

  #Choose the action with the highest weights.
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    actions = gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    values = [self.simulation(1,gameState, action, 0.001) for action in actions]
    maxResult = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxResult]
    chosenAction = random.choice(bestActions)
    
    return chosenAction