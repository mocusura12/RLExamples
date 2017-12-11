
# coding: utf-8

# In[1]:


import numpy as np
np.random.choice([0,-1,-2])
Qsol = np.load('Qsol.npy')


# In[2]:


# Ennvironment setup
# Variables
#   sizeX, sizeY : size of the gridworld
# 
# Functions
#   stateNum : input(x,y)= coordinate, output= state number
#   stateNumToCoord : input(st)= state number, output[x,y]= coordinate
#   keepInside : input(x,y)= coordinate, output[x,y]= coordinate
#   moveAgent : input(x,y,a)= coordinate, action, output[x,y]=coordinate
#   wind : input(x,y)=coordinate, output[x,y]=coordinate

def softmax(x, k=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(k*x - np.max(k*x))
    return e_x / e_x.sum(axis=0) # only difference

sizeX = 6
sizeY = 5
nstate = sizeX*sizeY
naction =4
def stateNum(x, y):
    return(int(x + (y-1) *sizeX -1))

dicAction = {'up':1, 'left':2, 'down':3, 'right':4}
dicAction2 = {1:'up', 2:'<-', 3:'dn', 4:'->'}

def stateNumToCoord(st):
    return([st % sizeX +1, 1+st // sizeX])  # //, %  division and the remainder

def keepInside(x, y):    
    if (x<1): 
        x=1
    elif (x>sizeX):
        x=sizeX
        
    if (y<1):
        y=1
    elif (y>sizeY):
        y=sizeY
        
    return([x,y])

goalState = stateNum(4,3)



def moveAgent(x, y, a):
   # if (stateNum(x, y) == goalState):
   #     return([x,y])
    if a in dicAction:
        a = dicAction[a]
    
    if (a == 1):
        y = y - 1
    if (a == 2):
        x = x - 1
    if (a == 3):
        y = y + 1
    if (a == 4):
        x = x + 1
    
    return(keepInside(x,y))

def wind(x,y):    
    if (x in [2,5]):
        
        y=y-np.random.choice([0,1,2])
    if (x in [3,4]):
        y=y-np.random.choice([1,2,3])
    return(keepInside(x,y))

def prMoveAgent(prIstate, a):
   # if (stateNum(x, y) == goalState):
   #     return([x,y])
    cumPr = np.zeros(nstate)
    cumPr[goalState] = prIstate[goalState]
    for istate in range(nstate):
        if (istate != goalState and prIstate[istate] > 0):
            x, y = stateNumToCoord(istate)
            pr = np.zeros(nstate)
               
            if a in dicAction:
                a = dicAction[a]
    
            if (a == 1):
                y = y - 1
            if (a == 2):
                x = x - 1
            if (a == 3):
                y = y + 1
            if (a == 4):
                x = x + 1
            x, y = keepInside(x,y)
            pr[stateNum(x,y)]=1
            cumPr = cumPr + prIstate[istate]*pr
            
    return(cumPr)
    

def prWind(prIstate):
    cumPr = np.zeros(nstate)    
    cumPr[goalState] = prIstate[goalState]
    for istate in range(nstate):
        if (istate != goalState and prIstate[istate] > 0):
            x, y = stateNumToCoord(istate)                
            pr = np.zeros(nstate)   
            if (x in [2,5]):
                x = np.ones(3)*x
                y = np.array([y, y-1, y-2])     
            elif (x in [3,4]):
                x = np.ones(3)*x
                y = np.array([y-1, y-2, y-3])     
            else:
                x = np.array([x])
                y = np.array([y])
            prob = 1/x.shape[0]            
            for i in range(x.shape[0]):                
                xTemp, yTemp = keepInside(x[i], y[i])
                pr[stateNum(xTemp, yTemp)]=pr[stateNum(xTemp, yTemp)]+prob
            
            cumPr = cumPr + prIstate[istate]*pr   
    
    
    
    return(cumPr)


# In[3]:


class prWindyGW:
    def __init__(self, x=None, y=None):
        self.sizeX = 6
        self.sizeY = 5
        self.nstate = self.sizeX*self.sizeY
        self.naction = 4
        self.goalState = stateNum(4,3)
        if (x == None):
            while True:
                self.x, self.y = np.random.choice(range(sizeX))+1, np.random.choice(range(sizeY))+1
                if stateNum(self.x,self.y) != goalState:
                    break
        elif (y == None): 
            print("warning x is not given but y is given!!!")
            self.x = x
            self.y = np.random.choice(range(sizeY))+1
        else:
            self.x = x
            self.y = y
        #self.obs = stateNum(self.x,self.y)          
    
    def moveAgent(self, a):    
        if (a == 1):
            self.y = self.y - 1
        if (a == 2):
            self.x = self.x - 1
        if (a == 3):
            self.y = self.y + 1
        if (a == 4):
            self.x = self.x + 1
                
    def keepInside(self):
        if (self.x<1): 
            self.x=1
        elif (self.x>self.sizeX):
            self.x=self.sizeX
        
        if (self.y<1):
            self.y=1
        elif (self.y>self.sizeY):
            self.y=self.sizeY              
        
    def wind(self):
        if (self.x in [2,5]):
            self.y=self.y-np.random.choice([0,1,2])
        if (self.x in [3,4]):
            self.y=self.y-np.random.choice([1,2,3])
                
    def step(self,action):  # observation, reward, done, info
        if ([self.x, self.y] == stateNumToCoord(self.goalState)):
            print("It in the terminal State. Nothing will change.")
            return([self.goalState, 0, True, None])
        self.wind()        
        self.moveAgent(action)
        self.keepInside()
        obs = stateNum(self.x, self.y)
        reward = -1
        if (obs == goalState):
            done = True            
        else:
            done = False
        info=None
        self.x, self.y = stateNumToCoord(obs)
        return([obs, reward, done, info])     
                
    
    
    


# In[4]:


env = prWindyGW(1,3)

obs, reward, done, info = env.step(3)
print (env.x, env.y)
print(env.nstate)
vState = np.zeros(env.nstate)
vState[obs]=1
matState = np.reshape(vState, (env.sizeY, env.sizeX))
import matplotlib.pyplot as plt
plt.imshow(matState)
plt.show()


# In[5]:


obs, reward, done, info = env.step(3)
print (env.x, env.y)
vState = np.zeros(env.nstate)
vState[obs]=1
matState = np.reshape(vState, (env.sizeY, env.sizeX))
import matplotlib.pyplot as plt
plt.imshow(matState)
plt.show()
print(done)
print(goalState)
print(stateNumToCoord(goalState))


# In[7]:


# What should be epsilon?
# What should be alpha?

# epsilon : how much more exploration???
# alpha : learning rate...

epsilon=1
alpha = 0.3
Q = np.zeros((env.nstate, env.naction))
nQ = np.zeros((env.nstate, env.naction))
for i in range(10000):
    env = prWindyGW(1,3)

    totalReward =0
    

    #epsilon=epsilon*0.999
    alpha = max(alpha*0.99, 0.01)
    epsilon = max(epsilon*0.99, 0.01)
    
    stateBefore = None
    actionBefore = None
    iteration = 1
    while True:
        state = stateNum(env.x, env.y)
        vQ = Q[state,:]
        vQmax = np.ndarray.max(vQ)    
        actionlist = np.arange(naction)
        actionMax = np.random.choice(actionlist[vQ == vQmax])
        
        if (np.random.uniform(0,1)<epsilon):
            action = np.random.choice(range(naction))
        else:
            action = actionMax
            
        nQ[state, action] +=1
        obs, reward, done, info=env.step(action+1)
            
        if (stateBefore != None) and (actionBefore != None):
            #Q[state,actionBefore]= (1-alpha)*Q[state,actionBefore] + alpha*(reward + Q[obs,actionMax])
            #Q[state,actionBefore]= Q[state,actionBefore]+ alpha*(reward + Q[obs,actionMax]-Q[state,actionBefore])
            Q[state,action]= Q[state,action]+ alpha*(reward + max(Q[obs,:])-Q[state,action])
                            
        totalReward += reward
                
        if (done): 
            break;
            
        stateBefore = obs
        actionBefore = action
        #epsilon=epsilon*0.99
    if (i % 100 == 0):
        print(i, "-th iteration, with epsilon=",epsilon,", alpha=",alpha,", Total reward=", totalReward)
        print("SSQ =", np.sum((Q-Qsol)**2))


# In[8]:


env=prWindyGW(1,3)


# In[9]:


state = stateNum(env.x, env.y)
vState = np.zeros(env.nstate)
vState[state]=1
matState = np.reshape(vState, (env.sizeY, env.sizeX))
import matplotlib.pyplot as plt
plt.subplot(221)
plt.imshow(matState)

vQ = Q[state,:]

vQmax = np.ndarray.max(vQ)    

actionlist = np.arange(naction)
actionMax = np.random.choice(actionlist[vQ == vQmax])
        
if (np.random.uniform(0,1)<epsilon):
    action = np.random.choice(range(naction))
    print("Random")
else:
    action = actionMax
    print("Greedy")
            
#nQ[state, action] +=1
obs, reward, done, info=env.step(action+1)
            
if (stateBefore != None) and (actionBefore != None):
    #Q[state,actionBefore]= (1-alpha)*Q[state,actionBefore] + alpha*(reward + Q[obs,actionMax])
    print(stateNumToCoord(state), dicAction2[action+1])
    print("Qestimates=",Q[state,:])
    #Qsol
    print("Qsol      =",Qsol[state,:])
    #Q[state,actionBefore]= Q[state,actionBefore]+ alpha*(reward + Q[obs,actionMax]-Q[state,actionBefore])
    Q[state,action]= Q[state,action]+ alpha*(reward + max(Q[obs,:])-Q[state,action])
                           
totalReward += reward
            
stateBefore = obs
actionBefore = action

vState = np.zeros(env.nstate)
vState[obs]=1
matState = np.reshape(vState, (env.sizeY, env.sizeX))
import matplotlib.pyplot as plt
plt.subplot(222)
plt.imshow(matState)
plt.show()


# In[10]:


Q


# In[11]:


Q-Qsol


# In[12]:


nQ
plt.imshow(nQ)
plt.colorbar()
plt.show()


# In[13]:


nV = np.ndarray.mean(nQ, axis=1)
matnV = np.reshape(nV, (env.sizeY, env.sizeX))
plt.imshow(matnV)
plt.colorbar()
plt.show()


# In[14]:


state = 2
vQ = Q[state,:]
print(vQ)
vQmax = np.ndarray.max(vQ)    
actionlist = np.arange(naction)
actionMax = np.random.choice(actionlist[vQ == vQmax])
print(actionMax)


# In[15]:


vState = np.ndarray.mean(Q, axis=1)
matState = np.reshape(vState, (env.sizeY, env.sizeX))

import matplotlib.pyplot as plt
plt.imshow(matState)
plt.show()


# In[16]:


np.random.uniform(0,1)

