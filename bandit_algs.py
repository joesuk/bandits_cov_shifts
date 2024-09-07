# contextual bandit algorithms 
import numpy as np
import scipy
from scipy.spatial import distance
from scipy import integrate
from matplotlib import pyplot as plt
import math

######################################################
# Adaptive Bandits" algorithm from Suk & Kpotufe, 2021.
###################################################

# "Adaptive Bandits" algorithm from Suk & Kpotufe, 2021.
def adaptiveBandits(X, Y, n_P, n_Q, K, L):
    n = n_P + n_Q # total horizon

    if n == len(X[:,0]):
        start = 0
    else:
        start = len(X[:,0]) - n 

    Xn = np.take(X, list(range(start,len(X[:,0]))), 0) # covariates
    Yn = Y[start:] # rewards

    pulls = np.zeros(n,dtype=int) # arm pulls
    exp = math.ceil(max(8, 1 / (L ** 2)) * (K ** 2) * np.log(K * n)) # exploration phase

    previous_bins = [] # history of partitions/bins used [x0,y0,r,covariates,candidate_arms]
    S = [] # schedule of arm-pulls via round-robin

    for t in range(0,n):
        x = Xn[t,:] # current covariate
        
        # spend first exp rounds exploring
        if t + 1 <= exp: 
            pulls[t] = t % K
            if t + 1 == exp:
                previous_bins.append([0, 0, 1, list(range(0,t)), list(range(0,K))])
        else:
            r, N, C = getBin(previous_bins, x, t, K) # get current bin

            levelCheck = True
            neighbors = N
            finalNeighbors = []

            # check if only one candidate arm left
            if len(C) == 1:
                pulls[t] = C[0]
                continue

            r_t = r
            # check if level should be lowered
            while levelCheck:
                finalNeighbors = neighbors
                neighbors = getNeighbors(Xn, neighbors, r_t, t)
                binCount = len(neighbors)
                if binCount == 0:
                    levelCheck = False
                elif L * r_t < math.sqrt(1 / binCount):
                    if r_t < r: r_t = r_t * 2
                    levelCheck = False
                else:
                    r_t = r_t / 2

            estimates = np.zeros(K) # reward estimates
            ## compute estimates
            for i in C:
                nearby_pulls = []
                estimate = 0    
                for s in finalNeighbors:    
                    if pulls[s] == i:
                        nearby_pulls.append(s)
                        estimate += Yn[s,i]    
                pull_count = len(nearby_pulls)    
                if pull_count != 0:
                    estimates[i] = estimate / pull_count

            old_C = [i for i in C] # copy candidate arm list
            # eliminate arms
            for i in C:
                maxReward = max(estimates)
                if estimates[i] != 0 and (maxReward - estimates[i] > (L * r_t) / 2):
                    C.remove(i)

            # update partition history
            x0 = roundPartial(x[0],r_t)
            y0 = roundPartial(x[1],r_t)
            finalNeighbors.append(t)
            if r_t == r:
                previous_bins.remove([x0, y0, r_t, N, old_C])
                previous_bins.append([x0, y0, r_t, finalNeighbors, C])
            else:
                previous_bins.append([x0, y0, r_t, finalNeighbors, C])

            # play arm at random
            play = np.random.randint(0, len(C))
            pulls[t] = C[play] 

    return pulls

# get neighbors in child bin of parent
def getNeighbors(X, parentBin, r, t):
    x0 = roundPartial(X[t,0],r)
    y0 = roundPartial(X[t,1],r)
    thisBin = []
    if r == 1:
        return list(range(0,t))
    else:
        for i in parentBin:
            if binContain(X[i,:], x0, y0, r):
                thisBin.append(i)
        return thisBin

# check for containment of point X in bin centered at (x0,y0) of radius r.
def binContain(X, x0, y0, r):
    x = X[0]
    y = X[1]
    x1 = x0 + r
    y1 = y0 + r
    if x >= x0 and x < x1 and y >= y0 and y < y1:
        return True
    else:
        return False

# round number to nearest value according to a (dyadic) resolution
def roundPartial(value, resolution):
    if resolution == 0:
        return 0
    return math.floor(value / resolution) * resolution

# get current bin of covariate x
def getBin(previous_bins, x, t, K):
    min_r = 2
    diff = 2
    min_N = list(range(0,t))
    min_C = list(range(0,K))

    for i in range((len(previous_bins)-1),-1,-1):
        b = previous_bins[i]
        x0 = b[0]
        y0 = b[1]
        r = b[2]
        if binContain(x, x0, y0, r) and r < min_r and diff > 0:
            min_r = r
            min_N = b[3]
            min_C = [i for i in b[4]]
            diff = diff - 1
            if diff == 0: break
    return min_r, min_N, min_C


##########################################
# Contextual zooming algorithm of Slivkins, 2014
##########################################

# "Contextual Zooming" algorithm from Slivkins, 2014 with EXP3 as base learner
def contextualZooming(X, Y, n_P, n_Q, K):
    n = n_P + n_Q # total horizon
    
    if n == len(X[:,0]):
        start = 0
    else:
        start = len(X[:,0]) - n

    Xn = np.take(X, list(range(start,len(X[:,0]))), 0) # covariates
    Yn = Y[start:] # rewards

    pulls = np.zeros(n,dtype=int)# arm pulls
    initialLosses = np.zeros(K)
    balls = [[0.5,0.5,1,0,initialLosses,True]] # center, radius, n_B, losses, active status

    for t in range(0,n):
        x = Xn[t,:] # current covariate
        relevant = findRelevant(balls,x) # get relevant balls
        B = [] # active ball containing 
        if len(relevant) != 0:
            B = relevant[0] # pick any arbitrary relevant ball
        else: # make new ball if no relevant balls
            r = 1
            for ball in balls:
                if checkBall(ball,x) and ball[2] < r:
                    r = ball[2]
            r = r / 2
            B = [x[0],x[1],r,0,initialLosses,True]
            balls.append(B)

        # EXP3 base learner
        maxlength = max(2,math.floor(math.sqrt(K) / (B[2] ** 2) * max(1,np.log(1/B[2])))) # max duration of ball
        eta = math.sqrt(2 * np.log(K) / (maxlength * K))
        p = np.exp(- B[4] * eta) / sum(np.exp(- B[4] * eta))
        
        np.random.seed(t)
        pulls[t] =  np.random.choice(a=range(K),p=p)
        if B[3] >= maxlength: # set ball to inactive if max duration surpassed
            B[5] = False
        else: # update loss vector for this ball
            indicator = np.zeros(K)
            indicator[pulls[t]] = 1
            add = np.divide(indicator,p) * (1 - Yn[t,pulls[t]]) # importance weighting
            B[4] = B[4] + add
            B[3] = B[3] + 1 # iterate n_B
    return pulls

# check containment of vector in ball
def checkBall(vec,x):
    x1 = vec[0]
    x2 = vec[1]
    r = vec[2]
    if (x1 - x[0])**2 + (x2 - x[1])**2 <= r**2:
        return True
    else:
        return False

# get relevant active balls containing x
def findRelevant(balls, x):
    relevant = []
    for b in balls:
        if checkBall(b,x) and b[5]:
            relevant.append(b)
    return relevant

