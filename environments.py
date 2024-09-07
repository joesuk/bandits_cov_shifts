# functions related to contextual bandit environments
import numpy as np
import math
import scipy
from scipy.spatial import distance
from scipy import integrate
from scipy.stats import multivariate_normal

# construct Lipschitz environment
def makeLipschitz(n_P, n_Q, K, gamma, centers, radii, rads, heights, signs, num_bumps, max_height):

    Q_X = np.random.uniform(0,1, size=(n_Q,2)) # target covariate distribution
    P_X = P_rejection_sampler(gamma, n_P) # source covariate distribution
    X = np.concatenate((P_X,Q_X),axis=0) 
    n = n_P + n_Q

    # first define reward function    
    f = reward_function(X, n, K, centers, radii, rads, heights, signs, num_bumps, max_height)

    # get best arms
    oracle = np.zeros(n,dtype=int)
    for i in range(n):
        argmaxs = np.argwhere(f[i,:] == np.amax(f[i,:])).reshape(-1)
        oracle[i] = np.random.choice(argmaxs)

    # generate rewards and rescale.
    Y = f + np.random.normal(0,0.05,size=f.shape)
    diff = Y.max() - Y.min()
    Y = (Y - Y.min()) / diff

    return X, Y, f, oracle

# construct Lipschitz reward function for K arms
def reward_function(X, n, K, centers, radii, rads, heights, signs, num_bumps, max_height):
    f = np.zeros((n, K))
    vals = np.zeros(n)
    for k in range(K):
        for j in range(0,n):
            inBump = False
            for i in range(0,num_bumps):
                if signs[i,k] == np.sign(heights[k]) and (np.linalg.norm(centers[i] - X[j,:]) <= radii[i,k]):
                    vals[j] = signs[i,k] * max(0,  1 - (np.linalg.norm(X[j,:] - centers[i]) / rads[i]))
                    inBump = True
                    break
                elif np.linalg.norm(centers[i] - X[j,:]) <= rads[i] and signs[i,k] != np.sign(heights[k]):
                    vals[j] = signs[i,k] * max(0,  1 - (np.linalg.norm(X[j,:] - centers[i]) / rads[i])) 
                    inBump = True
                    break
            if inBump:
                f[j,k] = vals[j]
            else:
                f[j,k] = heights[k]
                
    # constrain reward function to [0,1]
    if f.min() < 0 or f.max() > 1:
        diff = f.max() - f.min()
        f = (f - f.min()) / diff
        
    return f


# construct 2D bumps with random centers and radii
def construct_bumps(num_bumps, K, max_height,model):
    if model=="uniform":
        centers = np.random.uniform(0, 1, size= (num_bumps,2))
    elif model=="gaussian":
        centers = np.random.multivariate_normal([0.5, 0.5], [[0.5, 0], [0, 0.5]],num_bumps)  
        
    radii = np.full((num_bumps,K), math.sqrt(2))
    rads = np.full(num_bumps, math.sqrt(2))
    for i in range(num_bumps):
        if i == 0:
            others = np.delete(centers,i,0)
            rads[i] = np.min(distance.cdist([centers[i]],others)) / 2
        else:
            for j in range(0,num_bumps):
                if j < i:
                    rads[i] = min(rads[i], np.linalg.norm(centers[i] - centers[j]) - rads[j])
                elif j != i:
                    rads[i] = min(rads[i],np.linalg.norm(centers[i] - centers[j]) / 2)
    heights = np.random.uniform(-max_height, max_height, K)
    signs = np.random.choice([-1,1],size=(num_bumps,K))

    # modify radii
    for i in range(num_bumps):
        for k in range(K):
            if signs[i,k] == np.sign(heights[k]):
                radii[i,k] = rads[i] * (1 - heights[k] / signs[i,k])
            else:
                radii[i,k] = rads[i]
    return centers, radii, heights, signs, rads


# rejection sampling of covariate with density |x|_2^gamma
def P_rejection_sampler(gamma, num_samples):
    def f(x,y): return ((0.5 - x)**2 + (0.5 - y)**2)**(gamma / 2)
    max_f = ((0.5 - 0)**2 + (0.5 - 0)**2)**(gamma / 2)
    sample = np.zeros((num_samples,2))
    counter = 0
    while counter < num_samples:
        x = np.random.rand(1)[0]
        y = np.random.rand(1)[0]
        z = np.random.rand(1)[0]*max_f
        if z <= f(x,y):
            sample[counter,] = (x,y)
            counter += 1
    return sample


################################
# regret functions
################################

# compute the regret over n_Q rounds
def computeRegret(pulls, n_P, n_Q, oracle, f, X, Y):
    n = n_P + n_Q
    regret = np.zeros(n_Q)
    total = len(X[:,0])
    for i in range(0,n_Q):
        ind = i + total - n + n_P
        regret[i] =  Y[ind,oracle[ind]] - Y[ind,pulls[i+n_P]] 
    regret = np.cumsum(regret)
    return regret

# compute regret of a random oracle policy and worst arm choice policy
def computeRegretRandom(n_Q, n_P, oracle, f, Y):
    n = n_P + n_Q
    regret_random = np.zeros(n_Q)
    regret_worst = np.zeros(n_Q)
    total = len(X[:,0])
    for i in range(0,n_Q):
        ind = i + total - n + n_P
        regret_random[i] =  Y[ind,oracle[ind]] - Y[ind, np.random.choice(range(K))]
        regret_worst[i] = Y[ind,oracle[ind]] - min(Y[ind,list(range(K))])
    regret_random = np.cumsum(regret_random)
    regret_worst = np.cumsum(regret_worst)
    return regret_random, regret_worst

