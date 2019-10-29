import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import time


def frosenbrock(x):
    a = 1
    b = 100
    f = b * np.sum((x[0:-2]**2 - x[1:-1])**2) + a*np.sum((x[0:-2]-1)**2)
    return f



def cmaes(N, mu, lam):
    start = time.time()
    xmean = np.random.standard_normal(size=(N,1))
    sigma = 0.3
    stopfitness = 1e-10
    stopeval = 1e3*N**2

    weights = np.log(mu+1/2)-np.log(np.r_[1:mu+1])
    weights = weights / np.sum(weights)
    mueff = np.sum(weights)**2 / np.sum(weights**2)

    cc = (4 + mueff / N) / (N+4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((N + 2)**2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(N + 1))-1) + cs

    pc = np.zeros(shape=(N,1))
    ps = np.zeros(shape=(N,1))
    B = np.eye(N)
    D = np.ones(shape=(N,1))
    C = B * np.diag(D**2) * B.T
    invsqrtC = B * np.diag(np.reciprocal(D)) * B.T
    eigeneval = 0
    chiN = N**0.5 * (1 - 1/(4*N) + 1/(21 * N**2) )

    counteval = 0
    arx = np.zeros(shape=(N,lam))
    arfitness = np.zeros(shape=(lam,1))
    while counteval < stopeval:

        for k in range(0,lam):
            arx[:,[k]] = xmean + sigma * B.dot(np.vstack(D)*np.random.standard_normal(size=(N,1)))
            arfitness[k] = frosenbrock(arx[:,k])
            counteval += 1
        
        arindex = np.argsort(arfitness[:,0])
        arfitness=arfitness[arindex].copy()
        xold = xmean.copy()
        xmean = np.vstack(arx[:,arindex[0:mu]].squeeze().dot(weights))

        ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC.dot(xmean-xold) / sigma
        hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*counteval/lam))
        hsig = hsig/chiN < (1.4 + 2/(N+1))
        pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (xmean-xold)/sigma
        

        artmp = (1/sigma) * (arx[:,arindex[0:mu]]-np.tile(xold,(1,mu)))
        C = (1-c1-cmu)*C
        C += c1*(pc.dot(pc.T) + (1-hsig) * cc * (2-cc) * C)
        C += cmu * artmp.dot(np.diag(weights)).dot(artmp.T)
        
        sigma = sigma * np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1))
        
        if counteval - eigeneval > lam/(c1+cmu) / N / 10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C,1).T
            D, B = np.linalg.eig(C)
            D = np.sqrt(D)
            invsqrtC = B.dot(np.diag(np.reciprocal(D))).dot(B.T)

        #print(arfitness[0])
        if arfitness[0] <= stopfitness or np.max(D) > 1e7 * np.min(D):
            break
    
    end = time.time()
    print("{0}, {1:.2f}".format(counteval, end - start))
    return arx[:,arindex[0]]



for N in [6,7]:
    lam = int(4+np.floor(3*np.log(N)))
    mu = int(np.floor(lam/2))
    print("N={}, lambda={}, mu={}".format(N,lam,mu))
    for i in range(0,20):
        xmin = cmaes(N, mu,lam)
        #print(xmin)
