import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def decomp(B,D):
    C = B * np.diag(D**2) * B.conj().T
    invC = B * np.diag(np.reciprocal(D)) * B.conj().T
    return C, invC

def frosenbrock(x):
    a = 1
    b = 100

    f = b * np.sum(x[0:-1])
    return ## TODO



def cmaes(N, mu, lam):
    xmean = np.random.standard_normal(size=(N,1))
    sigma = 0.3
    stopfitness = 1e-10
    stopeval = 1e3*N**2

    weights = np.log(mu+1/2)-np.log(np.arange(1,mu))
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
    C, invsqrtC = decomp(B,D)
    eigeneval = 0
    chiN = N**0.5 * (1 - 1/(4*N) + 1/(21 * N**2) )

    counteval = 0

    # while counteval < stopeval:

    
    

N = 20
lam = 4+np.floor(3*np.log(N))
mu = np.floor(lam/2)

xmin = cmaes(N, mu,lam)
