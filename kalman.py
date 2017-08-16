import numpy as np
import scipy.linalg as la

# Note that LKF, EKF and test_s all expect trace to be a np.array of shape (1000).

### LKF ###

# Prepares the state prop matrices.
def prep_mat(omegam=2*np.pi*19.645*10**3,gamma=2*np.pi*0.561*10**3,s=10**6):
    dT = 2.*(10**-3)/1000
    # State prop matrix (continuous)
    A = np.array([[0,1],[-omegam**2,-gamma]])
    # State prop matrix (discrete)
    T = la.expm(A*dT)

    # Covariance matrix
    Qa = np.array([[0,0],[0,s**2]])
    # Find covariance prop matrix by integrating exp(At).Qa.exp(A^Tt) from t to dT
    it = lambda t: la.expm(A*t).dot(Qa).dot(la.expm(np.transpose(A)*t))
    ts = np.linspace(0,dT,100)
    Q = sum([it(t)*(ts[1]-ts[0]) for t in ts])
    return (T,Q)

# 449: 19.645,0.561,1e6: 21.12

# Call prep_mat to get mat.
def LKF(trace,mat,stoppoint=449,R=0.001):
    T,Q = TandQ

    dT = 2.*(10**-3)/1000

    # Initial state
    X = np.array([[trace[0]],[(trace[1]-trace[0])/dT]])
    H = np.array([[1,0]])

    P = np.diag([np.abs(trace[0])/1000,np.abs(trace[1]-trace[0])/(1000*dT)])**2

    Xs = []
    Ps = []
    for i,d in enumerate(trace):
        # State prop
        X = T.dot(X)
        P = T.dot(P).dot(np.transpose(T)) + Q
        if i < stoppoint:
            E = d - H.dot(X)
        else:
            E = np.array([[0]])

        # Optimisation to abuse the fact that H=[1,0,0,0]
        K = P.dot(np.transpose(H))*1./(R+P[0,0])
        X = X + K.dot(E)
        P = (np.eye(len(P)) - K.dot(H)).dot(P)
        Xs.append(X)
        Ps.append(P)
    return np.array(Xs)[:,:,0],np.array(Ps)


### EKF ###

# Originally based on http://www.acasper.org/tag/kalman-filter/, but tweaked significantly after
# the discovery of some bugs.

#dX/dt=X'
def createT(Xs):
    X = Xs[:,0]
    return np.array([[X[1]],[-2*X[2]*X[3]*X[1]-X[3]**2*X[0]],[0],[0]])

#dX'/dX
def createJ(Xs):
    X = Xs[:,0]
    return np.array([[0,1,0,0],[-X[3]**2,-2*X[2]*X[3],-2*X[1]*X[3],-2*X[2]*X[1]-2*X[3]*X[0]],[0,0,0,0],[0,0,0,0]])

#dP/dt
def createDP(X,P,s):
    F = createJ(X)
    return F.dot(P) + P.dot(np.transpose(F)) + np.array([[0,0,0,0],[0,s**2,0,0],[0,0,0,0],[0,0,0,0]])

def EKF(trace,omegam=19.645*10**3,gammam=0.561*10**3,s=10**6,xunc=1000,vunc=1000,zunc=10,wunc=10,stoppoint=449,R=0.001):
    dT = 2.*(10**-3)/1000
    zeta = gammam/(2*omegam)
    omega0=2*np.pi*omegam
    P = np.diag([np.abs(trace[0])/xunc,np.abs(trace[1]-trace[0])/(vunc*dT),zeta/zunc,omega0/wunc])**2

    X = np.array([[trace[0]],[(trace[1]-trace[0])/dT],[zeta],[omega0]])
    # If you change this then see comment below about abusing the fact that this is so simple.
    H = np.array([[1,0,0,0]])

    ni = 5
    ddT = dT/ni

    Xs = []
    Ps = []
    for i,d in enumerate(trace):
        for p in range(ni):
            # State propagation of X and P. From Lewis, Xie, Popa "Optimal and robust estimation:
            # with an introduction to stochastic control theory", second edition, p274, using RK4
            # to solve the IVP.
            dX = createT(X)

            k1 = dX*ddT
            dX2 = createT(X+0.5*k1)
            k2 = dX2*ddT
            dX3 = createT(X+0.5*k2)
            k3 = dX3*ddT
            dX4 = createT(X + k3)
            k4 = dX4*ddT
            Delta = (k1+2*k2+2*k3+k4)/6.

            dP = createDP(X,P,s)
            l1 = dP*ddT
            dP2 = createDP(X+0.5*k1,P+0.5*l1,s)
            l2 = dP2*ddT
            dP3 = createDP(X+0.5*k2,P+0.5*l2,s)
            l3 = dP3*ddT
            dP4 = createDP(X + k3,P+l3,s)
            l4 = dP4*ddT

            newP = P + (l1+2*l2+2*l3+l4)/6.

            X = X + Delta
            P = newP
        if i < stoppoint:
            E = d - H.dot(X)
            # Optimisation to abuse the fact that H=[1,0,0,0]
            K = P.dot(np.transpose(H))*1./(R+P[0,0])
            X = X + K.dot(E)
            P = (np.eye(4) - K.dot(H)).dot(P)
        else:
            E = np.array([[0]])

        Xs.append(X)
        Ps.append(P)
    return np.array(Xs)[:,:,0],np.array(Ps)


### Testing ###

# Generates a sample trace using the noise standard deviation s. Uses trace
# to get the initial position and velocity (and for plotting).
def test_s(trace,s,omegam=2*np.pi*20*10**3,gamma=2*np.pi*0.85*10**3):
    dT = 2*10**-3/1000
    # State prop matrix (continuous)
    A = np.array([[0,1],[-omegam**2,-gamma]])
    # State prop matrix (discrete)
    T = la.expm(A*dT)
    # Initial state
    M0 = np.array([[trace[0]],[(trace[1]-trace[0])/dT]])

    # Generates the covariance prop matrix.
    def get_Q(A,s):
        Qa = np.array([[0,0],[0,s**2]])
        it = lambda t: la.expm(A*t).dot(Qa).dot(la.expm(np.transpose(A)*t))
        ts = np.linspace(0,10**-3/1000,100)
        Q = sum([it(t)*(ts[1]-ts[0]) for t in ts])
        return Q

    M = M0
    Ms = [M]
    Q = get_Q(A,s)
    for i in range(1000):
        R = np.random.multivariate_normal([0,0],Q).reshape(2,1)
        M = T.dot(M)+R
        Ms.append(M)
    return np.array(Ms)[:,:,0]
