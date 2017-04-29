
from abc import ABCMeta, abstractmethod
import numpy as np
import time


class dynamicSystem:

    __metaclass__ = ABCMeta

    def __init__(self, order, nmbr_controls):

        self._order = order
        self._nmbrControls = nmbr_controls

        self._controller = constantControl(0)

        return

    def getNmbrStates(self):
        return self._order

    def getNmbrControls(self):
        return self._nmbrControls

    def setController(self, controller):
        self._controller = controller
        return

    @abstractmethod
    def f(self, x_k, u_k ,k): pass

    @abstractmethod
    def f_x(self, x_k, u_k, k): pass

    @abstractmethod
    def f_u(self, x_k, u_k, k): pass

    @abstractmethod
    def f_xx(self, x_k, u_k, k): pass

    @abstractmethod
    def f_uu(self, x_k, u_k, k): pass

    @abstractmethod
    def f_xu(self, x_k, u_k, k): pass

    def propagate(self, x0, N):

        x = np.zeros((N+1, self._order))
        u = np.zeros((N, self._nmbrControls))
        x[0,:] = x0
        for k in range(0, N):
            u[k, :] = self._controller.getControl(x[k], k)
            x[k+1, :] = self.f(x[k], u[k], k)
        return (x, u)

class costSystem:

    __metaclass__ = ABCMeta

    def __init__(self, order, nmbr_controls):
        self._order = order
        self._nmbrControls = nmbr_controls

        return

    def computeCost(self, x_history, u_history, N):
        J = self.phi(x_history[N])
        for k in range(0, N):
            J += self.L_k(x_history[k], u_history[k], k)
        return J

    @abstractmethod
    def L_k(self, x_k, u_k, k): pass

    @abstractmethod
    def L_k_x(self, x_k, u_k, k): pass

    @abstractmethod
    def L_k_u(self, x_k, u_k, k): pass

    @abstractmethod
    def L_k_xx(self, x_k, u_k, k): pass

    @abstractmethod
    def L_k_xu(self, x_k, u_k, k): pass

    @abstractmethod
    def L_k_uu(self, x_k, u_k, k): pass

    @abstractmethod
    def phi(self, x_N): pass

    @abstractmethod
    def phi_x(self, x_N): pass

    @abstractmethod
    def phi_xx(self, x_N): pass




class controller:

    __metaclass__ = ABCMeta

    def __init__(self):

        return

    @abstractmethod
    def getControl(self, x, k): pass

class constantControl(controller):

    def __init__(self, const):
        super(constantControl, self).__init__()
        self._const = const
        return

    def getControl(self, x, k):
        return self._const



class vectorControl(controller):

    def __init__(self, vect):
        super(vectorControl, self).__init__()
        self._length = vect.shape[0]
        self._vect = vect
        return

    def getControl(self, x, k):
        return self._vect[k % self._length, :]

    def getVector(self):
        return self._vect


class linearFeedbackControl(controller):

    def __init__(self, alpha, beta, x_ref, u_ref):
        super(linearFeedbackControl, self).__init__()
        self._alpha = alpha
        self._beta = beta
        self._x_ref = x_ref
        self._u_ref = u_ref
        self._length = u_ref.shape[0]
        return

    def getControl(self, x, k):
        u = self._u_ref[k % self._length] + self._alpha[k] + self._beta[k].dot(x - self._x_ref[k % self._length])
        return u

class linearNewtonControl(controller):

    def __init__(self, alpha, beta, x_ref, u_ref, f_x, f_u, x_0):
        super(linearNewtonControl, self).__init__()
        self._alpha = alpha
        self._beta = beta
        self._x_ref = x_ref
        self._u_ref = u_ref
        self._length = u_ref.shape[0]
        self._f_x = f_x
        self._f_u = f_u
        self._x_last = x_0
        return

    def getControl(self, x, k):
        u = self._u_ref[k % self._length] + self._alpha[k] + self._beta[k].dot(self._x_last - self._x_ref[k % self._length])
        self._x_last = self._x_ref[(k + 1)] + self._f_x[k].dot(self._x_last - self._x_ref[k]) +\
                       self._f_u[k].dot(u - self._u_ref[k])
        return u



class testSystem(dynamicSystem):

    def __init__(self, A, B, C, gamma, n, m):
        super(testSystem, self).__init__(n, m)
        self._A = A
        self._B = B
        self._C = C
        self._gamma = gamma
        return

    def f(self, x_k, u_k ,k):
        x_next = self._A.dot(x_k) + self._B.dot(u_k) + np.inner(x_k, self._C.dot(u_k)) * self._gamma
        return x_next

    def f_x(self, x_k, u_k, k):
        # f_x = self._A + np.outer(self._gamma, self._C.dot(u_k))
        f_x = self._A + np.outer(self._gamma, u_k).dot(self._C.T)
        return f_x

    def f_u(self, x_k, u_k, k):
        f_u = self._B + np.outer(self._gamma, x_k).dot(self._C)
        return f_u

    def f_xx(self, x_k, u_k, k):
        return np.zeros([self._order, self._order, self._order]) # Third order tensor

    def f_uu(self, x_k, u_k, k):
        return np.zeros([self._order, self._nmbrControls, self._nmbrControls]) # Third order tensor

    def f_xu(self, x_k, u_k, k):
        f_xu = np.zeros([self._order, self._order, self._nmbrControls]) # Third order tensor
        for i in range(0, self._order):
            f_xu[i,:,:] = self._gamma[i] * self._C
        return f_xu

class testCost(costSystem):

    def __init__(self, n, m):
        super(testCost, self).__init__(n, m)

        return

    def L_k(self, x_k, u_k, k):
        L_k = np.sum((x_k + 0.25)**4) + np.sum((u_k + 0.5)**4)
        return L_k

    def phi(self, x_N):
        phi = np.sum((x_N + 0.25)**4)
        return phi

    def L_k_x(self, x_k, u_k, k):
        L_k_x = 4*(x_k + 0.25)**3
        return L_k_x

    def L_k_u(self, x_k, u_k, k):
        L_k_u = 4*(u_k + 0.5)**3
        return L_k_u

    def L_k_xx(self, x_k, u_k, k):
        L_k_xx = np.diag(12*(x_k + 0.25)**2)
        return L_k_xx

    def L_k_xu(self, x_k, u_k, k):
        return np.zeros([self._order, self._nmbrControls])

    def L_k_uu(self, x_k, u_k, k):
        L_k_uu = np.diag(12*(u_k + 0.5)**2)
        return L_k_uu

    def phi_x(self, x_N):
        phi_x = 4*(x_N + 0.25)**3
        return phi_x

    def phi_xx(self, x_N):
        phi_xx = np.diag(12*(x_N + 0.25)**2)
        return phi_xx



def ddp(dynSys, costSys, x_0, u_guess_vect, N, cost_thres):
    """
    Differential Dynamic Programming (DDP) Method.
    The algorithm is similar to
    - Yakowitz, Rutherford, "Computational Aspects of Discrete-Time optimal Control" (1984)
    - Liao, Shoemaker, "Advantages of DDP over Newton's Method for Discrete-Time Optimal Control Problem" (1992).
    The notation is similar to the one used in
    - Ozaki, Campagnola, Yam, Funase, "DDP Approach for Robust-Optimal Low-Thrust Trajectory Design Considering Uncertainty" (2015).
    The original work about DDP is
    - Jacobson, Mayne, "Differential Dynamic Programming" (1970).
    :param dynSys:
    :param costSys:
    :param x_0:
    :param u_guess_vect:
    :param N:
    :param cost_thres:
    :return:
    """
    start_time = time.time()

    n = dynSys.getNmbrStates()
    m = dynSys.getNmbrControls()

    alpha = np.zeros((N, m))
    beta = np.zeros((N,m,n))

    controller = vectorControl(u_guess_vect)

    # Propagate
    dynSys.setController(controller)
    (x_ref, u_ref) = dynSys.propagate(x_0, N)

    # Compute total cost
    J = costSys.computeCost(x_ref, u_ref, N)

    it = 1
    while True:
        print "Iteration nbr", it, ". Cost: ", J
        it += 1

        V_xx = costSys.phi_xx(x_ref[N])
        V_x = costSys.phi_x(x_ref[N])

        cost_reduction = 0

        # Backward sweep
        for k in range(N-1, -1, -1):

            f_x = dynSys.f_x(x_ref[k], u_ref[k], k)
            f_u = dynSys.f_u(x_ref[k], u_ref[k], k)

            f_xx = dynSys.f_xx(x_ref[k], u_ref[k], k)
            f_xu = dynSys.f_xu(x_ref[k], u_ref[k], k)
            f_uu = dynSys.f_uu(x_ref[k], u_ref[k], k)

            L_k_x = costSys.L_k_x(x_ref[k], u_ref[k], k)
            L_k_u = costSys.L_k_u(x_ref[k], u_ref[k], k)

            L_k_xx = costSys.L_k_xx(x_ref[k], u_ref[k], k)
            L_k_xu = costSys.L_k_xu(x_ref[k], u_ref[k], k)
            L_k_uu = costSys.L_k_uu(x_ref[k], u_ref[k], k)

            q_x = f_x.T.dot(V_x) + L_k_x
            q_u = f_u.T.dot(V_x) + L_k_u

            Q_xx = f_x.T.dot(V_xx).dot(f_x) + L_k_xx
            Q_xu = f_x.T.dot(V_xx).dot(f_u) + L_k_xu
            Q_uu = f_u.T.dot(V_xx).dot(f_u) + L_k_uu
            for i in range(0, n):
                Q_xx += V_x[i] * f_xx[i]
                Q_xu += V_x[i] * f_xu[i]
                Q_uu += V_x[i] * f_uu[i]


            Q_uu_inv = np.linalg.inv(Q_uu)

            # With constraints, change alpha and beta
            alpha[k] = -Q_uu_inv.dot(q_u)
            beta[k] = -Q_uu_inv.dot(Q_xu.T)

            # V_xx = Q_xx + Q_xu.dot(beta[k]) + beta[k].T.dot(Q_xu.T) + beta[k].T.dot(Q_uu).dot(beta[k])
            # V_x = q_x + beta[k].T.dot(q_u) + Q_xu.dot(alpha[k]) + beta[k].T.dot(Q_uu).dot(alpha[k])

            V_xx = Q_xx - Q_xu.dot(Q_uu_inv.dot(Q_xu.T))
            V_x = -Q_xu.dot(Q_uu_inv.dot(q_u)) + q_x

            # cost_reduction += np.inner(q_u, alpha[k]) + 0.5 * np.inner(alpha[k], Q_uu.dot(alpha[k]))
            cost_reduction += np.inner(q_u, Q_uu_inv.dot(q_u))
        # End for

        #print "Cost reduction", cost_reduction

        # Update Control
        epsilon = 1.0
        while True:

            # Set new control
            dynSys.setController(linearFeedbackControl(epsilon*alpha, beta, x_ref, u_ref))

            # Propagate
            (x_eps, u_eps) = dynSys.propagate(x_0, N)

            # Compute cost
            J_eps = costSys.computeCost(x_eps, u_eps, N)

            if J_eps < J and J_eps - J < epsilon * cost_reduction/2:
                x_ref = x_eps
                u_ref = u_eps
                J = J_eps
                break
            else:
                epsilon = epsilon/2
        # End While

        if np.abs(cost_reduction) < cost_thres:
            break

    # End While

    elapsed_time = time.time() - start_time

    print "Time Elapsed (DDP)", elapsed_time

    return (x_ref, u_ref, J)



def stagewiseNewton(dynSys, costSys, x_0, u_guess_vect, N, cost_thres):
    """
    Stagewise Newton is equivalent to Newton's method, but with reduced dimension.
    The method was invented by
    - Pantoja, "Differential Dynamic Programming and Newton Method" (1988).
    The implementation is somewhat similar to the one given in
    - Liao, Shoemaker, "Advantages of DDP over Newton's Method for Discrete-Time Optimal Control Problem" (1992).
    Useful comparison between DDP and Newton's method
    - Murray, Yakowitz, "Differential Dynamic Programming and Newton's Method for Discrete Optimal Control Problems" (1984).
    :param dynSys:
    :param costSys:
    :param x_0:
    :param u_guess_vect:
    :param N:
    :param cost_thres:
    :return:
    """
    start_time = time.time()

    n = dynSys.getNmbrStates()
    m = dynSys.getNmbrControls()

    alpha = np.zeros((N, m))
    beta = np.zeros((N,m,n))

    controller = vectorControl(u_guess_vect)

    # Propagate
    dynSys.setController(controller)
    (x_ref, u_ref) = dynSys.propagate(x_0, N)

    # Compute total cost
    J = costSys.computeCost(x_ref, u_ref, N)

    it = 1
    while True:
        print "Iteration nbr", it, ". Cost: ", J
        it += 1

        P = costSys.phi_xx(x_ref[N])
        Q = costSys.phi_x(x_ref[N])
        G = costSys.phi_x(x_ref[N])

        cost_reduction = 0

        f_x = np.zeros([N, n, n])
        f_u = np.zeros([N, n, m])

        # Backward sweep: This is slightly different from DDP
        for k in range(N-1, -1, -1):

            f_x[k,:,:] = dynSys.f_x(x_ref[k], u_ref[k], k) # I need to store these two
            f_u[k,:,:] = dynSys.f_u(x_ref[k], u_ref[k], k)

            f_xx = dynSys.f_xx(x_ref[k], u_ref[k], k)
            f_xu = dynSys.f_xu(x_ref[k], u_ref[k], k)
            f_uu = dynSys.f_uu(x_ref[k], u_ref[k], k)

            L_k_x = costSys.L_k_x(x_ref[k], u_ref[k], k)
            L_k_u = costSys.L_k_u(x_ref[k], u_ref[k], k)

            L_k_xx = costSys.L_k_xx(x_ref[k], u_ref[k], k)
            L_k_xu = costSys.L_k_xu(x_ref[k], u_ref[k], k)
            L_k_uu = costSys.L_k_uu(x_ref[k], u_ref[k], k)

            A = f_x[k].T.dot(P).dot(f_x[k]) + L_k_xx
            B_T = f_x[k].T.dot(P).dot(f_u[k]) + L_k_xu
            C = f_u[k].T.dot(P).dot(f_u[k]) + L_k_uu
            for i in range(0, n):
                A += G[i] * f_xx[i]
                B_T += G[i] * f_xu[i]
                C += G[i] * f_uu[i]

            E = f_x[k].T.dot(Q) + L_k_x
            D = f_u[k].T.dot(Q) + L_k_u

            C_inv = np.linalg.inv(C)

            # With constraints, change alpha and beta
            alpha[k] = -C_inv.dot(D)
            beta[k] = -C_inv.dot(B_T.T)

            # P = A + B_T.dot(beta[k]) + beta[k].T.dot(B_T.T) + beta[k].T.dot(C).dot(beta[k])
            # Q = E + beta[k].T.dot(D) + B_T.dot(alpha[k]) + beta[k].T.dot(C).dot(alpha[k])
            P = A - B_T.dot(C_inv.dot(B_T.T))
            Q = -B_T.dot(C_inv.dot(D)) + E
            G = f_x[k].T.dot(G)

            # cost_reduction += np.inner(D, alpha[k]) + 0.5 * np.inner(alpha[k], C.dot(alpha[k]))
            cost_reduction += np.inner(D, C_inv.dot(D))
        # End for

        #print "Cost reduction", cost_reduction

        # Update Control: This is different from DDP!
        epsilon = 1.0
        while True:

            # Set new control
            dynSys.setController(linearNewtonControl(epsilon*alpha, beta, x_ref, u_ref, f_x, f_u, x_0))

            # Propagate
            (x_eps, u_eps) = dynSys.propagate(x_0, N)

            # Compute cost
            J_eps = costSys.computeCost(x_eps, u_eps, N)

            if J_eps < J and J_eps - J < epsilon * cost_reduction/2:
                x_ref = x_eps
                u_ref = u_eps
                J = J_eps
                break
            else:
                epsilon = epsilon/2
        # End While

        if np.abs(cost_reduction) < cost_thres:
            break

    # End While

    elapsed_time = time.time() - start_time

    print "Time Elapsed (Newton)", elapsed_time

    return (x_ref, u_ref, J)





