from numpy import ones, diag, array, zeros
from numpy.linalg import eig, norm

def lag_grad_p(P, Q, Y, rho, w, z, pi):
    n = P.shape[0]
    one = ones(n).reshape((n, 1))
    v2 = sorted(zip(*eig(P.T.dot(P))), key=lambda x: x[0], reverse=True)[1][1].reshape((n, 1))
    first_part = v2.dot(v2.T) - Y
    second_part = rho*(P - Q)
    third_part = z.dot(one.T)
    forth_part = pi.dot(w.T)
    fifth_part = rho*(P - one.T.dot(one))
    last_part = (P - one.dot(one.T)).dot(diag(pi.squeeze()))
    return first_part + second_part - third_part - forth_part + fifth_part + last_part

def lag_grad_q(P, Q, Y, rho, pi, s, c, q, u):
    first_part = c*q.dot(u.T)
    second_part = 2*Q*norm(q)**2*norm(pi)**2
    third_part = 2*s*q.dot(u.T)
    fourth_part = Y
    last_part = rho*(Q - P)
    return first_part + second_part - third_part + fourth_part + last_part

def solve(c, u, s, pi, q):
    n = u.shape[0]
    P = diag(ones(n))
    Q = diag(ones(n))*3
    Y = diag(ones(n))
    rho = 1
    gamma = 1.25
    w = ones(n).reshape((n, 1))
    z = ones(n).reshape((n, 1))
    one = ones(n).reshape((n, 1))
    eps = 1e-4
    step = lambda j: 1 / (1 + j)
    inner_steps_max = 500
    outer_steps_max = 500

    P_Q_difference = []
    P_grad_norm = []

    P_prev = 30*P
    i = 0
    while norm(P - Q) > eps and norm(P - P_prev) > eps and i < outer_steps_max:
        j = 0
        P_prev = P
        lag_grad_loc = lag_grad_p(P, Q, Y, rho, w, z, pi)
        while norm(lag_grad_loc)**2 > eps and j < inner_steps_max:
            P = P - step(j)*lag_grad_loc
            lag_grad_loc = lag_grad_p(P, Q, Y, rho, w, z, pi)
            j = j + 1
            lag_grad_loc_norm = norm(lag_grad_loc)

        P.clip(0, out=P)

        P_grad_norm.append(norm(P))

        j = 0
        lag_grad_loc = lag_grad_q(P, Q, Y, rho, pi, s, c, q, u)
        while norm(lag_grad_loc) ** 2 > eps and j < inner_steps_max:
            Q = Q - step(j) * lag_grad_loc
            lag_grad_loc = lag_grad_q(P, Q, Y, rho, pi, s, c, q, u)
            lag_grad_loc_norm = norm(lag_grad_loc)
            j = j + 1

        Q.clip(0, out=Q)


        P_Q_difference.append(norm(P - Q))

        Y = Y - rho*(P - Q)
        z = z  - rho*(P.dot(one) - one)
        w = w - rho*(P.T.dot(pi) - pi)
        rho = rho*gamma
        i = i + 1

    return P, {"P_grad_norm": P_grad_norm, "P_Q_difference": P_Q_difference}, j


if __name__ == "__main__":
    n = 3
    c = 1
    u = zeros(n).reshape((n, 1))
    u[0][0] = 1
    s = 1
    pi = (ones(n)/n).reshape((n, 1))
    q = ones(n).reshape((n, 1))
    P, log, converged = solve(c, u, s, pi, q)
    print(P, converged)

