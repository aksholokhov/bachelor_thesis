from numpy import ones, diag, array, zeros, tensordot, sqrt, eye, log
from numpy.random import randn
from numpy.linalg import eig, norm, inv, pinv, tensorinv

def lag_grad_p(P, Q, Y, rho_y, rho_z, rho_w, w, z, pi):
    n = P.shape[0]
    one = ones(n).reshape((n, 1))
    v2 = abs(sorted(zip(*eig(P.T.dot(P))), key=lambda x: x[0], reverse=True)[1][1].reshape((n, 1)))
    first_part = 2*P.dot(v2.dot(v2.T)) - Y
    second_part = rho_y*(P - Q)
    third_part = z.dot(one.T)
    forth_part = pi.dot(w.T)
    fifth_part = rho_z*(P.dot(one.dot(one.T)) - one.dot(one.T))
    last_part = rho_w*(pi.dot(pi.T).dot(P) - pi.dot(pi.T))
    return first_part + second_part - third_part - forth_part + fifth_part + last_part

def lag_grad_q(P, Q, Y, rho_y, pi, s, c, q, u):
    first_part = c*q.dot(u.T)
    second_part = 2*Q*norm(q)**2*norm(u)**2
    third_part = 2*s*q.dot(u.T)
    fourth_part = Y
    last_part = rho_y*(Q - P)
    return first_part + second_part - third_part + fourth_part + last_part

def solve(c, u, s, pi, q):
    n = u.shape[0]
    P = array([abs(x)/sum(abs(x)) for x in randn(n, n)])
    Q = array([abs(x)/sum(abs(x)) for x in randn(n, n)])
    Y = diag(zeros(n))
    rho_y = 1
    rho_z = 1
    rho_w = 1
    gamma = 3
    eta = 0.25
    w = zeros(n).reshape((n, 1))
    z = zeros(n).reshape((n, 1))
    one = ones(n).reshape((n, 1))
    eps_inner = 1e-3*n
    eps_outer = 2e-2*n # too big => too poor result
    step = lambda j, l: l / (2+ j)
    inner_steps_max = 1000*(n/5)
    outer_steps_max = 500

    P_Q_difference = []
    P_grad_norm = []
    Q_grad_norm = []
    pi_constr_violation = []

    i = 0

    constraint_violation_y = norm(P - Q) ** 2
    constraint_violation_z = norm(P.dot(one) - one) ** 2
    constraint_violation_w = norm(P.T.dot(pi) - pi) ** 2

    p_iters = []
    q_iters = []

    y_upd_num = 0
    z_upd_num = 0
    w_upd_num = 0

    y_upd = []
    w_upd = []
    z_upd = []

    while (norm(P - Q) > eps_outer or norm(P.T.dot(pi) - pi) > eps_outer) and i < outer_steps_max:

        j = 0
        P_prev = P
        lag_grad_loc = lag_grad_p(P, Q, Y, rho_y, rho_z, rho_w, w, z, pi)


        # H_inv = inv(H + randn(n, n, n, n)*0.01)
        while norm(lag_grad_loc)**2 > eps_inner and j < inner_steps_max:
            # H = zeros((n, n, n, n))
            # pipi_t = pi.dot(pi.T)
            # for k in range(n):
            #     H[k, k, :, :] += rho_y * ones((n, n))
            #     H[k, :, k, :] += rho_z * ones((n, n))
            #     H[k, :, :, k] += rho_w * pipi_t
            #
            # l2, v2 = sorted(zip(*eig(P.T.dot(P))), key=lambda x: x[0], reverse=True)[1]
            # pinv_comp = P.T.dot(pinv(l2 * eye(n) - P.T.dot(P)))
            #
            # for k in range(n):
            #     for l in range(n):
            #         ans = 4 * v2.dot(P[:, k]) * pinv_comp
            #         ans[k, :] += v2
            #         H[k, l, :, :] += v2[l] * ans
            #
            # P = P - tensordot(tensorinv(H), lag_grad_loc, axes=2)
            P = P - step(j,1/(i+1))*lag_grad_loc
            P.clip(min=0, out=P)
            lag_grad_loc = lag_grad_p(P, Q, Y, rho_y, rho_z, rho_w, w, z, pi)
            lag_grad_loc_norm = norm(lag_grad_loc)  #debug
            j = j + 1

        p_iters.append(j)

        P_grad_norm.append(lag_grad_loc_norm)

        j = 0
        lag_grad_loc = lag_grad_q(P, Q, Y, rho_y, pi, s, c, q, u)
        while norm(lag_grad_loc) ** 2 > eps_inner and j < inner_steps_max:
            Q = Q - step(j, 1/(i+1)) * lag_grad_loc
            Q.clip(0, out=Q)
            lag_grad_loc = lag_grad_q(P, Q, Y, rho_y, pi, s, c, q, u)
            lag_grad_loc_norm = norm(lag_grad_loc)  #debug
            j = j + 1

        q_iters.append(j)
        Q_grad_norm.append(lag_grad_loc_norm)
        P_Q_difference.append(norm(P - Q))

        prev_constraint_violation_z = constraint_violation_z
        constraint_violation_z = norm(P.dot(one) - one)**2
        if constraint_violation_z < eta * prev_constraint_violation_z:
            z = z - rho_z * (P.dot(one) - one)
            z_upd_num += 1
        else:
            rho_z = rho_z * gamma

        prev_constraint_violation_w = constraint_violation_w
        constraint_violation_w = norm(P.T.dot(pi) - pi)**2
        if constraint_violation_w < eta * prev_constraint_violation_w:
            w = w - rho_w * (P.T.dot(pi) - pi)
            w_upd_num += 1
        else:
            rho_w = rho_w * gamma

        prev_constraint_violation_y = constraint_violation_y
        constraint_violation_y = norm(P - Q)**2
        if constraint_violation_y < eta*prev_constraint_violation_y:
            Y = Y - rho_y * (P - Q)
            y_upd_num += 1
        else:
            rho_y = rho_y * gamma

        z_upd.append(z_upd_num)
        w_upd.append(w_upd_num)
        y_upd.append(y_upd_num)

        pi_constr_violation.append(norm(P.T.dot(pi) - pi))

        i = i + 1

    return P, {"P_grad_norm": P_grad_norm,
               "Q_grad_norm": Q_grad_norm,
               "P_Q_difference": P_Q_difference,
               "pi_constr" : pi_constr_violation,
               "P_iters": p_iters,
               "Q_iters": q_iters,
               "y_upd": y_upd,
               "w_upd": w_upd,
               "z_upd": z_upd}, i


if __name__ == "__main__":
    from time import time, sleep
    from Ensemble import AbstractEnsemble
    from scipy.optimize import minimize
    from numpy import array, arange, copy
    from numpy.random import randint
    from scipy.optimize import minimize

    K = 100
    N = 3

    ensemble = AbstractEnsemble(K, default_policy=ones((N, N)) / N, tick=0.01, random_seed=42)
    states = []
    consumption = []
    ss = []
    sol = []
    us = []
    pn = []
    sn = []
    s = 0

    q = (arange(N) + 1).reshape((N, 1))


    def c(t):
        return 1


    def find_pi_new(pi_old, q, s):
        n = len(q)
        f = lambda x: norm(x - pi_old) ** 2
        cons = [{"type": "eq", "fun": lambda x: x.dot(q) - s},
                {"type": "eq", "fun": lambda x: x.dot(ones(n)) - 1}]
        bnds = [(0, None) for _ in range(n)]
        res = minimize(f, pi_old, method='SLSQP', bounds=bnds, constraints=tuple(cons))
        return res.x.reshape((n, 1))


    try:
        ensemble.run()

        for i in range(1, 101):
            sleep(0.01)
            if i % 5 == 0:
                s = randint(1, N)
                u = states[-1]
                pi_new = find_pi_new(u, q, s)
                # print(c(i), u, s, pi_new, q)
                begin = time()
                us.append(u)
                pn.append(pi_new)
                sn.append(s)
                P, log, converged = solve(c(i), u, s, pi_new, q)
                end = time()
                sol.append(end - begin)
                ensemble.change_policy(P)
            ss.append(s)
            states.append(copy(ensemble.get_state_distribution().reshape((N, 1))))
            states[-1] /= sum(states[-1])
            consumption.append(q.T.dot(states[-1]))
    finally:
        ensemble.stop()
