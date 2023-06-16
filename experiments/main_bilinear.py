import optimizer_module as op
import numpy as np
import pickle

def run_tests():
    n = 5
    d = 100
    x0 = np.zeros(2*d)
    problem = op.problem(problem_type="bilinear", n=n, d=d, mu=1, L=2)
    optimizer = op.optimizer(problem)
    D =(n+1) + 2.5*(n-1)*(2*n-1)+ 24*n*(n-1)
    D1 = 1.0/2.0
    Omega = (problem.L_max) /(n*problem.mu) *((D*problem.L**2/D1) + (problem.A_const * ( 24 * (n **2) + 23 * n + 1)/(2* D1)))
    gamma1_max = min(problem.mu /(4*problem.L_max**2), 1 / (6 * n * (problem.L_max ** 2)),
                     (np.sqrt((problem.L_max ** 4) + (Omega * n * problem.mu / 4)) - problem.L_max ** 2) / Omega)
    print(gamma1_max)
    print(16//((optimizer.problem.n**3)*(gamma1_max**2)*(optimizer.problem.mu**2)))
    # _, results_EG_rrconstant = optimizer.SEG(gamma_1=gamma1_max, gamma_2=4*gamma1_max, x0=x0, n_iter=1 * 10**6,
    #                                            trials=3, rr=True)
    _, results_EG_rrdecreasing = optimizer.SEG(gamma_1 = 1e-3, gamma_2 = 5e-3, x0=x0, n_iter=1 * 10**6, trials = 3, stepsize_rule="decreasing", rr=True)

    # save results in file
    results = {"rr_decreasing": results_EG_rrdecreasing}
    with open('results_bilinear_decreasing_k_star_K_over_two.pkl', 'wb') as f:
        pickle.dump(results, f)
    # load results from file
    with open('results_bilinear_constant_vs_decreasing_as_in_thms.pkl', 'rb') as f:
        results = pickle.load(f)
    results_EG_rrconstant = results["rr_constant"]
    # results_EG_rrdecreasing = results["rr_decreasing"]
    sparcified_rel = [results_EG_rrconstant[i] for i in range(len(results_EG_rrconstant)) if i % 10**0 == 0]
    sparcified_rel2 = [results_EG_rrdecreasing[i] for i in range(len(results_EG_rrdecreasing)) if i % 10**0 == 0]
    op.plot_multiple_var([sparcified_rel, sparcified_rel2], labels=["Constant as in Thm 4.1", "Decreasing as in Thm 4.2"], x_label="Iterations (x10^5)", title="Bilinear Problem")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_tests()

