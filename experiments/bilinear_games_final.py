import optimizer_module as op
import numpy as np
import pickle

def SEG_RR_vs_SEG():
    #
    # SEG-RR vs SEG experiment (produces plot in the intro of the paper)
    #
    n = 100
    d = 1
    n_iter = 3 * 10 ** 6
    trials = 2
    x0 = np.random.normal(0, 1, 2 * d)
    problem = op.problem(problem_type="bilinear", n=n, d=d, mu=.1, L=10)
    optimizer = op.optimizer(problem)
    # steps
    Omega_bar = (optimizer.problem.L_max / (optimizer.problem.n * optimizer.problem.mu)) * (
            2 * (24 * n ** 2 - 23 * n + 1 + (2.5 * (n - 1) * (2 * n - 1))) * optimizer.problem.L ** 2 +
            optimizer.problem.A_const * (24 * (optimizer.problem.n ** 2) - 23 * optimizer.problem.n + 1))
    gamma_1_max = min(optimizer.problem.mu / (4 * (optimizer.problem.L_max ** 2)),
                      1.0 / (3 * np.sqrt(
                          2 * optimizer.problem.n * (optimizer.problem.n - 1)) * optimizer.problem.L_max), (np.sqrt(
            (optimizer.problem.L_max ** 4) + (optimizer.problem.n * optimizer.problem.mu * Omega_bar / 4)) - (
                                                                                                                    optimizer.problem.L_max ** 2)) / Omega_bar)
    # #Uncomment to run the experiments
    # finalSEG_RR, results_SEG_RR, SEG_RR_trajectory = optimizer.SEG(gamma_1=1/(1000*optimizer.problem.L), gamma_2=4*1/(1000*optimizer.problem.L), x0=x0, n_iter=n_iter, trials=trials, rr=True, return_trajectory=True)
    # finalSEG, results_SEG, SEG_trajectory = optimizer.SEG(gamma_1=1/(1000*optimizer.problem.L), gamma_2=4*1/(1000*optimizer.problem.L), x0=x0, n_iter=n_iter, trials=trials, rr=False, return_trajectory=True)

    # #save results
    # results = {"relative_error_SEG_RR": results_SEG_RR,
    #            "SEG_RR_trajectory": SEG_RR_trajectory,
    #            "relative_error_SEG": results_SEG,
    #            "SEG_trajectory": SEG_trajectory}
    # with open('final codes/2d_SEG_RR_vs_SEG_bilinear_huge_stepsizes2.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # load results
    with open('final codes/2d_SEG_RR_vs_SEG_bilinear_huge_stepsizes2.pkl', 'rb') as f:
        results = pickle.load(f)
    results_SEG_RR = results["relative_error_SEG_RR"]
    SEG_RR_trajectory = results["SEG_RR_trajectory"]
    results_SEG = results["relative_error_SEG"]
    SEG_trajectory = results["SEG_trajectory"]
    sol = [0.006, 0.009]

    sparsified_trajectory_SEG_RR = [SEG_RR_trajectory[i] for i in range(len(SEG_RR_trajectory)) if i % 10 ** 5 == 0]
    sparsified_trajectory_SEG = [SEG_trajectory[i] for i in range(len(SEG_trajectory)) if i % 10 ** 5 == 0]
    sparsified_rel_SEG_RR = [results_SEG_RR[i] for i in range(len(results_SEG_RR)) if i % 10 ** 5 == 0]
    sparsified_rel_SEG = [results_SEG[i] for i in range(len(results_SEG)) if i % 10 ** 5 == 0]

    op.plot_multiple_var([sparsified_rel_SEG, sparsified_rel_SEG_RR], labels=["SEG", "SEG-RR"],
                         y_label="Relative Error", x_label="Iterations (x$10^5$)", title="Bilinear Game", save_figure=True, filename="trajectory_2D")
    op.plot_2d([sparsified_trajectory_SEG, sparsified_trajectory_SEG_RR], sol=sol, title="2D Trajectory Plot",
               labels=["SEG", "SEG-RR"], save_figure=True, filename="2D_plot_main")

def experiment_SEG_RR_vs_SO_vs_IG():
    # SEG-RR vs SEG experiment
    #
    n = 10
    d = 1
    n_iter = 1 * 10 ** 6
    trials = 5
    condition_num_list = [1, 5, 10, 100]
    # for i in range(len(condition_num_list)):
    #     condition_number = condition_num_list[i]
    #     print("start exp with condition_number ", condition_number)
    #     problem = op.problem(problem_type="bilinear", n=n, d=d, mu=1, L=condition_number)
    #     optimizer = op.optimizer(problem)
    #
    #     x0 = np.random.normal(0, 1, 2 * d)
    #     # steps
    #     gamma_1_max = optimizer.problem.mu / (2 * np.sqrt(120) * problem.n * (optimizer.problem.L_max ** 2))
    #
    #     # Run SEG-RR
    #     final_point_SEG_RR, relative_error_SEG_RR = optimizer.SEG(gamma_1=gamma_1_max,
    #                                                                  gamma_2=4 * gamma_1_max, x0=x0,
    #                                                                  n_iter=n_iter, trials=trials,
    #                                                                  rr=True)
    #     # Run SEG
    #     final_point_SEG, relative_error_SEG = optimizer.SEG(gamma_1=gamma_1_max,
    #                                                             gamma_2=4 * gamma_1_max, x0=x0,
    #                                                             n_iter=n_iter, trials=trials, rr=False)
    #
    #     # Run SEG-IG
    #     final_point_SEG_IG, relative_error_SEG_IG = optimizer.SEG(gamma_1=gamma_1_max,
    #                                                                  gamma_2=4 * gamma_1_max, x0=x0,
    #                                                                  n_iter=n_iter, trials=trials,
    #                                                                  ig=True)
    #     # Run SEG-SO
    #     final_point_SEG_SO, relative_error_SEG_SO = optimizer.SEG(gamma_1=gamma_1_max,
    #                                                                          gamma_2=4 * gamma_1_max, x0=x0,
    #                                                                          n_iter=n_iter, trials=trials,
    #                                                                          so=True)
    #
    #     # save results
    #     results = {"n": n,
    #                "dimension": d,
    #                "lambda_min": 1,
    #                "condition_number": condition_num_list[i],
    #                "x0": x0,
    #                "stepsizes_used": {"gamma_1": gamma_1_max, "gamma_2": 4 * gamma_1_max},
    #                "relative_error_SEG_RR": relative_error_SEG_RR,
    #                "final_point_SEG_RR": final_point_SEG_RR,
    #                "relative_error_SEG": relative_error_SEG,
    #                "final_point_SEG": final_point_SEG,
    #                "relative_error_SEG_IG": relative_error_SEG_IG,
    #                "final_point_SEG_IG": final_point_SEG_IG,
    #                "relative_error_SEG_SO": relative_error_SEG_SO,
    #                "final_point_SEG_SO": final_point_SEG_SO,
    #                "solution": optimizer.problem.sol}
    #     with open('final codes/bilinear_SEG_RR_vs_IG_vs_SO_cond_num' + str(condition_number) + '.pkl', 'wb') as f:
    #         pickle.dump(results, f)

    for i in range(len(condition_num_list)):
        condition_number = condition_num_list[i]
        # load results
        with open('final codes/bilinear_SEG_RR_vs_IG_vs_SO_cond_num'+ str(
                condition_number) + ".pkl", 'rb') as f:
            results = pickle.load(f)

        relative_error_SEG_RR = results["relative_error_SEG_RR"]
        relative_error_SEG = results["relative_error_SEG"]
        relative_error_SEG_SO = results["relative_error_SEG_SO"]
        relative_error_SEG_IG = results["relative_error_SEG_IG"]
        solution = results["solution"]

        # sparsify vectors
        sparsified_relative_error_SEG_RR = [relative_error_SEG_RR[i] for i in range(len(relative_error_SEG_RR)) if
                                            i % 10 ** 5 == 0]
        sparsified_relative_error_SEG = [relative_error_SEG[i] for i in range(len(relative_error_SEG)) if
                                         i % 10 ** 5 == 0]
        sparsified_relative_error_SEG_SO = [relative_error_SEG_SO[i] for i in range(len(relative_error_SEG_SO)) if
                                            i % 10 ** 5 == 0]
        sparsified_relative_error_SEG_IG = [relative_error_SEG_IG[i] for i in range(len(relative_error_SEG_IG)) if
                                         i % 10 ** 5 == 0]

        op.plot_multiple_var([sparsified_relative_error_SEG[:60], sparsified_relative_error_SEG_RR[:60], sparsified_relative_error_SEG_SO[:60], sparsified_relative_error_SEG_IG[:60]], labels=["SEG", "SEG-RR","SEG-SO","IEG"],
                             x_label="Iterations (x$10^5$)",y_label="Relative Error",
                             title="Bilinear Problem", save_figure=True, filename="Bilinear_SEG_RR_vs_IG_vs_SO_cond_num"+str(condition_number))

def experiment_SEG_RR_vs_SEG_Hsieh_stepsizes():
    #
    # SEG-RR vs SEG experiment for stepsizes as in Hsieh et al.
    #
    n = 100
    d = 100
    n_iter = 1 * 10 ** 6
    trials = 5
    condition_num_list = [1, 5, 10]
    x0 = np.random.normal(0, 1, 2 * d)
    stepsize_rule_list = ["Hsieh_r_gamma_0"]

    # #Uncomment to run experiments
    # for i in range(len(condition_num_list)):
    #     condition_number = condition_num_list[i]
    #     print("start exp with condition_number ", condition_number)
    #     problem = op.problem(problem_type="bilinear", n=n, d=d, mu=1, L=condition_number)
    #     optimizer = op.optimizer(problem)
    #     # steps
    #     gamma_1_max = optimizer.problem.mu / (2 * np.sqrt(120) * problem.n*(optimizer.problem.L_max ** 2))
    #
    #     # Run SEG-RR
    #     final_point_SEG_RR, relative_error_SEG_RR = optimizer.SEG(gamma_1=gamma_1_max, gamma_2=4*gamma_1_max, stepsize_rule=stepsize_rule_list[i], x0=x0, n_iter=n_iter, trials=trials, rr=True)
    #     # Run SEG
    #     final_point_SEG, relative_error_SEG = optimizer.SEG(gamma_1=gamma_1_max, gamma_2=4*gamma_1_max, stepsize_rule=stepsize_rule_list[i], x0=x0, n_iter=n_iter, trials=trials, rr=False)
    #
    #     # save results
    #     results = {"n": n,
    #                 "dimension": d,
    #                 "lambda_min": 1,
    #                 "condition_number": condition_num_list[i],
    #                 "x0": x0,
    #                 "stepsizes_used": {"gamma_1": gamma_1_max, "gamma_2": 4*gamma_1_max},
    #                 "relative_error_SEG_RR": relative_error_SEG_RR,
    #                 "final_point_SEG_RR": final_point_SEG_RR,
    #                 "relative_error_SEG": relative_error_SEG,
    #                 "final_point_SEG": final_point_SEG,
    #                 "solution": optimizer.problem.sol}
    #     with open('final codes/bilinear_Hsieh_steps_cond_num' + str(condition_num_list[i]) + '.pkl', 'wb') as f:
    #         pickle.dump(results, f)

    for i in range(len(condition_num_list)):
        condition_number = condition_num_list[i]

        # load results
        with open('final codes/bilinear_Hsieh_steps_cond_num' + str(condition_num_list[i]) + '.pkl', 'rb') as f:
            results = pickle.load(f)

        relative_error_SEG_RR = results["relative_error_SEG_RR"]
        relative_error_SEG = results["relative_error_SEG"]
        solution = results["solution"]

        sparsified_relative_error_SEG_RR = [relative_error_SEG_RR[i] for i in range(len(relative_error_SEG_RR)) if
                                            i % 10 ** 5 == 0]
        sparsified_relative_error_SEG = [relative_error_SEG[i] for i in range(len(relative_error_SEG)) if
                                         i % 10 ** 5 == 0]

        op.plot_multiple_var([sparsified_relative_error_SEG, sparsified_relative_error_SEG_RR],
                             labels=["SEG", "SEG-RR"], y_label="Relative Error",
                             x_label="Iterations (x$10^4$)",
                             title="Bilinear Problem ($L_{max}=$" + str(condition_number) + ")", save_figure=True, filename="Bilinear_Hsieh2_stepsizes_cond_num"+str(condition_number))

def experiment_SEG_RR_vs_SEG_large_steps_further_experiments():
    # SEG-RR vs SEG experiment
    #
    n = 100
    d = 1
    n_iter = 1 * 10 ** 5
    trials = 5
    condition_num_list = [1, 5, 10]
    # #Uncomment to run experiments
    # for i in range(len(condition_num_list)):
    #     # initialize problem
    #     condition_number = condition_num_list[i]
    #     print("start exp with condition_number ", condition_number)
    #     problem = op.problem(problem_type="bilinear", n=n, d=d, mu=1, L=condition_number)
    #     optimizer = op.optimizer(problem)
    #     x0 = np.random.normal(problem.sol, problem.sol + 1, 2 * d)
    #     # steps
    #     stepsize_list = [1.0 / (10 * problem.L_max), 1.0 / (100 * problem.L_max)]
    #     relative_error_SEG_RR_list = []
    #     relative_error_SEG_list = []
    #     final_point_SEG_RR_list = []
    #     final_point_SEG_list = []
    #
    #     for step in stepsize_list:
    #         # Run SEG-RR
    #         final_point_SEG_RR, relative_error_SEG_RR = optimizer.SEG(gamma_1=step,
    #                                                                      gamma_2=4 * step, x0=x0,
    #                                                                      n_iter=n_iter, trials=trials,
    #                                                                      rr=True)
    #         # Run SEG
    #         final_point_SEG, relative_error_SEG = optimizer.SEG(gamma_1=step,
    #                                                             gamma_2=4 * step, x0=x0,
    #                                                             n_iter=n_iter, trials=trials, rr=False)
    #         relative_error_SEG_RR_list.append(relative_error_SEG_RR)
    #         relative_error_SEG_list.append(relative_error_SEG)
    #         final_point_SEG_RR_list.append(final_point_SEG_RR)
    #         final_point_SEG_list.append(final_point_SEG)
    #
    #     # save results
    #     results = {"n": n,
    #                "dimension": d,
    #                "lambda_min": 1,
    #                "condition_number": condition_num_list[i],
    #                "x0": x0,
    #                "stepsizes_used": stepsize_list,
    #                "relative_error_SEG_RR_list": relative_error_SEG_RR_list,
    #                "final_point_SEG_RR_list": final_point_SEG_RR_list,
    #                "relative_error_SEG_list": relative_error_SEG_list,
    #                "final_point_SEG_list": final_point_SEG_list,
    #                "solution": optimizer.problem.sol}
    #     with open('final codes/bilinear_SEG_RR_vs_SEG_large_steps_further_exp_cond_num' + str(condition_number) + '.pkl',
    #               'wb') as f:
    #         pickle.dump(results, f)

    for i in range(len(condition_num_list)):
        condition_number = condition_num_list[i]
        # load results
        with open('final codes/bilinear_SEG_RR_vs_SEG_large_steps_further_exp_cond_num' + str(
                condition_number) + ".pkl", 'rb') as f:
            results = pickle.load(f)

        relative_error_SEG_RR_list = results["relative_error_SEG_RR_list"]
        relative_error_SEG_list = results["relative_error_SEG_list"]
        solution = results["solution"]
        # run for different steps
        for step in range(2):

            relative_error_SEG_RR = relative_error_SEG_RR_list[step]
            relative_error_SEG = relative_error_SEG_list[step]

            sparsified_relative_error_SEG_RR = [relative_error_SEG_RR[i] for i in range(len(relative_error_SEG_RR))
                                                if
                                                i % 10 ** 4 == 0]
            sparsified_relative_error_SEG = [relative_error_SEG[i] for i in range(len(relative_error_SEG)) if
                                             i % 10 ** 4 == 0]

            op.plot_multiple_var([sparsified_relative_error_SEG, sparsified_relative_error_SEG_RR],
                                 labels=["SEG", "SEG-RR"], y_label="Relative Error",
                                 x_label="Iterations (x$10^4$)",
                                 title="Bilinear Problem ($L_{max}=$" + str(
                                     condition_number) + ", $\gamma_1 = 1/($" + str(10 ** (step + 1)) + "L))",
                                 save_figure=True,
                                 filename="Bilinear_further_exp_step_" + str(step) + "_cond_num_" + str(
                                     condition_number))


if __name__ == '__main__':
    print("Starting bilinear experiments ...")
    print("Experiment 1 started")
    SEG_RR_vs_SEG()

    print("Experiment 2 started")
    experiment_SEG_RR_vs_SO_vs_IG()

    print("Experiment 3 started")
    experiment_SEG_RR_vs_SEG_Hsieh_stepsizes()

    print("Experiment 4 started")
    experiment_SEG_RR_vs_SEG_large_steps_further_experiments()