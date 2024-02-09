import numpy as np
import multiprocessing

class optimizer():

    def __init__(self, problem):
        self.problem = problem


    def stepsize_scheduler(self, stepsize_rule):

        # Choose stepsize
        if stepsize_rule == "decreasing_1_over_t":
            gamma_1 = 1.0 / (self.t+1)
            gamma_2 = 2 * gamma_1

        elif (stepsize_rule == "decreasing3" and self.problem.problem_type == 'quadratic'):

            gamma_2 = 1.0 / (self.t + 1)
            gamma_1 = gamma_2 / self.problem.n

        #
        # Bilinear Problem stepsize rules
        #
        elif (stepsize_rule == 'decreasing' and self.problem.problem_type == 'bilinear'):
            gamma_1 = 1.0 / (self.t + 1)
            gamma_2 = 4 * gamma_1
        elif (stepsize_rule=='Hsieh_r_gamma_0.7' and self.problem.problem_type == 'bilinear'):
            gamma_1 = 1.0/((self.t+19)**(0.7))
            gamma_2 = 0.1/ ((self.t+19) ** 0.7)
        elif (stepsize_rule == 'Hsieh_r_gamma_0' and self.problem.problem_type == 'bilinear'):
            gamma_1 = 1.0 / ((self.t + 19) ** (0.5))
            gamma_2 = 0.1
        elif (stepsize_rule == 'Hsieh_r_gamma_0.2' and self.problem.problem_type == 'bilinear'):
            gamma_1 = 1.0 / ((self.t + 19) ** (0.7))
            gamma_2 = 0.1 / ((self.t + 19) ** 0.2)
        else:
            raise TypeError("Unknown stepsize rule selected!")
        return (gamma_1, gamma_2)

    #
    # optimization algorithms
    #
    def SEG(self, gamma_1, gamma_2, x0, n_iter=1000, trials=10, rr=False, so=False, ig=False, stepsize_rule=False, return_trajectory=False, multiprocessing_enabled=True):
        """
            Multiprocessing implementation of Stochastic Extragradient Algorithm (SEG)
        -----
        Inputs:
        :param gamma_1: stepsize for update step
        :param gamma_2: stepsize for extrapolation step
        :param x0: initial point
        :param n_iter: number of iterations
        :param trials: number of times the method will run before outputing the average of experiments
        :param rr (Boolean): Random Reshuffling (uniform without-replacement sampling) enabled if True
        :param so: Shuffle Once sampling
        :param ig: Incremental Gradient variant enabled if True
        :param stepsize_rule: rule to be selected from stepsize_scheduler() (if provided)
        :param return_trajectory: If true, returns also the trajectory of the method
        :param multiprocessing_enabled: If true, multiprocessing is used.
        -----
        Returns: list with the final point and the relative error (|| x^t - x^*||^2 / || x^0 - x^*||^2)
        """

        args = [gamma_1, gamma_2, x0, n_iter,rr, so, ig,
                stepsize_rule, return_trajectory]
        if (multiprocessing_enabled and trials < multiprocessing.cpu_count()):
            print("Multiprocessing started...")
            # Start multiple processes
            pool = multiprocessing.Pool(processes=trials)

            # Execute the function in parallel using the pool of processes
            inputs = trials * [args]
            results = pool.starmap(self.SEG_implementation, inputs)

            # Compute the output as the average of all trials
            output_final_point = [v[0] for v in results]
            output_final_point = np.mean(output_final_point, axis=0)
            output_error = [v[1] for v in results]
            output_error = np.mean(output_error, axis=0)

            if return_trajectory:
                # trajectory should be returned
                trajectory_list = [v[2] for v in results]
                trajectory = np.mean(trajectory_list, axis=0)
        else:
            # Single-processing is used
            output_final_point_list = [[] for _ in range(trials)]
            output_error_list = [[] for _ in range(trials)]
            trajectory_list = [[] for _ in range(trials)]
            for i in range(trials):
                results = self.SEG_implementation(*args)
                output_final_point_list[i] = results[0]
                output_error_list[i] = results[1]
                if len(results) == 3:
                    trajectory_list.append(results[2])
            output_final_point = np.mean(output_final_point_list, axis=0)
            output_error = np.mean(output_error_list, axis=0)
            if return_trajectory:
                trajectory = np.mean(trajectory_list, axis=0)

        if return_trajectory:
            return [output_final_point, output_error, trajectory]
        else:
            return [output_final_point, output_error]


    def SEG_implementation(self, gamma_1, gamma_2, x0, n_iter, rr, so, ig, stepsize_rule, return_trajectory):
        """
            Implementation of Stochastic Extragradient for 1 run/core
        """
        # variables initialization
        relative_error = [1]
        x = x0
        if (so):
            perm = np.random.permutation(self.problem.n)
        self.num_perms = 0
        self.n_iter = n_iter
        perm_index = 0
        current_trajectory = [x0]

        # main loop
        for t in range(n_iter):
            self.t = t

            # for random reshuffling
            if rr and (t % self.problem.n == 0):
                perm = np.random.permutation(self.problem.n)
                self.num_perms += 1
                perm_index = 0
                if (self.num_perms % 10**6 == 0):
                    print("reached", self.num_perms)

            # choose sample to use for computing gradient
            if rr:
                index_sample = perm[perm_index%self.problem.n]
            elif (so or ig):
                index_sample = self.t % self.problem.n
            else:
                batch_size = 1
                index_sample = np.random.randint(self.problem.n)

            # choose your specific (complex) stepsize rule
            if (stepsize_rule):
                gamma_1, gamma_2 = self.stepsize_scheduler(stepsize_rule)

            # Update of SEG
            x_extrap = x - gamma_2 * self.problem.compute_grad(x, sample_indices=index_sample)
            x = x - gamma_1 * self.problem.compute_grad(x_extrap, sample_indices=index_sample)

            # save trajectory
            if return_trajectory:
                if self.n_iter >= 10**6:
                    if self.t % self.problem.n == 0:
                        current_trajectory.append(x)
                else:
                    current_trajectory.append(x)
            # compute error
            relative_error.append((np.linalg.norm(x - self.problem.sol) / np.linalg.norm(x0 - self.problem.sol)) ** 2)


        # return output
        if return_trajectory:
            return [x, relative_error, current_trajectory]
        return [x, relative_error]

    def SGDA(self, gamma, x0, n_iter=1000, trials=10, rr=False, so=False, ig=False,
            stepsize_rule=False, return_trajectory=False):

        return self.SEG(gamma, gamma_2=0, x0=x0, n_iter=n_iter, trials=trials, rr=rr, so=so, ig=ig,
            stepsize_rule=stepsize_rule, return_trajectory= return_trajectory)

    def SHGD(self, gamma, x0, n_iter=1000, trials=10, rr=False, so=False, ig=False, stepsize_rule=False,
            return_trajectory=False):
        """
            Implementation of Stochastic Hamiltonian Method
        ------
        Inputs:
        :param gamma: stepsize used
        :param x0: initial point
        :param n_iter: number of iterations
        :param trials: number of runs/trials
        :param rr: If True, random reshuffling is used as sampling.
        :param so: If True, shuffle once is used as sampling
        :param ig: If True, incremental gradient is implemented
        :param stepsize_rule: specific rule to be given to stepsize_scheduler()
        :param return_trajectory: If True, returns the trajectory of points
        ------
        Returns: list including the final point and list with relative errors at each iteration
        """
        args = [gamma, x0, n_iter, rr, so, ig, stepsize_rule, return_trajectory]
        if (trials < multiprocessing.cpu_count()):
            print("Multiprocessing started...")
            # Start multiple processes
            pool = multiprocessing.Pool(processes=trials)

            # Execute the function in parallel using the pool of processes
            inputs = trials * [args]
            results = pool.starmap(self.SHGD_implementation, inputs)
            # print(results)

            # Compute the output as the average of all trials
            output_final_point = [v[0] for v in results]
            output_final_point = np.mean(output_final_point, axis=0)
            output_error = [v[1] for v in results]
            output_error = np.mean(output_error, axis=0)
            # trajectory
            if return_trajectory:
                trajectory_list = [v[2] for v in results]
                trajectory = np.mean(trajectory_list, axis=0)
        else:
            output_final_point_list = [[] for _ in range(trials)]
            output_error_list = [[] for _ in range(trials)]
            trajectory_list = [[] for _ in range(trials)]
            for i in range(trials):
                results = self.SEG_implementation(*args)
                output_final_point_list[i] = results[0]
                output_error_list[i] = results[1]
                if len(results) == 3:
                    trajectory_list.append(results[2])
            output_final_point = np.mean(output_final_point_list, axis=0)
            output_error = np.mean(output_error_list, axis=0)
            if return_trajectory:
                trajectory = np.mean(trajectory_list, axis=0)

        if return_trajectory:
            return [output_final_point, output_error, trajectory]
        else:
            return [output_final_point, output_error]

    def SHGD_implementation(self, gamma, x0, n_iter=1000, rr=False, so=False, ig=False,
             stepsize_rule=False, return_trajectory=False):
        """
        Caution: this implementation is for bilinear games Only!
        To be implemented for more general functions in the future...
        """
        # initialization of variables
        relative_error = [1]
        x = x0
        if so:
            fixed_perm = np.ndarray.tolist(np.random.permutation(self.problem.n))
        elif ig:
            fixed_perm = range(self.problem.n)
        perm = []
        self.num_perms = 0
        self.n_iter = n_iter
        current_trajectory = [x0]

        # main loop
        for t in range(n_iter):
            self.t = t
            if rr and len(perm)==0:
                perm = np.ndarray.tolist(np.random.permutation(self.problem.n))
                self.num_perms += 1
                if (self.num_perms % 10 ** 6 == 0):
                    print("reached", self.num_perms)
            if (so or ig) and len(perm)==0:
                perm = fixed_perm
            # choose sample to use for computing gradient
            if rr or so or ig:
                index_sample1 = perm.pop(0)
                index_sample2 = perm.pop(0)
            else:
                index_sample1 = np.random.randint(self.problem.n)
                index_sample2 = np.random.randint(self.problem.n)
            if (stepsize_rule):
                gamma = self.stepsize_scheduler(stepsize_rule)

            # compute gradient
            if self.problem.problem_type == "bilinear":
                Q_ij = np.block([[self.problem.B[index_sample1]@self.problem.B[index_sample2].T, np.zeros((self.problem.d, self.problem.d))], [np.zeros((self.problem.d, self.problem.d)), self.problem.B[index_sample1].T@self.problem.B[index_sample2]]])
                q_ij_1st_coordinate = (1.0/2) * (self.problem.c[index_sample2].T @ self.problem.B[index_sample1].T + self.problem.c[index_sample1].T @ self.problem.B[index_sample2].T)
                q_ij_2nd_coordinate = (1.0/2) * (self.problem.a[index_sample2].T @ self.problem.B[index_sample1] + self.problem.a[index_sample1].T @ self.problem.B[index_sample2])
                q_ij = np.block([q_ij_1st_coordinate, q_ij_2nd_coordinate])
                grad_H = (1.0/2) * (Q_ij+Q_ij.T) @ x + q_ij.T
            elif self.problem.problem_type == "quadratic":
                M_index1 = np.block([[self.problem.A[index_sample1], self.problem.B[index_sample1]], [-self.problem.B[index_sample1].T, self.problem.C[index_sample1]]])
                M_index2 = np.block([[self.problem.A[index_sample2], self.problem.B[index_sample2]], [-self.problem.B[index_sample2].T, self.problem.C[index_sample2]]])
                term1 = M_index1 @ (M_index2@x + np.concatenate((self.problem.a[index_sample2],self.problem.c[index_sample2])))
                term2 = M_index1 @ (M_index2 @ x + np.concatenate((self.problem.a[index_sample2], self.problem.c[index_sample2])))
                grad_H = (1./2) * (term1 + term2)
            else:
                raise TypeError("Unknown problem type for SHGD!")
            # Update of SHGD
            x = x - gamma * grad_H

            # save trajectory
            if return_trajectory:
                current_trajectory.append(x)
            # compute error
            relative_error.append((np.linalg.norm(x - self.problem.sol) / np.linalg.norm(x0 - self.problem.sol)) ** 2)
        if return_trajectory:
            return [x, relative_error, current_trajectory]
        return [x, relative_error]