from scipy.stats import ortho_group
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import multiprocessing

class problem:

    def __init__(self, problem_type, n, d, mu=0.1, L=1):
        self.problem_type = problem_type
        self.n = n
        self.d = d
        self.mu = mu
        self.L = L
        if self.problem_type == "quadratic":
            self.generate_quadratic_problem()
        elif self.problem_type == "bilinear":
            self.generate_bilinear_problem()
        elif self.problem_type == "robust_ls":
            self.generate_robust_ls_problem()
        else:
            print("Select a type of problem to be generated!")

    def generate_quadratic_problem(self):
        self.A = [[[]] for i in range(self.n)]
        self.C = [[[]] for i in range(self.n)]
        self.B = [[[]] for i in range(self.n)]
        self.a = np.empty(shape=(self.n, self.d))
        self.c = np.empty(shape=(self.n, self.d))


        self.A = [[[]] for i in range(self.n)]
        self.C = [[[]] for i in range(self.n)]
        self.B = [[[]] for i in range(self.n)]
        self.a = np.empty(shape=(self.n, self.d))
        self.c = np.empty(shape=(self.n, self.d))

        Q1 = ortho_group.rvs(self.d)
        Q2 = ortho_group.rvs(self.d)
        Q3 = ortho_group.rvs(self.d)
        for i in range(self.n):
            # generate matrix A
            diagonal_entries = np.random.uniform(self.mu, self.L, size=self.d - 2)
            diagonal_entries = np.append(diagonal_entries, [self.mu, self.L])
            diagonal_entries = [self.mu + (diagonal_entries[i] - self.mu) *(self.L - self.mu)/ (self.L - self.mu) for i in range(self.d)]
            D = np.diag(diagonal_entries)
            self.A[i] = Q1 @ D @ Q1.T

            # generate matrix C
            diagonal_entries = np.random.uniform(self.mu, self.L, size=self.d - 2)
            diagonal_entries = np.append(diagonal_entries, [self.mu, self.L])
            diagonal_entries = [self.mu + (diagonal_entries[i] - self.mu) *(self.L - self.mu)/ (self.L - self.mu) for i in range(self.d)]
            D = np.diag(diagonal_entries)
            self.C[i] = Q2 @ D @ Q2.T

            # generate matrix B
            diagonal_entries = np.random.uniform(self.mu, self.L, size=self.d - 2)
            diagonal_entries = np.append(diagonal_entries, [self.mu, self.L])
            diagonal_entries = [self.mu + (diagonal_entries[i] - self.mu) *(self.L - self.mu)/ (self.L - self.mu) for i in range(self.d)]
            D = np.diag(diagonal_entries)
            self.B[i] = Q3 @ D @ Q3.T

            # generate matrix  a, c
            self.a[i] = np.random.normal(0, 1, size=self.d)
            self.c[i] = np.random.normal(0, 1, size=self.d)

        sum_matrix_A = np.mean(self.A, axis=0)
        sum_matrix_B = np.mean(self.B, axis=0)
        sum_matrix_C = np.mean(self.C, axis=0)
        self.sum_matrix_a = np.mean(self.a, axis=0)
        self.sum_matrix_c = np.mean(self.c, axis=0)

        self.M = np.block([[sum_matrix_A, sum_matrix_B], [-sum_matrix_B.T, sum_matrix_C]])
        self.sol = np.linalg.solve(self.M, - np.concatenate((self.sum_matrix_a, self.sum_matrix_c)))

        # To be deleted...
        L_i_cur = []
        for i in range(self.n):
            M_i = np.block([[self.A[i], self.B[i]], [-self.B[i], self.C[i]]])
            eigenvalues, _ = np.linalg.eig(M_i.T @ M_i)
            L_i_cur.append(np.real(np.sqrt(max(eigenvalues))))
        self.L_max = max(np.real(L_i_cur))
        eig_A, _ = np.linalg.eig(sum_matrix_A)
        eig_C, _ = np.linalg.eig(sum_matrix_C)
        eig_B, _ = np.linalg.eig(sum_matrix_B)
        print("mu are:", min(np.real(eig_A)), min(np.real(eig_B)), min(np.real(eig_C)))
        print("L are:", max(np.real(eig_A)), max(np.real(eig_B)), max(np.real(eig_C)))

        # computation of values of A,B,C constants in Assumption on Growth Condition
        self.A_const = 0
        for i in range(len(L_i_cur)):
            self.A_const += L_i_cur[i] ** 2
        self.A_const = self.A_const / self.n

    def generate_bilinear_problem(self):
        # Bilinear Problem (generate the problem)
        ## Initialize Variables
        self.A = np.zeros(shape=(self.n, self.d, self.d))
        # for i in range(self.n):
        #   for k in range(self.n):
        #     for l in range(self.n):
        #       if(i==k and k==l):
        #         self.A[i][k][l] = 1
        Q = ortho_group.rvs(self.d)
        for i in range(self.n):
            # generate matrix A_bil
            diagonal_entries = np.random.uniform(self.mu, self.L, size=self.d - 2)
            diagonal_entries = np.append(diagonal_entries, [self.mu, self.L])
            D = np.diag(diagonal_entries)
            self.A[i] = Q @ D @ Q.T
        self.b = []
        self.c = []
        for i in range(self.n):
            self.b.append(np.random.normal(0, 1 / self.n, size=self.d))
            self.c.append(np.random.normal(0, 1 / self.n, size=self.d))
        # Compute solution
        A_sum = np.mean(self.A, axis=0)
        self.b_sum = np.mean(self.b, axis=0)
        self.c_sum = np.mean(self.c, axis=0)

        self.M = np.block([[np.zeros((self.d, self.d)), A_sum], [-A_sum.T, np.zeros((self.d, self.d))]])
        eig, _ = np.linalg.eig(self.M.T @ self.M)
        L = np.sqrt(np.real(max(eig)))
        self.sol = np.linalg.solve(self.M, - np.concatenate((self.b_sum, -self.c_sum)))

        L_i = []
        mu_i = []
        for i in range(self.n):
            M_i = np.block(
                [[np.zeros((self.d, self.d)), self.A[i]], [-self.A[i].T, np.zeros((self.d, self.d))]])
            eigenvalues, _ = np.linalg.eig(M_i.T @ M_i)
            L_i.append(np.sqrt(max(eigenvalues)))
            mu_i.append(np.sqrt(min(eigenvalues)))
        self.L_max = np.real(max(L_i))
        eig, _ = np.linalg.eig(A_sum)
        mu = np.real(min(eig))
        print(mu, self.L_max, L)
        # computation of value of A_const = (1/n) * sum(L_i**2)
        self.A_const = 0
        for i in range(len(L_i)):
            self.A_const += L_i[i] ** 2
        self.A_const = self.A_const / self.n

    def generate_robust_ls_problem(self):
        # Robust Least Squares Problem (generate the problem)
        #
        # Initialize Variables
        self.lambda_coef = 10
        self.A = np.random.normal(0, 1, size=(self.d, self.d))
        # make A symmetric
        # self.A[f_component] = (np.array(self.A[f_component]) + np.array(self.A[f_component]).T) / 2
        epsilon = np.random.normal(0, 0.01, size=self.d)
        x_star = np.random.normal(0, 1, size=self.d).T
        sum_A = np.mean(self.A, axis=0)
        self.y_0 = sum_A @ x_star + epsilon
        self.sol = np.zeros(2*self.d)

    def function_value(self, x):
        x1 = x[0:self.d]
        x2 = x[self.d:]
        f = 0
        for i in range(self.n):
            if (self.problem_type == "quadratic"):
                f += (1 / 2) * x1.T @ self.A[i] @ x1 + x1.T @ self.B[i] @ x2 - (1 / 2) * x2.T @ self.C[i] @ x2 + self.a[
                    i].T @ x1 - self.c[i].T @ x2
            elif (self.problem_type == "bilinear"):
                f += x1.T @ self.b_bil[i] + x1.T @ self.A_bil[i] @ x2 + self.c_bil[i].T @ x2
            else:
                print("Unknown problem type!")
        return (f / self.n)

    def operator(self, x, index=None):
        if self.problem_type == "quadratic":
            if index == None:
                return (self.M @ x + np.concatenate((self.sum_matrix_a, self.sum_matrix_c)))
            else:
                M_index = np.block([[self.A[index], self.B[index]], [-self.B[index].T, self.C[index]]])
                grad = (1.0/self.n) * (M_index @ x + np.concatenate((self.a[index], self.c[index])))
                return (grad)
        elif self.problem_type == "bilinear":
            if index == None:
                return (self.M @ x + np.concatenate((self.b_sum, -self.c_sum)))
            else:
                M_i = np.block([[np.zeros((self.d, self.d)), self.A[index]],
                                [-self.A[index].T, np.zeros((self.d, self.d))]])
                return ((1.0/self.n) * (M_i @ x + np.concatenate((self.b[index], -self.c[index]))))

        elif self.problem_type == "robust_ls":
            grad_x = np.zeros(self.d)
            grad_y = np.zeros(self.d)
            if (index == None):
                for elem in range(self.n):
                    grad_x += (2 / self.n) * np.array(self.A[elem]).T @ (self.A[elem] @ x[0:self.d] - x[self.d:])
                    grad_y += (2 / self.n) * (self.A[elem] @ x[0:self.d] - x[self.d:]) - (2 * self.lambda_coef / self.n) * (x[self.d:] - self.y_0)
                return (np.concatenate((grad_x, grad_y), axis=0))
            else:
                grad_x = (2 / self.n) * np.array(self.A[index]).T @ (self.A[index] @ x[0:self.d] - x[self.d:][index])
                grad_y = (2 / self.n) * (self.A[index] @ x[0:self.d] - x[self.d:][index]) + (2 * self.lambda_coef) * (x[self.d:][index] - self.y_0[index])
                # print(np.shape(np.concatenate((grad_w,grad_delta)))[0])
                return (np.concatenate((grad_x, grad_y), axis=0))
        else:
            print("Unknown problem type!")


class optimizer():

    def __init__(self, problem):
        self.problem = problem


    def stepsize_scheduler(self, stepsize_rule):

        # Choose stepsize
        if (stepsize_rule == 'decreasing' and self.problem.problem_type == 'quadratic'):
            gamma_2_max = min(self.problem.mu / (self.problem.n * self.problem.L_max * np.real(
                24 * np.sqrt(165 * self.problem.A_const + 126 * (self.problem.L ** 2)))),
                              (-self.problem.mu + np.sqrt((self.problem.mu ** 2) + (self.problem.L ** 2) * (
                                      1 - (1 / np.sqrt(self.problem.n))))) / (
                                      self.problem.L ** 2))

            if (self.num_perms <= self.num_iter // self.problem.n):
                gamma_2 = gamma_2_max
            else:
                gamma_2 = 1.0 / (self.problem.n*(self.num_perms + 1))
            gamma_1 = 1 / self.problem.n
        elif (stepsize_rule == 'EG_rr_decreasing' and self.problem.problem_type == 'quadratic'):
            gamma_2_max = min(self.problem.mu / (self.problem.n * self.problem.L_max * np.real(
                24 * np.sqrt(165 * self.problem.A_const + 126 * (self.problem.L ** 2)))),
                              (-self.problem.mu + np.sqrt((self.problem.mu ** 2) + (self.problem.L ** 2) * (
                                          1 - (1 / np.sqrt(self.problem.n))))) / (
                                      self.problem.L ** 2))

            k_star = 64 / ((self.problem.mu ** 2) * (gamma_2_max ** 2) * (
                        (24 * (self.problem.n ** 2) - 23 * self.problem.n + 1) / (self.problem.n ** 2)))

            if (self.num_perms <= k_star):
                gamma_2 = gamma_2_max
            else:
                gamma_2 = (4.0) * (2 * self.num_perms + 1) / (self.problem.mu * ((self.num_perms + 1) ** 2))
            gamma_1 = gamma_2 / self.problem.n
        elif(stepsize_rule == 'EG_rr_decreasing_2h' and self.problem.problem_type == 'quadratic'):
            gamma_2_max = min(self.problem.mu / (self.problem.n * self.problem.L_max * np.real(
                24 * np.sqrt(165 * self.problem.A_const + 126 * (self.problem.L ** 2)))),
                              (-self.problem.mu + np.sqrt((self.problem.mu ** 2) + (self.problem.L ** 2) * (
                                      1 - (1 / np.sqrt(self.problem.n))))) / (
                                      self.problem.L ** 2))

            k_star = self.n_iter // (2*self.problem.n)

            if (self.num_perms <= k_star):
                gamma_2 = gamma_2_max
            else:
                gamma_2 = (4.0) * (2 * self.num_perms + 1) / (self.problem.mu * ((self.num_perms + 1) ** 2))
            gamma_1 = gamma_2 / self.problem.n

        elif (stepsize_rule == 'EG_rr_decreasing2' and self.problem.problem_type == 'quadratic'):
            gamma_2_max = min(self.problem.mu / (self.problem.n * self.problem.L_max * np.real(
                24 * np.sqrt(165 * self.problem.A_const + 126 * (self.problem.L ** 2)))),
                              (-self.problem.mu + np.sqrt((self.problem.mu ** 2) + (self.problem.L ** 2) * (
                                          1 - (1 / np.sqrt(self.problem.n))))) / (
                                      self.problem.L ** 2))

            self.k_star = 64 /((self.problem.mu**2) * (gamma_2_max**2)*((24*(self.n**2) - 23*self.n+1)/(self.n**2)))
            if (self.num_perms <= self.k_star):
                gamma_2 = gamma_2_max
            else:
                gamma_2 = (4.0) * (2 * self.num_perms + 1) / (self.problem.mu * ((self.num_perms + 1) ** 2))
            gamma_1 = gamma_2 / self.problem.n
        elif (stepsize_rule == "decreasing2" and self.problem.problem_type == 'quadratic'):
            gamma_2_max = min(self.problem.mu / (self.problem.n * self.problem.L_max * np.real(
                        24 * np.sqrt(165 * self.problem.A_const + 126 * (self.problem.L ** 2)))),
                        (-self.problem.mu + np.sqrt((self.problem.mu ** 2) + (self.problem.L ** 2) * (
                        1 - (1 / np.sqrt(self.problem.n))))) / (
                        self.problem.L ** 2))

            if (gamma_2_max <= 1.0 / (self.problem.n* ((self.num_perms) ** 2))):
                gamma_2 = gamma_2_max
            else:
                gamma_2 = 1.0 / (self.problem.n* ((self.num_perms) ** 2))
            gamma_1 = gamma_2 / self.problem.n
        elif (stepsize_rule == "decreasing3" and self.problem.problem_type == 'quadratic'):
            gamma_2_max = min(self.problem.mu / (self.problem.n * self.problem.L_max * np.real(
                        24 * np.sqrt(165 * self.problem.A_const + 126 * (self.problem.L ** 2)))),
                        (-self.problem.mu + np.sqrt((self.problem.mu ** 2) + (self.problem.L ** 2) * (
                        1 - (1 / np.sqrt(self.problem.n))))) / (
                        self.problem.L ** 2))

            if (self.num_perms <= self.n_iter / (2*self.problem.n)):
                gamma_2 = 5e-3 #gamma_2_max
            else:
                gamma_2 = 100.0 / (self.problem.n*((self.num_perms + 1) ** 3))
            gamma_1 = gamma_2 / self.problem.n
        elif (stepsize_rule == 'decreasing' and self.problem.problem_type == 'bilinear'):
            gamma_1 = 1.0 / (self.t + 1)
            gamma_2 = 4 * gamma_1
        elif (stepsize_rule == 'EG_rr_decreasing' and self.problem.problem_type == 'bilinear'):
            D = (self.problem.n + 1) + (2.5 * (self.problem.n - 1) * (2 * self.problem.n - 1)) + (
                        24 * self.problem.n * (self.problem.n - 1))
            D_1 = 1.0 / 2
            Omega = (self.problem.L_max / (self.problem.n * self.problem.mu)) * (2 * (D * (self.problem.L ** 2)) + (
                        self.problem.A_const * (24 * (self.problem.n ** 2) + 23 * self.problem.n - 1)))
            gamma_1_max = min(self.problem.mu / (4 * (self.problem.L_max ** 2)),
                              1.0 / (3 * np.sqrt(self.problem.n * (self.problem.n - 1)) * self.problem.L_max), (np.sqrt(
                    (self.problem.L_max ** 4) + (self.problem.n * self.problem.mu * Omega / 4)) - (
                                                                                                                            self.problem.L_max ** 2)) / Omega)
            k_star = self.n_iter // (2*self.problem.n) #16//((self.problem.n**3)*(gamma_1_max**2)*(self.problem.mu**2))
            if (self.num_perms <= k_star):
                gamma_1 = gamma_1_max
            else:
                gamma_1 = (2.0) * (2 * self.num_perms + 1) / (
                            self.problem.mu * self.problem.n * ((self.num_perms + 1) ** 2))
            gamma_2 = 4 * gamma_1
        elif (stepsize_rule == 'decreasing' and self.problem.problem_type == 'bilinear'):
            D = (self.problem.n + 1) + (2.5 * (self.problem.n - 1) * (2 * self.problem.n - 1)) + (
                        24 * self.problem.n * (self.problem.n - 1))
            D_1 = 1.0 / 2
            Omega = (self.problem.L_max / (self.problem.n * self.problem.mu)) * (2 * (D * (self.problem.L ** 2)) + (
                        self.problem.A_const * (24 * (self.problem.n ** 2) + 23 * self.problem.n - 1)))
            gamma_1_max = min(self.problem.mu / (4 * (self.problem.L_max ** 2)),
                              1.0 / (3 * np.sqrt(self.problem.n * (self.problem.n - 1)) * self.problem.L_max), (np.sqrt(
                    (self.problem.L_max ** 4) + (self.problem.n * self.problem.mu * Omega / 4)) - (
                                                                                                                            self.problem.L_max ** 2)) / Omega)
            k_star = self.n_iter // (2*self.problem.n) #16//((self.problem.n**3)*(gamma_1_max**2)*(self.problem.mu**2))
            if (self.num_perms <= k_star):
                gamma_1 = gamma_1_max
            else:
                gamma_1 = (2.0) * (2 * self.num_perms + 1) / (
                            self.problem.mu * self.problem.n * ((self.num_perms + 1) ** 2))
            gamma_2 = 4 * gamma_1
        elif (stepsize_rule == "another_rule"):
            # decreasing for
            gamma_2 = 2 / (self.problem.L_max + coef * ((num_perms * self.problem.n) + self.t))
            gamma_1 = 1 / (self.problem.L_max + coef * ((num_perms * self.problem.n) + self.t))
        else:
            print("No stepsize rule specified!")
        return (gamma_1, gamma_2)

    def SEG(self, gamma_1, gamma_2, x0, n_iter=1000, trials=1, rr=False, so=False, ig=False,
            stepsize_rule=False, coef=0.7):

        args = [gamma_1, gamma_2, x0, n_iter, trials, rr, so, ig,
                stepsize_rule, coef]
        if (args[4] > 1):
            # Start multiple processes
            pool = multiprocessing.Pool(processes=args[4])

            # Execute the function in parallel using the pool of processes
            inputs = args[4] * [args]
            results = pool.starmap(self.SEG_implementation, inputs)
            # print(results)

            # Compute the output as the average of all trials
            ouput_final_point = [v[0] for v in results]
            ouput_final_point = np.mean(ouput_final_point, axis=0)
            output_error = [v[1] for v in results]
            output_error = np.mean(output_error, axis=0)
        else:
            ouput_final_point, output_error = self.SEG_implementation(*args)
        return (ouput_final_point, output_error)

    def SEG_implementation(self, gamma_1, gamma_2, x0, n_iter, trials, rr, so, ig, stepsize_rule, coef):
        relative_error = [1]
        x = x0
        if (so):
            perm = np.random.permutation(self.problem.n)
        self.num_perms = 0
        self.n_iter = n_iter
        for t in range(n_iter):
            self.t = t
            if rr and (t % self.problem.n == 0):
                perm = np.random.permutation(self.problem.n)
                self.num_perms += 1
                if (self.num_perms % 10000 == 0):
                    print("reached", self.num_perms)
            # choose sample to use for computing gradient
            if (rr or so):
                index_sample = perm[t % self.problem.n]
            elif (ig):
                index_sample = self.t
            else:
                index_sample = np.random.randint(self.problem.n)
            # lr = L_quad / np.sqrt(n)
            if (stepsize_rule):
                gamma_1, gamma_2 = self.stepsize_scheduler(stepsize_rule)

            # Added for experiments with stepsize
            # if (stepsize_rule == "decreasing"):
            #     if(self.num_perms <= self.n_iter / (2*self.problem.n)):
            #         gamma_2 = 5e-3 #gamma_2_max
            #     else:
            #         gamma_2 = 1.0 / ((self.num_perms + 1))
            #         print(gamma_2)
            #     gamma_1 = gamma_2 / 5#gamma_2 / self.problem.n
            # elif(stepsize_rule == "EG_rr_decreasing"):
            #     if (self.num_perms <= self.n_iter / (2 * self.problem.n)):
            #         gamma_2 = 5e-3  # gamma_2_max
            #     else:
            #         gamma_2 = (4.0) * (2 * self.num_perms + 1) / (self.problem.mu * ((self.num_perms + 1) ** 2))
            #     gamma_1 = 1e-3  # gamma_2 / self.problem.n
            # elif(stepsize_rule == "decreasing2"):
            #     if (self.num_perms <= self.n_iter / (2 * self.problem.n)):
            #         gamma_2 = 5e-3  # gamma_2_max
            #     else:
            #         gamma_2 = 1.0 / (((self.num_perms + 1) ** 2))
            #     gamma_1 = 1e-3  # gamma_2 / self.problem.n
            # Update of SEG
            x_extrap = x - gamma_2 * self.problem.operator(x, index=index_sample)
            x = x - gamma_1 * self.problem.operator(x_extrap, index=index_sample)
            # compute error
            relative_error.append((np.linalg.norm(x - self.problem.sol) / np.linalg.norm(x0 - self.problem.sol)) ** 2)
        # print("Number of permutations", self.num_perms)
        return ([x, relative_error])


def plot(var, y_label=" ", x_label="Iterations", title=""):
    # Plot results
    plt.plot(np.arange(np.shape(var)[0]), var, marker="o", linestyle='-', linewidth=2)

    plt.yscale('log')
    plt.grid(True)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_multiple_var(multiple_vars, labels, y_label=" ", x_label="Iterations", title=""):
    # Plot results
    markers = itertools.cycle(('o', 'P', '>', 5, '*'))
    for i in range(len(multiple_vars)):
        plt.plot(np.arange(np.shape(multiple_vars[i])[0]), multiple_vars[i], marker=next(markers), label=labels[i])

    plt.yscale('log')
    plt.grid(True)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.show()