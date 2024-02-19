import numpy as np

class problem:

    def __init__(self, problem_type, n, d, mu=0.1, L=1, A = None, B = None, C=None, a=None, c=None):
        self.problem_type = problem_type
        self.n = n
        self.d = d
        self.mu = mu
        self.L = L
        np.random.seed(1)
        if self.problem_type == "quadratic":
            self.generate_quadratic_problem(A, B, C, a, c)
        elif self.problem_type == "bilinear":
            self.generate_bilinear_problem(B, a, c)
        elif self.problem_type == "convex_concave":
            self.generate_convex_concave_problem(A, C, a, c)
        elif self.problem_type == "c_c-counter_example":
            self.generate_convex_concave_counter_example()
        elif self.problem_type == "weak_minty":
            self.generate_weak_minty_problem()
        elif self.problem_type == "ridge":
            self.generate_ridge_problem()
        else:
            print("Select a type of problem to be generated!")

    def make_sym_matrix(self, mu=None, L=None, num_zeros=0):
        """
        Create a list of symmetric matrices with given minimum and maximum eigenvalues.
        ----
        Input:
            mu (float): Minimum eigenvalue.
            L (float): Maximum eigenvalue.
        Returns:
            symmetric_matrix (list): list of symmetric matrices with eigenvalues ranging from mu to L.
        """
        if mu is None:
            mu = 0.1
        if L is None:
            L = 1

        symmetric_matrix = [[[]] for _ in range(self.n)]
        eigenvalues = [[] for _ in range(self.n)]

        # Generate n random eigenvalues between mu and L
        for i in range(self.n):
            eigenvalues[i] = np.linspace(start=mu, stop=L, num=self.d)

        # Create a random orthogonal matrix with eigenvalues on the diagonal
        Q, _ = np.linalg.qr(np.random.rand(self.d, self.d))
        for i in range(self.n):
            symmetric_matrix[i] = Q @ np.diag(eigenvalues[i]) @ Q.T

        return symmetric_matrix

    # different types of problems
    def generate_quadratic_problem(self, A, B, C, a, c):
        """
        Function generating quadratic problem of the form:
            f(x,y) = (1/2) x^T A x + x^T B y -(1/2) y^T C y + a^T x - c^T y

            Note: each one of A, B, C, a, c are assumed to have
            finite-sum structure, i.e.: A = (1/n) \sum_{i=1}^n A_i
        ---------
        Inputs:
        - A: (n, d, d) numpy matrix
        - B: (n, d, d) numpy matrix
        - C: (n, d, d) numpy matrix
        - a: (n, d) numpy matrix
        - c: (n, d) numpy matrix
        -------
        Returns: an instance of the class "problem"
        """

        if A is None:
            self.A = self.make_sym_matrix(mu=self.mu, L=self.L)
            eig, _ = np.linalg.eig(np.mean(self.A, axis=0))
        else:
            self.A = A

        if B is None:
            self.B = self.make_sym_matrix(mu=0, L=1.0)
        else:
            self.B = B

        if C is None:
            self.C = self.make_sym_matrix(mu=self.mu, L=self.L)
            eig, _ = np.linalg.eig(np.mean(self.C, axis=0))
        else:
            self.C = C

        if a is None:
            self.a = np.random.normal(0, 1, size=(self.n, self.d))
        else:
            self.a = a

        if c is None:
            self.c = np.random.normal(0, 1, size=(self.n, self.d))
        else:
            self.c = c

        sum_matrix_A = np.mean(self.A, axis=0)
        sum_matrix_B = np.mean(self.B, axis=0)
        sum_matrix_C = np.mean(self.C, axis=0)
        self.sum_matrix_a = np.mean(self.a, axis=0)
        self.sum_matrix_c = np.mean(self.c, axis=0)

        self.M = np.block([[sum_matrix_A, sum_matrix_B], [-sum_matrix_B.T, sum_matrix_C]])
        self.sol = np.linalg.solve(self.M, - np.concatenate((self.sum_matrix_a, self.sum_matrix_c)))

        #compute L_max
        L_i_cur = []
        for i in range(self.n):
            M_i = np.block([[self.A[i], self.B[i]], [-self.B[i], self.C[i]]])
            eigenvalues, _ = np.linalg.eig(M_i.T @ M_i)
            L_i_cur.append(np.real(np.sqrt(max(eigenvalues))))
        self.L_max = max(np.real(L_i_cur))

        # computation of values of A,B,C constants in paper's Assumption
        self.A_const = 0
        for i in range(len(L_i_cur)):
            self.A_const += L_i_cur[i] ** 2
        self.A_const = 2.0*self.A_const / self.n

    def generate_bilinear_problem(self, B=None, a=None, c=None):
        """
        Generate bilinear problem:
            f(x,y) = x^T B y + a^T x + c^T y
                where each of B, a, c have finite-sum form, i.e. B = (1/n) \sum_{i=1}^n B_i
        ------
        Inputs:
            - B: (n, d, d) numpy array
            - a: (n, d) numpy array
            - c: (n, d) numpy array
        Returns: instance of class "problem"
        """

        self.A = np.zeros((self.n, self.d, self.d))
        self.C = np.zeros((self.n, self.d, self.d))

        if B is None:
            # self.B = np.zeros(shape=(self.n, self.d, self.d))
            # for i in range(self.d):
            #     self.B[i][i][i] = 1
            self.B = self.make_sym_matrix(mu=self.mu, L=self.L) #here mu is the minimum eigenvalue of B
        else:
            self.B = B

        if a is None:
            self.a = np.random.normal(0, 1./self.n, size=(self.n, self.d))
        else:
            self.a = a

        if c is None:
            self.c = np.random.normal(0, 1./self.n, size=(self.n, self.d))
        else:
            self.c = c

        sum_matrix_A = np.mean(self.A, axis=0)
        sum_matrix_B = np.mean(self.B, axis=0)
        sum_matrix_C = np.mean(self.C, axis=0)
        self.sum_matrix_a = np.mean(self.a, axis=0)
        self.sum_matrix_c = np.mean(self.c, axis=0)

        self.M = np.block([[sum_matrix_A, sum_matrix_B], [-sum_matrix_B.T, sum_matrix_C]])
        eig, _ = np.linalg.eig(self.M.T @ self.M)
        self.L = np.sqrt(np.real(max(eig)))
        self.mu = np.sqrt(np.real(min(eig))) # minimum eigenvalue of B
        self.sol = - np.linalg.inv(self.M) @ np.concatenate((self.sum_matrix_a, - self.sum_matrix_c))

        # compute L_max
        L_i_cur = []
        for i in range(self.n):
            M_i = np.block([[self.A[i], self.B[i]], [-self.B[i], self.C[i]]])
            eigenvalues, _ = np.linalg.eig(M_i.T @ M_i)
            L_i_cur.append(np.real(np.sqrt(max(eigenvalues))))
        self.L_max = max(np.real(L_i_cur))

        # computation of values of A,B,C constants in paper's Assumption
        self.A_const = 0
        for i in range(len(L_i_cur)):
            self.A_const += L_i_cur[i] ** 2
        self.A_const = 2.0*self.A_const / self.n

    def generate_convex_concave_problem(self, A=None, C=None, a=None, c=None):
        """
        Generate convex - concave problem
            f(x,y) = (1/2) x^T A x + x^T B y -(1/2) y^T C y + a^T x - c^T y
                where each of A, C, a, c have finite-sum form, i.e. A = (1/n) \sum_{i=1}^n A_i
        ------
        Inputs:
            - A: (n, d, d) numpy array
            - C: (n, d, d) numpy array
            - a: (n, d) numpy array
            - c: (n, d) numpy array
        Returns: instance of class "problem"
        """

        self.mu = 0 #set minimum eigenvalue (mu = 0) to ensure problem is monotone
        if A is None:
            self.A = self.make_sym_matrix(mu=self.mu, L=self.L)
            eig, _ = np.linalg.eig(np.mean(self.A, axis=0))
        else:
            self.A = A

        self.B = np.zeros((self.n, self.d, self.d))

        if C is None:
            self.C = self.make_sym_matrix(mu=self.mu, L=self.L)
            eig, _ = np.linalg.eig(np.mean(self.C, axis=0))
        else:
            self.C = C

        if a is None:
            self.a = np.random.normal(0, 1, size=(self.n, self.d))
        else:
            self.a = a

        if c is None:
            self.c = np.random.normal(0, 1, size=(self.n, self.d))
        else:
            self.c = c

        sum_matrix_A = np.mean(self.A, axis=0)
        sum_matrix_B = np.mean(self.B, axis=0)
        sum_matrix_C = np.mean(self.C, axis=0)
        self.sum_matrix_a = np.mean(self.a, axis=0)
        self.sum_matrix_c = np.mean(self.c, axis=0)

        self.M = np.block([[sum_matrix_A, sum_matrix_B], [-sum_matrix_B.T, sum_matrix_C]])
        self.sol = np.linalg.solve(self.M, - np.concatenate((self.sum_matrix_a, self.sum_matrix_c)))

        # compute maximum eigenvalue (L_max)
        L_i_cur = []
        for i in range(self.n):
            M_i = np.block([[self.A[i], self.B[i]], [-self.B[i], self.C[i]]])
            eigenvalues, _ = np.linalg.eig(M_i.T @ M_i)
            L_i_cur.append(np.real(np.sqrt(max(eigenvalues))))
        self.L_max = max(np.real(L_i_cur))

        # computation of values of A constant in paper's Assumption
        self.A_const = 0
        for i in range(len(L_i_cur)):
            self.A_const += L_i_cur[i] ** 2
        self.A_const = 2.0*self.A_const / self.n

        # sum_noise = 0
        # for i in range(self.n):
        #     sum_noise += (1.0/self.n) * np.linalg.norm(self.M[i] @ self.sol + np.concatenate((self.a[i], self.c[i])))
        # print("Noise :", sum_noise)
        # further constants for paper experiments
        gamma_1_max = 1 / (3 * np.sqrt(2 * self.n * (self.n - 1)) * self.L_max)
        C = ((50 + 2 * self.n) * self.A_const + 2 * (10 * (self.n ** 2) + 50 * self.n + 2) * (
                    self.L ** 2)) * (gamma_1_max ** 2) + (self.A_const + 2 * (self.L ** 2)) * 4 * (
                        gamma_1_max ** 2)
        self.G = 2 * (1 + (gamma_1_max ** 2) * (self.n ** 2) * (2 * self.L ** 2 + 8 * (gamma_1_max ** 2) * (self.L ** 4) + 3) + 3 * C * (self.L_max ** 2))

    def compute_grad(self, x, sample_indices=None):
        """
        Computes the gradient (for minimization problems) or
        the vector of gradients (\nabla_x f(x,y), - \nabla_y f(x,y)) (for minimax problems).
        ------
        Inputs:
            - x: point at which to compute the gradient(s)
            - sample_indices: list of indices of samples that will be used for computing stochastic gradient.
                     If None, then the full-batch (deterministic) gradient is computed.
        Returns: gradient(s)
        """
        if self.problem_type == "quadratic" or self.problem_type == "convex_concave":
            if sample_indices is None:
                return self.M @ x + np.concatenate((self.sum_matrix_a, self.sum_matrix_c))
            else:
                M_index = np.block([[self.A[sample_indices], self.B[sample_indices]], [-self.B[sample_indices].T, self.C[sample_indices]]])
                grad = (1/self.n) * (M_index @ x + np.concatenate((self.a[sample_indices], self.c[sample_indices])))
                return (grad)
        elif self.problem_type == "bilinear":
            if sample_indices is None:
                return self.M @ x + np.concatenate((self.sum_matrix_a, -self.sum_matrix_c))
            else:
                grad_x = self.B[sample_indices] @ x[self.d:2 * self.d] + self.a[sample_indices]
                grad_y = - self.B[sample_indices].T @ x[:self.d] - self.c[sample_indices]
                grad = np.concatenate((grad_x, grad_y))
                return grad
        else:
            raise TypeError("Unknown problem type!")