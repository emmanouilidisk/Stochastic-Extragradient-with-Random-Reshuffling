# Stochastic Extragradient with Random Reshuffling: Improved Convergence for Variational Inequalities

This is the official implementation of the paper "Stochastic Extragradient with Random Reshuffling: Improved Convergence for Variational Inequalities", which was accepted to AISTATS 2024.   
As part of the paper, we provide the Optimizer Module, a library of common optimization algorithms that can be used for solving minimization and minimax problems. 

If you find our code useful consider citing our paper. Also, star-ing the Github Repo is more than welcomed! 🌟
```
citation
```

# Optimizer Module 

<!-- GETTING STARTED -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="logo.jpg" alt="Logo" width="70" height="70">
  </a>
</p>

The Optimizer Module is an efficient and easy-to-customize implementation of common optimization algorithms, allowing you to easily implement your own variant of an optimization method or run experiments in less than 5 lines of code.   
To start using the module and run your experiments, follow the details below.


## Getting Started
To get started make sure you have installed all the prerequisites on your computer and then follow the instructions in the installation section.

### Prerequisites
To compile this implementation you will need:
- [numpy](https://numpy.org/install/)
- [pickle](https://docs.python.org/3/library/pickle.html) library
- [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module
- Python 3.7 or later

You can install the prerequisites with the following command inside the project's folder
```
$ pip install -r requirements.txt
```

## Run your experiments!
Run your own experiments using the `optimizer_module` in less than 5 commands:  
```python
import optimizer_module as op

# initialize the problem
problem = op.problem(problem_type="quadratic", n=100, d=100, mu=0.1, L=10)

# initialize optimizer object  
optimizer = op.optimizer(problem)

# run your favorite optimization algorithm
results = optimizer.SEG(gamma_1=0.001, gamma_2=0.005, x0=np.zeros(200), n_iter=10**6, trials=10, rr=True)
```
Arguments of ```op.problem```:
* problem_type: type of the problem (i.e. quadratic, affine, etc.)
* n: number of data
* d: dimension of data
* mu: strongly convex parameter for strongly monotone functions
* L: Lipschitz parameter for smooth problems  

For plotting the results, you can use the command:  
```python
op.plot(results, y_label = "Relative Error", title = "Strongly Monotone Game")
```

For more details on the currently available commands and optimization algorithms please check [here](https://github.com/emmanouilidisk/Stochastic-ExtraGradient-with-RR/blob/main/docs/supported_opts_algo). 

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be, learn, inspire, and create.  
Contribute following the above steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b new_branch_name`)
3. Commit your Changes (`git commit -m 'Add some extra functionality'`)
4. Push to the Branch (`git push origin new_branch_name`)
5. Open a Pull Request  
