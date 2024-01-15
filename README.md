# Stochastic ExtraGradient with Random Reshuffling: Improved Convergence for Variational Inequalities

This is the official implementation of the paper "Stochastic Extragradient with Random Reshuffling: Improved Convergence for Variational Inequalities" and the ``optimizer_module`` associated with it. To reproduce the experiments of the paper, clone the project and run the files in the [experiments]() folder.      

If you found our code useful consider citing our paper. Also, star-ing the Github Repo is more than welcomed!
```
citation
```

# Optimizer Module 

As part of the paper, we provide the Optimizer's module that is an efficient and easy-to-customize implementation of common optimization algorithms. To use the module and run your experiments follow the details in the sections below.  

<!-- GETTING STARTED -->
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
```
import optimizer_module as op

# initialize the problem and optimizer object
problem = op.problem(problem_type="quadratic", n=100, d=100, mu=0.1, L=10)

# initialize optimizer object  
optimizer = op.optimizer(problem)

# run your favorite optimization algorithm (e.g.SGD, SEG, SGDA, etc.)
results = optimizer.SEG(gamma_1=0.001, gamma_2=0.005, x0=np.zeros(200), n_iter=10**6, trials=10, rr=True)
```

For plotting the results, you can use the command:  
```
op.plot(results, y_label = "Relative Error", title = "Strongly Monotone Game")
```

or for plotting multiple variables you can do:
```
op.plot_multiple_var([var1, var2], labels = ["Constant Stepsize", "Decreasing stepsize"], x_label="Iterations", title = "Strongly Monotone Game")
```
## Currently supported
For more details on the available optimization algorithms please check [here](https://github.com/emmanouilidisk/Stochastic-ExtraGradient-with-RR/blob/main/docs/supported_opts_algo). 



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be, learn, inspire, and create.  
Contribute following the above steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b new_branch_name`)
3. Commit your Changes (`git commit -m 'Add some extra functionality'`)
4. Push to the Branch (`git push origin new_branch_name`)
5. Open a Pull Request  
