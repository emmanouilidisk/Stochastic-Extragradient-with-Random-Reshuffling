# Stochastic ExtraGradient with Random Reshuffling

//<p align="center">
//<img src="https://github.com/emmanouilidisk/RSA_Algorithm/blob/main/Demonstration.gif" align="center" width="705" height="380" />
//</p>

<!-- ABOUT THE PROJECT -->
## About The Project
This is the official implementation of the paper "Stochastic Extragradient with Random Reshuffling". In order to use our code, please cite our paper.  

<!-- GETTING STARTED -->
## Getting Started
To get started make sure you have installed all the prerequisites in your computer and then follow the instuctions in the installation section.

### Prerequisites
To compile this implementation you will need:
- [numpy]([https://cmake.org/download/](https://numpy.org/install/))
- [pickle]([https://www.boost.org/users/download/](https://docs.python.org/3/library/pickle.html)) library
- [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module
- Python 3.7 or later

### Running the Experiments of the paper
To reproduce the experiments of the paper "Stochastic Extragradient with Random Reshuffling" clone the project and run the files in the "experiments" folder.  

### Optimizer Module 
Run your own experiments using the `optimizer_module` by typing just 3 commands:  
```
import optimizer_module as op

# initialize the problem, e.g. quadratic
problem = op.problem(problem_type="quadratic", n=100, d=100, mu=0.1, L=0.6)

# initialize optimizer object for this specific problem
optimizer = op.optimizer(problem)

# run your favorite optimization Algorithm (e.g. SEG)
results = optimizer.SEG(gamma_1=0.001, gamma_2=0.005, x0=np.zeros(200), n_iter=10**6, trials=10, rr=True)
```

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be, learn, inspire, and create.  
Contribute following the above steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b new_branch_name`)
3. Commit your Changes (`git commit -m 'Add some extra functionality'`)
4. Push to the Branch (`git push origin new_branch_name`)
5. Open a Pull Request  
