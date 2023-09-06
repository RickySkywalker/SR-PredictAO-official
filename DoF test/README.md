This part contains the code for the experiments mentioned in Appendix-B, where we study the relationship between the NDT-pruning method and the degree of freedom of our model.

## Requirements
- Python 3.7
- Pytorch 2.0.1
- scikit-learn 1.2.2
- numpy 1.24.3

## Structure of Code
- deep_neural_regression_tree.py and deep_neural_regression_forest.py: the configuration of a regression model which is modified from our enhancement module
- dof.py: the code for estimating the degree of freedom (dof) of the regression model

## Usage
1. Install required envs
2. Run the code by executing 'python dof.py {n} {p} {s} {sim_type} {DNRT_depth} {DNRT_dropout} {num_trails}', where n is the size of the dataset, p denotes the total number of features generated, s(<=p) is the number of features with a nonzero coefficient (thus considered signal), DNRT_depth is the depth of the DNRT, DNRT_dropout is the dropout rate of the NDT-pruning method, and num_trails is the number of trails for the Monte Carlo estimation. sim_type is the type of data-simulating function. Here we provide three different types of data-simulating functions: linear, 'MARSadd', and 'POWERadd', where the latter two are mentioned in the appendix.
3. The estimated dof of our regression model and the linear regression model will be printed out. The results and the parameters will also be saved in a txt file.
