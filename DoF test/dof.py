import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
from sklearn.datasets import make_regression

import deep_neural_regression_forests
from deep_neural_regression_forests import NeuralRegressionForest

# Define linear model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(LinearRegression, self).__init__()
        self.device = device
        self.linear = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        # Forward pass
        out = self.linear(x)
        return out

# simulate data by a linear model
def sim_xy(n, p, s, snr, rho):
    # Create the covariance matrix
    Sigma = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            Sigma[i,j] = rho**abs(i-j)

    # Generate X matrix
    mean = np.zeros(p)
    X = np.random.multivariate_normal(mean, Sigma, size=n)
    
    beta = np.zeros(p)
    beta[:s] = np.full(s, 1)
    
    var = np.inner(beta, Sigma @ beta)/snr
    cov_matrix = np.identity(n) * var
    mean = np.zeros(n)
    noise = np.random.multivariate_normal(mean, cov_matrix, size=1)
    noise = noise.reshape(n)

    Y = X @ beta + noise
    
    return X, Y, var

# simulate data by a marsadd model
def sim_marsadd(n, p=10, s=5):
    X = np.random.uniform(0, 1, (n, p))
    noise = np.random.normal(0, 1, n)
    Y = []
    for i in range(n):
        marsadd = 0.1 * (np.e ** (4 * X[i][0])) + 4/(1 + np.e ** (-20 * (X[i][1] - 0.5))) + 3 * X[i][2] + 2 * X[i][3] + X[i][4]
        Y.append(marsadd)
    Y_array = np.array(Y) + noise
    var = 1
    return X, Y_array, var

# simulate data by a poweradd model
def sim_poweradd(n, p, s, power = 4):
    X = np.random.normal(0, 1, (n, p))
    noise = np.random.normal(0, 1, n)
    Y = []
    for i in range(n):
        poweradd = 0
        for j in range(s):
            poweradd += X[i][j] ** power
        Y.append(poweradd)
    Y_array = np.array(Y) + noise
    return X, Y_array, 1

# fit the Deep Neural Regression Tree model
def fit_DNRT(n, p, x_tensor, y_tensor, num_epoch, depth, dropout, var):
    model_DNRT = NeuralRegressionForest(num_trees=128, 
                                        num_features=p, 
                                        used_features_rate=0.5, 
                                        num_classes=1, 
                                        depth=depth, 
                                        dropout_rate=dropout, 
                                        device=torch.device("cuda:0"))
    model_DNRT = model_DNRT.to(torch.device("cuda:0"))
    optimizer = torch.optim.Adam([
        {"params": [param for name, param in model_DNRT.named_parameters() if 'pi' not in name], "lr": 0.5},
        {"params": [model_DNRT.ensemble[i].pi for i in range(len(model_DNRT.ensemble))], "lr": 0.5}
    ])
    criterion = nn.MSELoss()
    
    best_model = model_DNRT
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(num_epoch):

        output = model_DNRT(x_tensor)
        loss = criterion(output.reshape(y_tensor.shape[0]), y_tensor)
        # get the best model
        if loss < best_loss:
            best_Yhat = output
            best_model = model_DNRT
            best_loss = loss
            best_epoch = epoch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'epoch: {epoch}, loss: {loss}')
    print(f'best DNRT model: epoch {best_epoch}, loss: {best_loss}')
    
    yhat_array = best_model(x_tensor).reshape(n).to('cpu').detach().numpy()
    y_array = y_tensor.to('cpu').detach().numpy()
    dof = (np.inner(yhat_array, y_array)/n - yhat_array.mean() * y_array.mean())/var
    
    return dof

# fit the linear model
def fit_linear(n, p, x_tensor, y_tensor, num_epoch, var):
    model_linear = LinearRegression(input_dim=p, output_dim=1)
    optimizer = torch.optim.SGD(model_linear.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    best_model = model_linear
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(num_epoch):

        output = model_linear(x_tensor)
        loss = criterion(output.reshape(y_tensor.shape[0]), y_tensor)
        # get the best model
        if loss < best_loss:
            best_Yhat = output
            best_model = model_linear
            best_loss = loss
            best_epoch = epoch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if epoch % 10 == 0:
        #     print(f'epoch: {epoch}, loss: {loss}')

    # print(f'epoch: {epoch}, loss: {loss}')
    print(f'best linear model: epoch {best_epoch}, loss: {best_loss}')
    
    yhat_array = best_model(x_tensor).reshape(n).to('cpu').detach().numpy()
    y_array = y_tensor.to('cpu').detach().numpy()
    dof = (np.inner(yhat_array, y_array)/n - yhat_array.mean() * y_array.mean())/var
    
    return dof

# repeat the experiments for num_trails times
def dof_pipeline(n, p, s, sim_type, 
                 DNRT_depth, DNRT_dropout,
                 num_trails = 500, snr =3.52, rho=0.35):
    
    DNRT_dof_list = []
    linear_dof_list = []
    
    for trail in range(num_trails):
        show_trail = trail + 1
        print(f'trail {show_trail} start')
        if sim_type == 'linear':
            X, Y, var = sim_xy(n, p, s, snr, rho)
            DNRT_epoch = 200
            linear_epoch = 300
        elif sim_type == 'marsadd':
            X, Y, var = sim_marsadd(n, p, s)
            DNRT_epoch = 300
            linear_epoch = 300            
        elif sim_type == 'poweradd':
            X, Y, var = sim_poweradd(n, p, s)
            DNRT_epoch = 500
            linear_epoch = 300            
        x_tensor = torch.tensor(X).to(torch.float32)
        x_tensor = x_tensor.to('cuda:0')
        y_tensor = torch.tensor(Y).to(torch.float32)
        y_tensor = y_tensor.to('cuda:0')
        DNRT_dof = fit_DNRT(n, p, x_tensor, y_tensor, DNRT_epoch, DNRT_depth, DNRT_dropout, var)
        DNRT_dof_list.append(DNRT_dof)
        
        x_tensor = x_tensor.to('cpu')
        y_tensor = y_tensor.to('cpu')
        linear_dof = fit_linear(n, p, x_tensor, y_tensor, linear_epoch, var)
        linear_dof_list.append(linear_dof)
    
    DNRT_dof_mean = sum(DNRT_dof_list) / len(DNRT_dof_list)
    linear_dof_mean = sum(linear_dof_list) / len(linear_dof_list)
    print(f'DNRT_dof: {DNRT_dof_mean}, linear_dof: {linear_dof_mean}')
    
    return DNRT_dof_mean, linear_dof_mean

if __name__ == "__main__":
    # python dof.py <n> <p> <s> <sim_type> <DNRT_depth> <DNRT_dropout> <num_trails>
    n = int(sys.argv[1])
    p = int(sys.argv[2])
    s = int(sys.argv[3])
    sim_type = sys.argv[4]
    DNRT_depth = int(sys.argv[5])
    DNRT_dropout = float(sys.argv[6])
    num_trails = int(sys.argv[7])
    
    DNRT_dof, linear_dof = dof_pipeline(n, p, s, sim_type, DNRT_depth, DNRT_dropout, num_trails)
    print('estimation finished')
    print(f'the DNRT dof is {DNRT_dof}, the linear dof is {linear_dof}')
    
    # save the result in a txt file
    result_str = f'{sim_type},{n},{p},{s},{DNRT_depth},{DNRT_dropout},{DNRT_dof},{linear_dof}'
    with open(sim_type + '.txt', 'a') as f:
        f.write(result_str + "\n")