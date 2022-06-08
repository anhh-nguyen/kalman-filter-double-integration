import numpy as np
from time import time
import random
from math import pi,exp,sqrt
import warnings
import matplotlib.pyplot as plt

def kalman_xy(x, P, measurement, R,
              motion = np.matrix('0. 0. 0. 0.').T,
              Q = np.matrix(np.eye(4))):

    '''
    See http://en.wikipedia.org/wiki/Kalman_filter    
    x: initial state contains 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    '''
    return kalman(x, P, measurement, R, motion, Q,
                  F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      '''),
                  H = np.matrix('''
                      1. 0. 0. 0.;
                      0. 1. 0. 0.'''))

def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: the covariance of the observation noise (same shape as H)
    motion: external motion added to state vector x
    Q:  the covariance of the process noise, motion noise (same shape as P)
    F: the state-transition model, next state function: x_prime = F*x
    H: the observation model, measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)
    '''
    # UPDATE x, P based on measurement m    
    y = np.matrix(measurement).T - H * x #measurement pre-fit residual
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K * y          # Updated (a posteriori) state estimate
    P = (np.matrix(np.eye(F.shape[0])) - K*H)*P      # Updated (a posteriori) estimate covariance

    # PREDICT x, P based on motion
    x = F*x + motion     # Predicted (a priori) state estimate
    P = F*P*F.T + Q      # Predicted (a priori) estimate covariance

    return x, P

# gaussian function
def f(mu, sigma2, x):
    ''' f takes in a mean and squared variance, and an input x
       and returns the gaussian value.'''
    coefficient = 1.0 / sqrt(2.0 * pi *sigma2)
    exponential = exp(-0.5 * (x-mu) ** 2 / sigma2)
    return coefficient * exponential



def demo_kalman_xy():
    x = np.matrix('0. 0. 0. 0.').T 
    P = np.matrix(np.eye(4))*1000 # initial uncertainty

    N = 100
    true_x = np.linspace(0.0, 10.0, N)
    true_y = true_x**2
    observed_x = true_x + 0.05*np.random.random(N)*true_x
    observed_y = true_y + 0.05*np.random.random(N)*true_y
    ############
    plot1 = plt.figure(1)
    plt.plot(true_x)
    plt.plot(observed_x, 'r')
    result = []
    xestimators = []
    yestimator = []
    xdotestimator = []
    R = 0.01**2
    for meas in zip(true_x, true_y):
        x, P = kalman_xy(x, P, meas, R)
        result.append((x[:2]).tolist())
        xestimators.append(x[0].tolist())
        yestimator.append(x[1].tolist())
    kalman_x, kalman_y = zip(*result)
    for i in range(len(xestimators)):
        xestimators[i] = xestimators[i][0][0]
        yestimator[i] = yestimator[i][0][0]
        #xdotestimator[i] = xdotestimator[i][0][0]

    xestimators = xestimators + 0.01*np.random.random(N)*true_x
    yestimator = yestimator + 0.008*np.random.random(N)*true_y

    plt.plot(xestimators, 'g-')
    ########
    plot2 = plt.figure(2)
    plt.plot(true_y)
    plt.plot(observed_y, 'r')
    plt.plot(yestimator, 'g-')
    ########
    #plot3 = plt.figure(3)
    #plt.plot(xdotestimator, 'r')
    plt.show()
    print(xdotestimator)

demo_kalman_xy()