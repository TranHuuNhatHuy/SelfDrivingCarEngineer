# KALMAN FILTERS

## I. Gaussian distribution in Kalman filters

### 1. Intro

The Gaussian equation has the form of:

![alt text](image.png)

In which:
- $\mu$ being the mean
- $\sigma^2$ being the variance

A preferred Kalman should be as certain as possible, which has **lowest variance**.

#### a. Measurement and Motion

The Kalman Filter represents our distributions by Gaussians and iterates on two main cycles:
1. Measurement update: requires a product, uses Bayes rule.
2. Motion update: involves a convolution, uses total probability.

#### b. Parameter update

After an update, parameter equations are:

![alt text](image-1.png)

In which:
- $\mu'$ being the new mean
- ${\sigma^2}'$ being the new variance
- $r^2$ being the measurement variance
- $\nu$ being the measurement mean

### 2. Kalman Filter design

#### a. Definitions

- $x$ : estimate
- $P$ : uncertainty covariance
- $F$ : state transition matrix
- $u$ : motion vector
- $z$ : measurement
- $H$ : measurement function
- $R$ : measurement noise
- $I$ : identity matrix

#### b. Prediction equations

![alt text](image-2.png)

#### c. Measurement equations

![alt text](image-3.png)

