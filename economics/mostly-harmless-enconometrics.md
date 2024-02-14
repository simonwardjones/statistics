# Mostly Harmless Econometrics
## An empiricists companion

> Joshua D. Angrist and Jörn-Steffen Pischke

## Chapter 1 - Questions about Questions
 - What is the causal question of interest?
 - What is the experiment that could ideally be used to capture the causal effect of interest?
 - What is your identification strategy? (approximating an experiment with observational data)
 - What is your mode of statistical inference?

## Chapter 2 - The experimental ideal

### Potential Outcomes Framework

Given treatment $D_i \in {1, 2}$ the potential outcomes of the dependent variable $Y$ is defined as
$$
\text{Potential Outcome} = \begin{cases}
    Y_{1i} & \text{ if $D_i = 1$ }\\
    Y_{0i} & \text{ if $D_i = 0$ }\\
\end{cases}    
$$
The outcome variable $Y_i$ can be written in terms of the potential outcomes as
$$
    Y_i = Y_{0i} + (Y_{1i} - Y_{0i})*D_i
$$

We are often interested in the average causal effect. This can be written as the sum of the average treatment effect on the treated plus the selection bias.

$$
    E[Y_i|D_i = 1] - E[Y_i|D_i=0] = E[Y_{1i}|D_i=1] - E[Y_{0i}|D_i=1] + E[Y_{0i}|D_i=1] - E[Y_{0i}|D_i=0]
$$

In a randomised experiment $Y_{0i}$ and $Y_{1i}$ are independent of $D_i$

### Regression Analysis of experiments

Assuming the treatment effect is the same for everyone $\rho = Y_{1i}-Y_{0i}$ then we can write the outcome variable as

$$
Y_i = \underbrace{\alpha}_{E[Y_{0i}]} 
  + \underbrace{\rho}_{(Y_{1i}-Y_{0i})}D_i 
  + \underbrace{\eta_i}_{(Y_{0i}-E[Y_{0i}])}
$$

## Chapter 3 - Making sense of regression

### Conditional expectation function (CEF)
The conditional expectation function (CEF) for a dependent variable $Y_i$, given a Kx1 vector of covariates $X_i$ (with elements ($x_{ik}$)) is the expectation, or population average of $Y_i$ given $X_i$ held fixed. The CEF is written $E[Y_i|X_i]$ and is a function of $X_i$.

### The law of iterated expectations
The law of iterated expectations states the unconditional average can be written as the unconditional average of the CEF:
$$
    E[Y_i] = E[E[Y_i|X_i]]
$$
Where the outer expectation uses the distribution of $X_i$.

### The CEF decomposition property
$$
    Y_i = E[Y_i|X_i] + \epsilon_i
$$
Where 
 1. $\epsilon_i$ is mean independent of $X_i$ or $E[\epsilon_i|X_i] = 0$
 2. $\epsilon_i$ is uncorrelated with any function of $X_i$

In words this theorem means that $Y_i$ can be written as a piece that is iz explainable by $X_i$ and a piece that is orthogonal or independent of any function of $X_i$.

### The CEF Prediction Property

Let $m(x)$ be any function of $X_i$. The CEF solves
$$
E[Y_i|X_i] = \argmin_{m(X_i)} E[(y - m(X_i))^2]
$$
So it is the MMSE predictor of $Y_i$ given $X_i$

### The ANOVA Theorem

$$
    V(Y_i) = V(E[Y_i|X_i]) + E[V(Y_i|X_i)]
$$

### The anatomy of regression

For the case when we just have one covariate we have 
$$
    y_i = \alpha + \beta x_i + \epsilon_i
$$
And the ols estimator for $\beta$ is given by
$$
\beta
    = \frac{\sum_i^N(x_i-\bar{x})(y_i-\bar{y})}{\sum_i^N(x_i-\bar{x})^2}
    = \frac{Cov(y_i, x_i)}{Var(x_i)}
$$

In the general case with k covariates we have the formula
$$
    \beta = (X'X)^{-1}X'y
$$
However due to "the anatomy of regression"  we can also write 
$$
    \beta_k = \frac{Cov(y_i, \tilde{x}_i^k)}{Var(\tilde{x}_i^k)}
$$
Where $\tilde{x}_i^k$ is the residual obtained by regressing $\tilde{x}_i^k$ on all remaining K-1 independent variables.

---
A further deep dive into the anatomy of regression can be found here: https://journals.sagepub.com/doi/pdf/10.1177/1536867X1301300107


We have the main regression model 
$$
    Y_i = \beta_0 + \beta_1x_1 + ... + \beta_kx_k + ... + \beta_Kx_K + e_i
$$

We define the auxiliary regression for $x_{ki}$ against all remaining independent variables
$$
    x_{ki} = \gamma_0 + \gamma_1x_{1i} + ... + \gamma_{k-1}x_{k-1i} + \gamma_{k+1}x_{k+1i} + ... + \gamma_Kx_{Ki} + f_i 
$$
Then the residual is defined as
$$
\begin{aligned}
\tilde{x}_i^k 
    &=  x_{ki} - \hat{x}_i^k \\
    &=  x_{ki} - (\gamma_0 + \gamma_1x_{1i} + ... + \gamma_{k-1}x_{k-1i} 
                  + \gamma_{k+1}x_{k+1i} + ... + \gamma_Kx_{Ki})\\
    &= f_i
\end{aligned}
$$

If we are looking at the regression
$$
\begin{aligned}
Cov(y_i, \tilde{x}_i^k)
    &= Cov(\beta_0 + \beta_1x_1 + ... + \beta_kx_k + ... + \beta_Kx_K + e_i, \tilde{x}_i^k) \\
    &= Cov(\beta_0 + \beta_1x_1 + ... + \beta_kx_k + ... + \beta_Kx_K + e_i, f_i) \\
    &= Cov(\beta'X + e_i, f_i) \\
    &= E[(\beta'X + e_i)f_i] - E[\beta'X + e_i]E[f_i] \quad\text{from definition of Cov} \\
    &= E[(\beta'X + e_i)f_i] \quad\quad\text{as }E[f_i] = 0 \\
    &= E[\beta'Xf_i] + E[e_if_i] \quad\quad\text{by linearity} \\
    &= E[\beta'Xf_i]  \quad\quad\text{as }E[H(x_{ki})e_i] = 0 \text{ for any function of } x_{ki} \text{which }f_i\text{is.}\\
    &= E[\beta_kx_{ki}f_i] \quad\quad\text{as }E[H(x_{ri})f_i] = 0 \text{ for any function of } x_{ri}\,r\neq k \\
    &= E[\beta_k(\tilde{x}_i^k + \hat{x}_i^k)f_i] \quad\quad\text{expanding }x_{ki}\\
    &= E[\beta_k\tilde{x}_i^kf_i] \quad\quad\text{as }E[H(x_{ri})f_i] = 0 \text{ for any function of } x_{ri}\,r\neq k \\
    &= \beta_kE[(\tilde{x}_i^k)^2] \quad\text{as }f_i=\tilde{x}_i^k\\
    &= \beta_k Var(\tilde{x}_i^k)
\end{aligned}
$$

---
