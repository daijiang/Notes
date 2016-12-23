
## Chapter 2 General linear model

### Classic inference for the traditional linear statistical model

$$y = X\beta + \mathbf{e}$$
where y is a $T \times 1$ vector of observed values of the random variable y on a sample space defined on the Euclidean line. X is a know $T \times K$ nonstochastic design matrix of rank $K$, $\beta$ is a K-dimensional fixed vector of unknown paramters and $\mathbf{e}$ is a $T \times 1$ vector of unobservable random variables with mean vector $E[\mathbf{e}]=0$ and finite covariane matrix. Assume $\mathbf{e}$ to be iid, then the covariance matrix is $E[ee']=\sigma^2I_T$.

The assumption that X is a matrix of fixed variables means that for all the possible situations that might take place **under repeated sampling**, the matrix X would take on the same values.

#### The least squars rules
The observed values of the random variables y contain **all** pf the information available about the unknown $\beta$ vector and the unknown scalar $\sigma^2$. Consequently, the point estimation problem is one of finding a suitable function of the observed values of the random variables y, given the design matrix X, that will yield, **in a repeated sample sense**, the "best" estimator of the unknown parameters. If, when estimate $\beta$, we restrict to the class of rules that are linear functions of y, we may write the general linear estimator as 
$$b_0 = Ay$$
where A is some $K \times T$ matrix. This means that the estimation problem is reduced to determining a matrix A that "appropriately" **summarizes the information contained in y**. Because $b_0$ is a linear function of the random vector y, it is also a random vector.

*Sampling theory approach*:
1. Select a criterion;
2. determine the matrix A
3. evaluate the sampling performance of the estimator: **unbiasedness, efficiency, invariance**.

The least squares **point estimator of $\beta$** is 
$$b = (X'X)^{-1}X'y$$

$$E[b]=E[(X'X)^{-1}X'y]=\beta + E[(X'X)^{-1}X'e]=\beta : Unbiased$$

Precision (variance-covariance) matrix:
$$\Sigma_b = E[(b-E(b))(b-E(b))']=\sigma^2(X'X)^{-1}$$

The Gauss-Markov theorem provides proof that out of the *class of linear unbiased rules*, the least squares estimator is *the best* in terms of minimum variance. In another word, the least squares estimator is equal to or better in terms of sampling precision than all others in its class.

**Estimator of $\sigma^2$**
From above, we know that $E[ee']=\sigma^2I_T$ and $E[e'e]=T\sigma^2$. Thus, if we have a sample of obervations for the random error vector $y-X\beta=e$, we would use this sample as a basis for estimation of $\sigma^2$. Unfortunately, since $\beta$ is **unknown and unobservable the random vector $e$ is unobservable**. Thus, the estimation will involve the observed random y vector and its least squares $\hat{y}=Xb$. The vector of least squares residuals $\hat{e}=y=Xb$ provides a **least squares analogue** of the vector of unobservable error $e$.

$$E[\hat{e}'\hat{e}]=E[e'(I_T - X(X'X)^{-1}X')e]=\sigma^2(T-K)$$

$$\hat{\sigma}^2 = E[\frac{\hat{e}'\hat{e}}{T-K}] = \sigma^2 : unbiased$$

$$\hat{\sigma}^2 = \frac{(y-Xb)'(y-Xb)}{T-K} = \frac{y'(I_T-X(X'X)^{-1}X')y}{T-K}$$

#### The Maximum likelihood rule

Under the maximum likelihood criterion, the parameter estimates are chosen so as to maximize the probability of generating or obtaining the observed sample.

$$L(\beta,\sigma | \mathbf{y}) = \frac{1}{(2\pi\sigma^{2})^{n/2}} \exp\left(\frac{-(y-X\beta)'(y-X\beta)}{2\sigma^2}\right)$$

$$\tilde{\beta} = (X'X)^{-1}X'y=b : same\ as\ least\ squres$$

$$\tilde{\sigma}^2=\frac{(y-X\tilde{\beta})'(y-X\tilde{\beta})}{T}$$

The point estimate $\tilde{\beta}$ is a best linear unbiased estimator, same as that from least squares rule.

$$E[\tilde{\sigma}^2] = \sigma^2\frac{(T-K)}{T}$$ and thus is biased in finite samples. However, the estimator $[T/(T-K)]\tilde{\sigma}^2 = \hat{\sigma}^2$ is an unbiased estimator. 

The term $(T-K)\hat{\sigma}^2/\sigma^2$ is distributed as a chi-square random variable with $(T_K)$ degrees of freedom, i.e $(T-K)\hat{\sigma}^2/\sigma^2 \sim \chi^2_{(T_K)}$.

$$E[\hat{\sigma}^{2}]=\frac{\sigma^{2}}{T-K}E[\chi_{(T-K)}^{2}]=\sigma^{2}$$

$$E[(\hat{\sigma}^{2}-E([\hat{\sigma}^{2}])^{2}]=E[(\hat{\sigma}^{2}-\sigma^{2})^{2}]=\frac{\sigma^{4}}{(T-K)^{2}}Var(\chi_{(T-K)}^{2})=2\sigma^{4}/(T-K)$$

$$E[\tilde{\sigma}^{2}]=\frac{\sigma^{2}}{T}E[\chi_{(T-K)}^{2}]=\frac{T-K}{T}\sigma^{2}$$

$$E[(\tilde{\sigma}^{2}-E([\tilde{\sigma}^{2}])^{2}]=\frac{\sigma^{4}}{T^{2}}Var(\chi_{(T-K)}^{2})=2\sigma^{4}(T-K)/T^2$$

Thus $\tilde{\sigma}^{2}$ has a bias $-(K/T)\sigma^2$ but its variance in finite samples is smaller than the variance for the unbiased estimator $\hat{\sigma}^{2}$, i.e.  there is a **trade-off between bias and precision**. 

To get a minimum mean squared error (MSE) estimator of $\sigma^2$, that is a esimator $\bar{\sigma}^2$ where
$$E[(\bar{\sigma}^2 - \sigma^2)^2]=variance(\bar{\sigma}^2) + bias(\bar{\sigma}^2)^2$$ is minimum. It turns out 
$$\bar{\sigma}^2 = \hat{\sigma}^2\frac{T-K}{T-K+2}$$
Its bias is $\sigma^{2}(T-K)/(T-K+2)$ and variance is $2\sigma^{4}(T-K)/(T-K+2)^{2}$.


```julia

```
