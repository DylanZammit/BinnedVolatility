# Histogram-type prior for volatility estimation

This repo implements the method find in "Nonparametric Bayesian Estimation of a Holder continuous diffusion coefficient"
by Gugusvhili et. al. They take a histogram-type prior, by partitioning the domain into bins <img src="https://render.githubusercontent.com/render/math?math=B_k">, and placing a prior distribution on the space of piecewise constant functions of the form

<img src="https://render.githubusercontent.com/render/math?math=\sigma^2(t)=\sum_{k=1}^N\theta_k I_{B_k}(t).">

This is done by placing an inverse gamma prior on the coefficients <img src="https://render.githubusercontent.com/render/math?math=\theta_k">.

## Results
We consider daily Apple stock data (AAPL) and consider the log returns. Below we plot the resultant estimate of the
volatility function along with the 95% confidence band in blue. Superimposed in orange is the kernel estimate using the
boxcar estimate <img src="https://render.githubusercontent.com/render/math?math=K_\epsilon(x, x_i)=I_\{|x-x_i|<\epsilon\}(x)">, so that the frequentist estimator is given by

<img src="https://render.githubusercontent.com/render/math?math=\sigma^2(t)=\sum_{i=1}^nK(t, x_i)Y_i^2,">

where <img src="https://render.githubusercontent.com/render/math?math=Y_i"> are the log return increments.
![vol](https://github.com/DylanZammit/BinnedVolatility/blob/master/images/AAPL_volatility.png)
