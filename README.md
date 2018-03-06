
### Installation

**scPoissonGamma** relies on the following R packages: **Rcpp**, **RcppArmadillo**. All packagess are hosted on CRAN. 
  ```R
  install.packages ("Rcpp")
  install.packages ("RcppArmadillo")
  ```

**scPoissonGamma** can be installed from github directly as follows:

  ```R
  install.packages ("devtools")
  library(devtools)
  install_github("ChenMengjie/scPoissonGamma")
  ```
  
  
### Simulation example 1: Model in Section 1

```R
allY <- NULL
mu2 <- 2
beta <- 0
mu <- -0.25
alpha <- 2
n <- 200

for(kk in 1:100){

  set.seed(kk)
  X <- c(rep(0, n/2), rep(1, n/2))
  Xbeta <- X*beta
  E <- rgamma(n, 5, 5)
  lambda <- exp(Xbeta + mu2)*E
  pi <- 1- 1/(exp(mu + alpha*Xbeta) + 1)

  ind <- NULL
  for(i in 1:n){
    ind <- c(ind, sample(c(1, 0), 1, prob = c(pi[i], 1 - pi[i])))
  }

  Y <- rep(0, n)
  for(i in 1:n){
    if(ind[i] == 1){
      Y[i] <- rpois(1, lambda[i])
    }
  }

  allY <- rbind(allY, Y)
}

library(scPoissonGamma)
n <- 200
all1 <- NULL
for(kk in 1:nrow(allY)){
  Y <- allY[kk, 1:n]
  X <- c(rep(0, n/2), rep(1, n/2))
  est <- try(PoissonGamma_Mix(Y, X, psi = 10, gamma = 0.25, steps = 30, EM_steps = 10, LRT = FALSE, group = TRUE, down = 0.05, ReportAll = FALSE))
  if(class(est) != "try-error"){
    all1 <- rbind(all1, unlist(est))
  }
}
  ```
 **Y** is the data matrix. *X* is the phenotype vector. *gamma* and *down* are the line search paramters in the gradient descent algorithm. *steps* is the number of steps used in gradient descent algorithm. When *ReportAll* set to be TRUE, estimates and likelihood at each EM step will be output.  
 
 ### Simulation example 2: Model in Section 5, add a sample specific parameter through Z
 
```R
allY <- NULL
mu2 <- 2
beta <- 0
mu <- 1
alpha <- 1
n <- 1000
X <- c(rep(0, n/2), rep(1, n/2))

Z <- rep(1:5, 200)
delta <- 0.5

for(kk in 1:200){

  set.seed(kk)
  Xbeta <- X*beta
  E <- rgamma(n, 5, 5)
  lambda <- exp(Xbeta + mu2)*E

  alphaXbeta <- Xbeta*alpha
  pi <- 1- 1/(exp(mu + alphaXbeta + delta*Z) + 1)

  ind <- NULL
  for(i in 1:n){
    ind <- c(ind, sample(c(1, 0), 1, prob = c(pi[i], 1 - pi[i])))
  }

  Y <- rep(0, n)
  for(i in 1:n){
    if(ind[i] == 1){
      Y[i] <- rpois(1, lambda[i])
    }
  }

  allY <- rbind(allY, Y)
}


all2 <- NULL
system.time(
for(kk in 1:nrow(allY)){
  Y <- allY[kk, ]
  est <- try(PoissonGamma_Mix_multiple_beta(Y, as.matrix(X), Z, psi = 10, gamma = 0.6, steps = 20, EM_steps = 8, down = 0.05, group = FALSE, ReportAll = FALSE))
  if(class(est) != "try-error"){
    all2 <- rbind(all2, unlist(est))
  }
}
)
```
This is a general function that can take *X* of any dimension. 

### Simulation example 3: Model in Section 6, drop-out rate is independent of beta

 ```R
 allY <- NULL
mu2 <- 2
beta <- 0
mu <- 1
alpha <- 1
n <- 1000
X <- c(rep(0, n/2), rep(1, n/2))

Z <- rep(1:5, 200)
delta <- 0.5

for(kk in 1:200){

  set.seed(kk)
  E <- rgamma(n, 5, 5)
  lambda <- exp(X*beta + mu2)*E

  alphaX <- X*alpha
  pi <- 1- 1/(exp(mu + alphaX + delta*Z) + 1)

  ind <- NULL
  for(i in 1:n){
    ind <- c(ind, sample(c(1, 0), 1, prob = c(pi[i], 1 - pi[i])))
  }

  Y <- rep(0, n)
  for(i in 1:n){
    if(ind[i] == 1){
      Y[i] <- rpois(1, lambda[i])
    }
  }

  allY <- rbind(allY, Y)
}


all3 <- NULL
system.time(
for(kk in 1:nrow(allY)){
  Y <- allY[kk, ]
  est <- try(PoissonGamma_Mix_multiple_beta_nowith_alpha(Y, as.matrix(X), Z, psi = 10, gamma = 0.6, steps = 20, EM_steps = 8, down = 0.05, group = FALSE, ReportAll = FALSE))
  if(class(est) != "try-error"){
    all3 <- rbind(all3, unlist(est))
  }
}
)
 ```
 
  
### Author

**Mengjie Chen** (UChicago)
