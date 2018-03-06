glm_poisson <- function(Y, X){
  model1 <- glm(Y ~ X, family = poisson(link = 'log'))
  coef <- model1$coefficients
  coef <- ifelse(abs(coef) > 5, sign(coef)*2, coef)
  res <- list(coef = coef)
  return(res)
}

glm_poisson_no_inter <- function(Y, X){
  model1 <- glm(Y ~ X - 1, family = poisson(link = 'log'))
  coef <- model1$coefficients
  coef <- ifelse(abs(coef) > 5, sign(coef)*2, coef)
  res <- list(coef = coef)
  return(res)
}

glm_binomial <- function(Y, X, beta){
  if(all(X == 1)){
    res <- list(coef = c(1, 1))
  } else {
    Y_10 <- as.numeric(Y>0)
    X_beta <- X*beta
    model2 <- glm(Y_10 ~ X_beta, family = binomial(link = 'logit'))
    coef <- model2$coefficients
    coef <- ifelse(abs(coef) > 5, sign(coef)*2, coef)
    res <- list(coef = coef)
  }
  return(res)
}

glm_binomial_m <- function(Y, X_beta){
  Y_10 <- as.numeric(Y>0)
  model2 <- glm(Y_10 ~ X_beta, family = binomial(link = 'logit'))
  coef <- model2$coefficients
  coef <- ifelse(abs(coef) > 5, sign(coef)*5, coef)
  res <- list(coef = coef)
  return(res)
}

initialize_coef <- function(Y, X){
  flag <- Y>0
  prop <- length(Y[flag])/length(Y)
  if(prop < 0.5){
     subY <- Y[flag]
     subX <- X[flag]
     model1 <- glm(subY ~ subX, family = poisson(link = 'log'))
  } else {
     model1 <- glm(Y ~ X, family = poisson(link = 'log'))
  }
  coef <- model1$coefficients
  model2 <- lm(log(Y+1)~X)
  coef2 <- model2$coefficients
  if(abs(coef2)[2] < abs(coef)[2]){
    coef <- coef2
  }
  coef <- ifelse(abs(coef) > 5, sign(coef)*2, coef)
  coef[is.na(coef)] <- 0
  res <- list(coef = coef)
  return(res)
}

initialize_coef_no_inter <- function(Y, X){
  flag <- Y>0
  prop <- length(Y[flag])/length(Y)

  if(prop < 0.5){
    subY <- Y[flag]
    subX <- X[flag]
    model1 <- glm(subY ~ subX - 1, family = poisson(link = 'log'))
  } else {
    model1 <- glm(Y ~ X - 1, family = poisson(link = 'log'))
  }
  coef <- model1$coefficients
  model2 <- lm(log(Y+1)~X-1)
  coef2 <- model2$coefficients
  if(abs(coef2) < abs(coef)){
    coef <- coef2
  }
  coef <- model1$coefficients
  coef <- ifelse(abs(coef) > 5, sign(coef)*2, coef)
  coef[is.na(coef)] <- 0
  res <- list(coef = coef)
  return(res)
}

initialize_mu_alpha <- function(Y, X_beta){
  flag <- Y>0
  prop <- length(Y[flag])/length(Y)
  if(mean(abs(X_beta)) > 0.2){
    prob0 <- dpois(0, ceiling(Y))
    pis <- ifelse(prob0 < 0.1, prop, prop/(1-prob0))
    pis[pis>0.95] <- 0.95
    aa <- log(pis/(1-pis))
    coef1 <- lm(aa ~ X_beta)$coefficients
    res <- list(coef = coef1)
  } else{
    alpha <- 1
    count <- ceiling(mean(Y[flag]))
    prob0 <- dpois(0, count)
    pis <- ifelse(prob0 < 0.1, prop, prop/(1-prob0))
    if(pis > 0.95){
      pis <- 0.95
    }
    mu <- log(pis/(1-pis))
    res <- list(coef = c(mu, alpha))
  }
  return(res)
}



initialize_mu_alpha_with_group <- function(Y, X_beta, group){

  id <- as.numeric(as.factor(group))
  if(length(unique(id)) != 2){
    Stop("Use group = NULL instead.")
  }
    flag <- Y>0
    group1_Y <- Y[id==1]
    group2_Y <- Y[id==2]
    prop1 <- length(group1_Y[group1_Y>0])/length(group1_Y)
    prop2 <- length(group2_Y[group2_Y>0])/length(group2_Y)
    prob0 <- dpois(0, mean(group1_Y[group1_Y>0]))
    pis1 <- ifelse(prob0 < 0.1, prop1, prop1/(1-prob0))
    pis1[pis1>0.95] <- 0.95
    aa1 <- log(pis1/(1-pis1))

    prob0 <- dpois(0, mean(group2_Y[group2_Y>0]))
    pis2 <- ifelse(prob0 < 0.1, prop2, prop2/(1-prob0))
    pis2[pis2>0.95] <- 0.95
    aa2 <- log(pis2/(1-pis2))

    aa <- c(aa1, aa2)
    X_beta <- c(0, 1)
    coef1 <- lm(aa ~ X_beta)$coefficients
    res <- list(coef = coef1)

  return(res)
}

initialize_lambda_with_group <- function(Y, group){
  id <- as.numeric(as.factor(group))
  flag <- Y==0
  group1_mean <- mean(Y[id == 1])
  group2_mean <- mean(Y[id == 2])
  lambda <- Y
  lambda[flag & id == 1] <- group1_mean
  lambda[flag & id == 2] <- group2_mean
  res <- list(lambda = lambda)
  return(res)
}

dmix <- function(x){
  dens <- function(x, df) dchisq(x^2, df = df)*2*x
  0.5*dens(x, df = 1) + 0.5*dens(x, df = 2)
}

pmix <- function(x, lower.tail = TRUE){
  p <- integrate(dmix, lower = 0, upper = x)
  if(lower.tail) p$value else 1 - p$value
}



initialize_coef_multiple_beta <- function(Y, X){
  flag <- Y>0
  prop <- length(Y[flag])/length(Y)
  if(prop < 0.5){
    subY <- Y[flag]
    subX <- X[flag, ]
    model1 <- glm(subY ~ subX, family = poisson(link = 'log'))
  } else {
    model1 <- glm(Y ~ X, family = poisson(link = 'log'))
  }
  coef <- model1$coefficients
  model2 <- lm(log(Y+1) ~ X)
  coef2 <- model2$coefficients
  if(abs(coef2)[2] < abs(coef)[2]){
    coef <- coef2
  }
  coef <- ifelse(abs(coef) > 5, sign(coef)*2, coef)
  coef[is.na(coef)] <- 0
  res <- list(coef = coef)
  return(res)
}




initialize_mu_alpha_multiple_beta <- function(Y, X, betas, Z){
  X_beta <- X%*%betas
  flag <- Y > 0
  prop <- length(Y[flag])/length(Y)

  Xbeta_mat <- X
  k <- ncol(X)
  for(i in 1:k){
    Xbeta_mat[, i] <- X[, i]*betas[i]
  }

  if(mean(abs(X_beta)) > 0.2){
    prob0 <- dpois(0, ceiling(Y))
    pis <- ifelse(prob0 > 1 - 10^-10, prop*(1- 10^-10), prop*(1-prob0))
    aa <- log(pis/(1-pis))
    coef1 <- lm(aa ~ Z + Xbeta_mat)$coefficients
    coef1[is.na(coef1)] <- 1
    coef1 <- ifelse(abs(coef1) > 5, sign(coef1)*2, coef1)
    res <- list(coef = coef1)
  } else{
    alphas <- rep(1, k)
    delta <- 0.1
    count <- ceiling(mean(Y[flag]))
    prob0 <- dpois(0, count)
    pis <- ifelse(prob0 < 0.1, prop, prop*(1-prob0))
    if(pis > 0.95){
      pis <- 0.95
    }
    mu <- log(pis/(1-pis))
    res <- list(coef = c(mu, delta, alphas))
  }
  return(res)
}




initialize_mu_alpha_multiple_beta_nowith_alpha <- function(Y, X, Z){

  flag <- Y > 0
  prop <- length(Y[flag])/length(Y)
  prob0 <- dpois(0, ceiling(Y))
  pis <- ifelse(prob0 > 1 - 10^-10, prop*(1- 10^-10), prop*(1-prob0))
  aa <- log(pis/(1-pis))
  coef1 <- lm(aa ~ Z + X)$coefficients
  coef1[is.na(coef1)] <- 1
  coef1 <- ifelse(abs(coef1) > 5, sign(coef1)*2, coef1)
  res <- list(coef = coef1)

  return(res)
}


initialize_mu_alpha_with_group_multiple_beta  <- function(Y, X, betas, Z, group){

  id <- as.numeric(as.factor(group))
  if(length(unique(id)) != 2){
    Stop("Use group = NULL instead.")
  }
  flag <- Y>0
  group1_Y <- Y[id==1]
  group2_Y <- Y[id==2]
  prop1 <- length(group1_Y[group1_Y>0])/length(group1_Y)
  prop2 <- length(group2_Y[group2_Y>0])/length(group2_Y)
  prob0 <- dpois(0, mean(group1_Y[group1_Y>0]))
  pis1 <- ifelse(prob0 < 0.1, prop1, prop1/(1-prob0))
  pis1[pis1>0.95] <- 0.95
  aa1 <- log(pis1/(1-pis1))

  prob0 <- dpois(0, mean(group2_Y[group2_Y>0]))
  pis2 <- ifelse(prob0 < 0.1, prop2, prop2/(1-prob0))
  pis2[pis2>0.95] <- 0.95
  aa2 <- log(pis2/(1-pis2))

  Xbeta_mat <- X
  k <- ncol(X)
  for(i in 1:k){
    Xbeta_mat[, i] <- X[, i]*betas[i]
  }

  aa <- c(rep(aa1, length(group1_Y)), rep(aa2, length(group2_Y)))

  coef1 <- lm(aa ~ Z + Xbeta_mat)$coefficients
  res <- list(coef = coef1)

  return(res)
}


initialize_mu_alpha_with_group_multiple_beta_nowith_alpha <- function(Y, X, Z, group){

  id <- as.numeric(as.factor(group))
  if(length(unique(id)) != 2){
    Stop("Use group = NULL instead.")
  }
  flag <- Y>0
  group1_Y <- Y[id==1]
  group2_Y <- Y[id==2]
  prop1 <- length(group1_Y[group1_Y>0])/length(group1_Y)
  prop2 <- length(group2_Y[group2_Y>0])/length(group2_Y)
  prob0 <- dpois(0, mean(group1_Y[group1_Y>0]))
  pis1 <- ifelse(prob0 < 0.1, prop1, prop1/(1-prob0))
  pis1[pis1>0.95] <- 0.95
  aa1 <- log(pis1/(1-pis1))

  prob0 <- dpois(0, mean(group2_Y[group2_Y>0]))
  pis2 <- ifelse(prob0 < 0.1, prop2, prop2/(1-prob0))
  pis2[pis2>0.95] <- 0.95
  aa2 <- log(pis2/(1-pis2))

  aa <- c(rep(aa1, length(group1_Y)), rep(aa2, length(group2_Y)))

  coef1 <- lm(aa ~ Z + X)$coefficients
  res <- list(coef = coef1)

  return(res)
}
