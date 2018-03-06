calculate_weights <- function(z, X){

	require(quadprog)

	xx <- X[, 1]
	remain <- X[, -1]
	k <- ncol(remain)
	n <- length(xx)

	w <- z - xx
	Y <- apply(remain, 2, function(ll){ ll - xx })


	D <- matrix(0, ncol = k, nrow = k)
	for(i in 1:n){
		D <- D + Y[i, ]%*%t(Y[i, ])
	}

	d <- rep(0, k)
	for(i in 1:n){
		d <- d + w[i]*t(Y[i, ])
	}

	A <- rbind(diag(k), - diag(k))
	b <- c(rep(1, k), rep(0, k))

	res <- solve.QP(D, d, -t(A), -b)$solution

	coef <- c(1-sum(res), res)
	return(coef)
}


fitting_lasso <- function(y, X, type = "min"){

	require(glmnet)
	cv.lasso <- cv.glmnet(X, y, intercept = FALSE)

	if(type == "min"){
	  coeff <- as.vector(coef(cv.lasso, s = cv.lasso$lambda.min))
	} else {
    coeff <- as.vector(coef(cv.lasso, s = cv.lasso$lambda.1se))
  }
  selected <- which(coeff!=0)
  res <- list(coeff = coeff[selected], selected = selected - 1)
	return(res)
}


reweighting <- function(prior, Y, Yflag){
	k <- length(Y[-1])
	zero_rate <- length(Yflag[-1][Yflag == 0])/k
	if(zero_rate < 0.9){
	  meanY <- mean(Y[Yflag != 0])
	  pN <- dpois(0, meanY)
	  non.dropout <- (1-zero_rate)*pN/(zero_rate + (1-zero_rate)*pN)
	  zero_vec <- rep(1, k+1)
	  zero_vec[Yflag == 0] <- non.dropout
	  xx <- prior*zero_vec
	  return(xx/sum(xx))
	} else {
	  return(prior/sum(prior))
	}
}


reweighting_sum <- function(prior, Y, Yflag){
  k <- length(Y)
  zero_rate <- length(Yflag[Yflag == 0])/k
  if(zero_rate < 0.9){
    meanY <- mean(Y[Yflag != 0])
    sdY <- sd(Y[Yflag != 0])
    if(is.na(sdY)) sdY <- 1
    pN <- dnorm(0, meanY, sdY)
    non.dropout <- (1-zero_rate)*pN/(zero_rate + (1-zero_rate)*pN)
    zero_vec <- rep(1, k)
    zero_vec[Yflag == 0] <- non.dropout
    xx <- prior*zero_vec
    return(xx/sum(xx))
  } else {
    return(prior/sum(prior))
  }
}


Imputation1 <- function(gene.expression, percentage.cutoff = 0.1, num = 5000, ImputeAll = FALSE){

  xx <- gene.expression # p*n
  p <- nrow(xx)
  n <- ncol(xx)
  zero.rate <- apply(xx, 1, function(x){length(x[x == 0])})/n
  flag <-  zero.rate <= percentage.cutoff

  logxx <- apply(xx, 2, function(y){log(y + 0.1)})
  data <- logxx[round(runif(num)*p), ]
  zero.matrix <- xx != 0
  zero.matrix <- apply(zero.matrix, 2, as.numeric)
  t_logxx <- t(logxx)
  t_zero.matrix <- t(zero.matrix)

  imputed <- logxx
  weights.list <- list(NULL)
  outlier.list <- NULL

  for(j in 1:n){

    remain <- data[, -j]
    res <- fitting_lasso(data[, j], remain)
    coeff <- res$coeff
    selected <- res$selected

    if(length(selected) < 3){

      weights.list[[j]] <- cbind(c(1:n)[-j][selected], rep(1, length(selected)))
      outlier.list <- c(outlier.list, j)

    } else {
      prior.weight <- calculate_weights(logxx[flag, j], logxx[flag, -j][, selected])

      sub.selected <- selected[prior.weight >= 10^-4]
      sub.prior.weight <- prior.weight[prior.weight >= 10^-4]
      sub.prior.weight <- sub.prior.weight/sum(sub.prior.weight)

      weights.list[[j]] <- cbind(c(1:n)[-j][sub.selected], sub.prior.weight)

      Ymat <- t_logxx[-j, ][sub.selected, ]
      Yflagmat <- t_zero.matrix[-j, ][sub.selected, ]

      if(ImputeAll == TRUE){
        tt <- reweighting_sum_C(Ymat, Yflagmat, logxx[, j], zero.matrix[, j], sub.prior.weight, TRUE)
      } else {
        tt <- reweighting_sum_C(Ymat, Yflagmat, logxx[, j], zero.matrix[, j], sub.prior.weight, FALSE)
      }

      imputed[, j] <- tt
    }

  }

  res <- list(weights.list = weights.list, imputed = imputed)
  return(res)
}


Imputation1_star <- function(gene.expression, percentage.cutoff = 0.1, num = 5000){

  xx <- gene.expression # p*n
  p <- nrow(xx)
  n <- ncol(xx)
  zero.rate <- apply(xx, 1, function(x){length(x[x == 0])})/n
  flag <-  zero.rate <= percentage.cutoff

  logxx <- apply(xx, 2, function(y){log(y + 0.1)})
  data <- logxx[round(runif(num)*p), ]
  zero.matrix <- xx != 0

  imputed <- logxx
  weights.list <- list(NULL)
  outlier.list <- NULL
  for(j in 1:n){

    remain <- data[, -j]
    res <- fitting_lasso(data[, j], remain)
    coeff <- res$coeff
    selected <- res$selected

    if(length(selected) < 3){
      weights.list[[j]] <- cbind(c(1:n)[-j][selected], rep(1, length(selected)))
      outlier.list <- c(outlier.list, j)
    } else {
      prior.weight <- calculate_weights(logxx[flag, j], logxx[flag, -j][, selected])

      sub.selected <- selected[prior.weight >= 10^-4]
      sub.prior.weight <- prior.weight[prior.weight >= 10^-4]
      sub.prior.weight <- sub.prior.weight/sum(sub.prior.weight)

      weights.list[[j]] <- cbind(c(1:n)[-j][sub.selected], sub.prior.weight)
      for(i in 1:p){
          if(zero.matrix[i, j] == 0){
            Y <- logxx[i, -j][sub.selected]
            Yflag <- zero.matrix[i, -j][sub.selected]
            new.weights <- reweighting_sum(sub.prior.weight, Y, Yflag)
            imputed[i, j] <- sum(new.weights*Y)
          }
       }
    }

  }

  res <- list(weights.list = weights.list, imputed = imputed)
  return(res)
}



Imputation2 <- function(gene.expression, gc, percentage.cutoff = 0.1, num = 5000, psi = 10, gamma = 0.8, steps = 100, down = 0.1){

  xx <- gene.expression # p*n
  p <- nrow(xx)
  n <- ncol(xx)
  zero.rate <- apply(xx, 1, function(x){length(x[x == 0])})/n
  flag <- which(zero.rate <= percentage.cutoff)
  logxx <- apply(xx, 2, function(y){log(y + 0.1)})
  data <- logxx[round(runif(num)*p), ]

  zero.matrix <- xx != 0
  zero.matrix <- apply(zero.matrix, 2, as.numeric)
  t_xx <- t(xx)
  t_zero.matrix <- t(zero.matrix)

  imputed <- xx
  para.list <- list(NULL)
  selection.list <- list(NULL)
  gc_copy <- rep(0, p)
  gc_copy[!is.na(gc)] <- gc[!is.na(gc)]

  for(j in 1:n){

    remain <- data[, -j]
    res <- fitting_lasso(data[, j], remain)
    coeff <- res$coeff
    selected <- res$selected
    if(length(selected) > 3){

      prior.weight <- calculate_weights(logxx[flag, j], logxx[flag, -j][, selected])
      sub.selected <- selected[prior.weight >= 10^-4]


      if(length(sub.selected) >= 2){

        selection.list[[j]] <- c(1:n)[-j][sub.selected]
        Ymat <- t_xx[-j, ][sub.selected, ]
        Yflagmat <- t_zero.matrix[-j, ][sub.selected, ]
        reads <- reweighting_C(Ymat, Yflagmat, xx[, j], zero.matrix[, j])

      } else {

        selection.list[[j]] <- c(1:n)[-j][selected]
        Ymat <- t_xx[-j, ][selected, ]
        Yflagmat <- t_zero.matrix[-j, ][selected, ]
        reads <- reweighting_C(Ymat, Yflagmat, xx[, j], zero.matrix[, j])

      }
      # prior.weight <- rep(1, length(sub.selected)+1)
      # reads <- NULL
      # for(i in 1:p){
      #   Y <- unlist(c(xx[i, j], xx[i, -j][sub.selected]))
      #   Yflag <- unlist(c(zero.matrix[i, j], zero.matrix[i, -j][sub.selected]))
      #   new.weights <- reweighting(prior.weight, Y, Yflag)
      #   sumY <- sum(new.weights*Y)*length(Y)
      #   reads <- c(reads, sumY)
      # }

    } else {
      selection.list[[j]] <- c(1:n)[-j][selected]

      Ymat <- as.matrix(t_xx[-j, ][selected, ])
      Yflagmat <- as.matrix(t_zero.matrix[-j, ][selected, ])
      reads <- reweighting_C(Ymat, Yflagmat, xx[, j], zero.matrix[, j])

      #
      # reads <- NULL
      # for(i in 1:p){
      #   Y <- unlist(c(xx[i, j], xx[i, -j][selected]))
      #   Yflag <- unlist(c(zero.matrix[i, j], zero.matrix[i, -j][selected]))
      #   new.weights <- reweighting(prior.weight, Y, Yflag)
      #   sumY <- sum(new.weights*Y)*length(Y)
      #   reads <- c(reads, sumY)
      # }

    }

    labels <- reads > 0 & reads <= quantile(reads, 0.995) & gc_copy!=0 & xx[, j] > 0
    est <- new_initialization_of_parameters(xx[, j][labels], gc[labels], reads[labels])
    as <- unlist(est$as)
    bs <- unlist(est$bs)
    a0 <- as[1]
    a1 <- as[2]
    a2 <- as[3]
    a3 <- as[4]
    a4 <- as[5]
    b1 <- bs[1]
    b2 <- bs[2]
    b3 <- bs[3]
    b4 <- bs[4]
    est <- Mix_gradient_descent_for_individual_sample(xx[, j][labels], gc[labels], reads[labels], a0, a1, a2, a3, a4, b1, b2, b3, b4, psi, gamma, steps, down)
    parameters <- est$parameters
    to.pred <- xx[, j] == 0
    gc[is.na(gc)] <- median(gc,na.rm=T)
    predicted <- Predict_for_individual_sample(gc[to.pred], reads[to.pred], parameters[1], parameters[2], parameters[3], parameters[4],
                                               parameters[5], parameters[6], parameters[7], parameters[8], parameters[9])
    imputed[to.pred, j] <- predicted
    para.list[[j]] <-  parameters

  }

  res <- list(imputed = imputed, para.list = para.list, selection.list = selection.list)
  return(res)
}




