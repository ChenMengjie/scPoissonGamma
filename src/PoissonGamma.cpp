#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;
using namespace std;


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec initialize_coef(arma::vec Y, arma::vec X){
  Environment myEnv("package:scPoissonGamma");
  Function initialize_coef = myEnv["initialize_coef"];
  Rcpp::List initialize_coef_res = wrap(initialize_coef(Y, X));
  return as<NumericVector>(initialize_coef_res["coef"]);
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec initialize_mu_alpha(arma::vec Y, arma::vec X_beta){
  Environment myEnv("package:scPoissonGamma");
  Function initialize_mu_alpha = myEnv["initialize_mu_alpha"];
  Rcpp::List initialize_mu_alpha_res = wrap(initialize_mu_alpha(Y, X_beta));
  return as<NumericVector>(initialize_mu_alpha_res["coef"]);
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec initialize_mu_alpha_with_group(arma::vec Y, arma::vec X_beta, arma::vec group){
  Environment myEnv("package:scPoissonGamma");
  Function initialize_mu_alpha_with_group = myEnv["initialize_mu_alpha_with_group"];
  Rcpp::List initialize_mu_alpha_with_group_res = wrap(initialize_mu_alpha_with_group(Y, X_beta, group));
  return as<NumericVector>(initialize_mu_alpha_with_group_res["coef"]);
}



// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec initialize_coef_multiple_beta(arma::vec Y, arma::mat X){
  Environment myEnv("package:scPoissonGamma");
  Function initialize_coef_multiple_beta = myEnv["initialize_coef_multiple_beta"];
  Rcpp::List initialize_coef_multiple_beta_res = wrap(initialize_coef_multiple_beta(Y, X));
  return as<NumericVector>(initialize_coef_multiple_beta_res["coef"]);
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec initialize_mu_alpha_multiple_beta(arma::vec Y, arma::mat X, arma::vec betas, arma::vec Z){
  Environment myEnv("package:scPoissonGamma");
  Function initialize_mu_alpha_multiple_beta = myEnv["initialize_mu_alpha_multiple_beta"];
  Rcpp::List initialize_mu_alpha_multiple_beta_res = wrap(initialize_mu_alpha_multiple_beta(Y,  X, betas, Z));
  return as<NumericVector>(initialize_mu_alpha_multiple_beta_res["coef"]);
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec initialize_mu_alpha_multiple_beta_nowith_alpha(arma::vec Y, arma::mat X, arma::vec Z){
  Environment myEnv("package:scPoissonGamma");
  Function initialize_mu_alpha_multiple_beta_nowith_alpha = myEnv["initialize_mu_alpha_multiple_beta_nowith_alpha"];
  Rcpp::List initialize_mu_alpha_multiple_beta_nowith_alpha_res = wrap(initialize_mu_alpha_multiple_beta_nowith_alpha(Y,  X, Z));
  return as<NumericVector>(initialize_mu_alpha_multiple_beta_nowith_alpha_res["coef"]);
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec initialize_mu_alpha_with_group_multiple_beta(arma::vec Y, arma::mat X, arma::vec betas, arma::vec Z, arma::vec group){
  Environment myEnv("package:scPoissonGamma");
  Function initialize_mu_alpha_with_group_multiple_beta = myEnv["initialize_mu_alpha_with_group_multiple_beta"];
  Rcpp::List initialize_mu_alpha_with_group_multiple_beta_res = wrap(initialize_mu_alpha_with_group_multiple_beta(Y, X, betas, Z, group));
  return as<NumericVector>(initialize_mu_alpha_with_group_multiple_beta_res["coef"]);
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec initialize_mu_alpha_with_group_multiple_beta_nowith_alpha(arma::vec Y, arma::mat X, arma::vec Z, arma::vec group){
  Environment myEnv("package:scPoissonGamma");
  Function initialize_mu_alpha_with_group_multiple_beta_nowith_alpha = myEnv["initialize_mu_alpha_with_group_multiple_beta_nowith_alpha"];
  Rcpp::List initialize_mu_alpha_with_group_multiple_beta_nowith_alpha_res = wrap(initialize_mu_alpha_with_group_multiple_beta_nowith_alpha(Y, X, Z, group));
  return as<NumericVector>(initialize_mu_alpha_with_group_multiple_beta_nowith_alpha_res["coef"]);
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec gradient_beta_psi(arma::vec Y, arma::vec X, double beta, double psi, int n){
  
  arma::vec gradient = arma::zeros<arma::vec>(2);
  
  double common_terms = log(psi) - R::digamma(psi);
  for(int i = 0; i < n; ++i){
    
    double e_term = exp(X(i)*beta);
    double e_term_psi = e_term + psi;
    double Y_i_psi = Y(i) + psi;
    double e_term_fac = Y_i_psi/e_term_psi;
    double f_term = Y(i) - e_term*e_term_fac;
    
    gradient(0) += f_term*X(i);
    
    double gradient_psi_i = -log(e_term_psi) - e_term_fac + R::digamma(Y_i_psi);
    gradient(1) += gradient_psi_i;
    
  }
  gradient(1) += common_terms*n + n;
  return gradient;
}


// [[Rcpp::export]]
double log_factorial(int Y){
  double res = 0;
  for(int kk = 1; kk <= Y; ++kk){
    res += log(kk);
  }
  return res;
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec log_factorial_calculated(int N){
  
  arma::vec values = arma::zeros<arma::vec>(N+1);
  
  for(int kk = 1; kk <= N; ++kk){
    values(kk) = values(kk-1) + log(kk);
  }
  
  return values;
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

double LogLikelihood_beta_psi(arma::vec Y, arma::vec X, double beta, double psi, int n, double sum_log_factorial_Y){
  
  double Likelihood = 0;
  double common_term = -lgamma(psi) + psi*log(psi);
  
  for(int i = 0; i < n; ++i){
    
    double Xbeta_i = X(i)*beta;
    double e_term = exp(Xbeta_i);
    double psi_Yi = psi + Y(i);
    double ll = lgamma(psi_Yi) - psi_Yi*log(psi + e_term) + Y(i)*(Xbeta_i);
    Likelihood += ll;
    
  }
  Likelihood += common_term*n - sum_log_factorial_Y;
  return Likelihood;
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

double select_stepsize_for_beta(arma::vec Y, arma::vec X, double gra_beta, double ll, double beta, double psi, int n, double gamma, double sum_log_factorial_Y, double down){
  
  double gra_beta2 = gra_beta*gra_beta*gamma;
  //  double lb = LogLikelihood_beta_psi(Y, X, beta, psi, n);
  double start = 0.1;
  if(beta >= 0.001) start = sqrt(abs(beta/gra_beta))/2;
  
  double aa = start;
  double selected = beta;
  while(aa > 0){
    double aa2 = aa*aa;
    double beta_prime = beta + aa2*gra_beta;
    double lb_prime = LogLikelihood_beta_psi(Y, X, beta_prime, psi, n, sum_log_factorial_Y);
    if(lb_prime - ll - aa2*gra_beta2 > 0 | abs(lb_prime - ll - aa2*gra_beta2) < 0.0001) {
      selected = beta_prime;
      break;
    }
    aa = aa - start*down;
  }
  if(selected >= 20) selected = beta;
  return selected;
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

double select_stepsize_for_psi(arma::vec Y, arma::vec X, double gra_psi, double ll, double beta, double psi, int n, double gamma, double sum_log_factorial_Y, double down){
  
  double gra_psi2 = gra_psi*gra_psi*gamma;
  //  double lb = LogLikelihood_beta_psi(Y, X, beta, psi, n);
  double start = sqrt(abs(psi/gra_psi))/2;
  
  double aa = start;
  double selected = psi;
  while(aa > 0){
    double aa2 = aa*aa;
    double psi_prime = psi + aa2*gra_psi;
    if(psi_prime > 0){
      double lpsi_prime = LogLikelihood_beta_psi(Y, X, beta, psi_prime, n, sum_log_factorial_Y);
      if(lpsi_prime - ll - aa2*gra_psi2 > 0 | abs(lpsi_prime - ll - aa2*gra_psi2) < 0.0001) {
        selected = psi_prime;
        break;
      }
    }
    aa = aa - start*down;
  }
  if(selected >= 20) selected = psi;
  return selected;
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec gradient_descent_beta_psi(arma::vec Y, arma::vec X, double beta, double psi, double gamma, int steps, double sum_log_factorial_Y, double down){
  
  int n = Y.n_elem;
  arma::vec gradient = gradient_beta_psi(Y, X, beta, psi, n);
  double ll = LogLikelihood_beta_psi(Y, X, beta, psi, n, sum_log_factorial_Y);
  
  for(int i = 0; i < steps; ++i){
    if(abs(gradient(0)) >= 0.0001){
      beta = select_stepsize_for_beta(Y, X, gradient(0), ll, beta, psi, n, gamma, sum_log_factorial_Y, down);
    }
    if(abs(gradient(1)) >= 0.0001){
      psi = select_stepsize_for_psi(Y, X, gradient(1), ll, beta, psi, n, gamma, sum_log_factorial_Y, down);
    }
    
    gradient = gradient_beta_psi(Y, X, beta, psi, n);
    ll = LogLikelihood_beta_psi(Y, X, beta, psi, n, sum_log_factorial_Y);
  }
  
  arma::vec est = arma::zeros<arma::vec>(3);
  est(0) = beta;
  est(1) = psi;
  est(2) = ll;
  
  return est;
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

double gradient_psi(arma::vec Y,  double psi, int n){
  
  double gradient = 0;
  double common_terms = log(psi) - R::digamma(psi);
  
  for(int i = 0; i < n; ++i){
    
    double Y_i_psi = Y(i) + psi;
    double gradient_i = -log(psi + 1) - (Y_i_psi)/(1 + psi) + R::digamma(Y_i_psi);
    gradient += gradient_i;
    
  }
  
  gradient += common_terms*n + n;
  return gradient;
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

double LogLikelihood_psi(arma::vec Y, double psi, int n, double sum_log_factorial_Y){
  
  double Likelihood = 0;
  double common_term = -lgamma(psi) + psi*log(psi);
  
  for(int i = 0; i < n; ++i){
    double Y_i_psi = Y(i) + psi;
    double ll = lgamma( Y_i_psi) - Y_i_psi*log(psi + 1) ;
    Likelihood += ll;
    
  }
  Likelihood += common_term*n - sum_log_factorial_Y;
  return Likelihood;
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

double select_stepsize_psi_alt(arma::vec Y, double gra_psi, double ll, double psi, int n, double gamma, double sum_log_factorial_Y, double down){
  
  double gra_psi2 = gra_psi*gra_psi*gamma;
  double start = sqrt(abs(psi/gra_psi))/2;
  
  double aa = start;
  double selected = psi;
  while(aa > 0){
    double aa2 = aa*aa;
    double psi_prime = psi + aa2*gra_psi;
    if(psi_prime > 0){
      double lpsi_prime = LogLikelihood_psi(Y, psi_prime, n, sum_log_factorial_Y);
      if(lpsi_prime - ll - aa2*gra_psi2 > 0 | abs(lpsi_prime - ll - aa2*gra_psi2) < 0.0001) {
        selected = psi_prime;
        break;
      }
    }
    aa = aa - start*down;
  }
  if(selected >= 20) selected = psi;
  return selected;
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec gradient_descent_alt(arma::vec Y, double psi, double gamma, int steps, double sum_log_factorial_Y, double down){
  
  int n = Y.n_elem;
  double gradient = gradient_psi(Y, psi, n);
  double ll = LogLikelihood_psi(Y, psi, n, sum_log_factorial_Y);
  
  for(int i = 0; i < steps; ++i){
    if(abs(gradient) >= 0.0001){
      psi = select_stepsize_psi_alt(Y, gradient, ll, psi, n, gamma, sum_log_factorial_Y, down);
    }
    gradient = gradient_psi(Y, psi, n);
    ll = LogLikelihood_psi(Y, psi, n, sum_log_factorial_Y);
  }
  
  arma::vec est = arma::zeros<arma::vec>(2);
  est(0) = psi;
  est(1) = ll;
  
  return est;
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

Rcpp::List PoissionGamma(arma::vec Y, arma::vec X, double beta, double psi, double gamma, int steps, double down){
  
  
  arma::vec calculated_values = log_factorial_calculated(Y.max());
  
  int n = Y.n_elem;
  arma::vec log_factorial_Y = arma::zeros<arma::vec>(n);
  for(int i = 0; i < n; ++i){
    log_factorial_Y(i) = calculated_values(Y(i));
  }
  
  double sum_log_factorial_Y = sum(log_factorial_Y);
  
  arma::vec est = gradient_descent_beta_psi(Y, X, beta, psi, gamma, steps, sum_log_factorial_Y, down);
  arma::vec est_alt = gradient_descent_alt(Y, psi, gamma, steps, sum_log_factorial_Y, down);
  
  double test_statistics = est(2) - est_alt(1);
  double p_value = 1 - R::pchisq(2*test_statistics, 1, TRUE, FALSE);
  
  return Rcpp::List::create(Rcpp::Named("beta") = est(0), Rcpp::Named("p_value") = p_value,
                            Rcpp::Named("psi") = est(1), Rcpp::Named("alt_psi") = est_alt(0),
                            Rcpp::Named("likelihood") = est(2), Rcpp::Named("alt_likelihood") = est_alt(1));
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::vec gradient_all(arma::vec Y, arma::vec X, double beta, double psi, double mu1, double mu2, double alpha, arma::vec posterior, int n){
  
  arma::vec gradient = arma::zeros<arma::vec>(5);
  double common_term = log(psi) - R::digamma(psi);
  double sum_posterior = sum(posterior);
  
  for(int i = 0; i < n; ++i){
    
    double e_term = exp(X(i)*beta + mu2);
    double e_term_psi = e_term + psi;
    double Y_i_psi = Y(i) + psi;
    double e_term_fac = Y_i_psi/e_term_psi;
    double f_term = Y(i) - e_term*e_term_fac;
    
    double e_term_prime = exp(alpha*X(i)*beta + mu1);
    double e_term_prime_frac = e_term_prime/(e_term_prime + 1);
    
    gradient(0) += (f_term + alpha)*X(i)*posterior(i) - e_term_prime_frac*alpha*X(i); //beta
    
    double gradient_psi_i =  -log(e_term_psi) - e_term_fac + R::digamma(Y_i_psi);
    gradient(1) += gradient_psi_i*posterior(i); //psi
    
    double g_term = posterior(i) -e_term_prime_frac;
    
    gradient(2) += g_term; //mu1
    gradient(3) += f_term*posterior(i); //mu2
    gradient(4) += g_term*X(i); //alpha
    
  }
  
  gradient(1) += sum_posterior*(common_term + 1);
  gradient(4) *= beta;
  
  return gradient;
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

arma::mat gradient_all_cov(arma::vec Y, arma::vec X, double beta, double psi, double mu1, double mu2, double alpha, arma::vec posterior, int n){
  
  arma::mat gradient = arma::zeros<arma::mat>(n, 5);
  double common_term = log(psi) - R::digamma(psi);
  
  for(int i = 0; i < n; ++i){
    
    double e_term = exp(X(i)*beta + mu2);
    double e_term_psi = e_term + psi;
    double Y_i_psi = Y(i) + psi;
    double e_term_fac = Y_i_psi/e_term_psi;
    double f_term = Y(i) - e_term*e_term_fac;
    
    double e_term_prime = exp(alpha*X(i)*beta + mu1);
    double e_term_prime_frac = e_term_prime/(e_term_prime + 1);
    
    gradient(i, 0) = (f_term + alpha)*X(i)*posterior(i) - e_term_prime_frac*alpha*X(i); //beta
    
    double gradient_psi_i = common_term - log(e_term_psi) - e_term_fac + R::digamma(Y_i_psi);
    gradient(i, 1) = gradient_psi_i*posterior(i) + posterior(i); //psi
    
    double g_term = posterior(i) -e_term_prime_frac;
    
    gradient(i, 2) = g_term; //mu1
    gradient(i, 3) = f_term*posterior(i); //mu2
    gradient(i, 4) = g_term*beta*X(i); //alpha
    
  }
  
  gradient.col(4) *=beta;
  arma::mat est = arma::cov(gradient);
  
  return est;
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

double Expected_logLikelihood(arma::vec Y, arma::vec X, double beta, double psi, double mu1, double mu2, double alpha, arma::vec posterior, int n, arma::vec log_factorial_Y){
  
  double Likelihood = 0;
  double common_term = -lgamma(psi) + psi*log(psi);
  double sum_posterior = sum(posterior);
  
  for(int i = 0; i < n; ++i){
    
    double Xbeta_mu2 = X(i)*beta + mu2;
    double e_term = exp(Xbeta_mu2);
    double l_term = alpha*X(i)*beta + mu1;
    double psi_Yi = psi+Y(i);
    double h_term = lgamma(psi_Yi) - log_factorial_Y(i) - psi_Yi*log(psi + e_term) + Y(i)*(Xbeta_mu2);
    Likelihood += -log(exp(l_term) + 1) + posterior(i)*l_term + posterior(i)*h_term;
    
  }
  Likelihood += sum_posterior*common_term;
  return Likelihood;
  
}


// [[Rcpp::export]]

double LogLikelihood_single(double Y_i, double X_i, double beta, double psi, double mu2, double log_factorial_Y_i){
  double Xbeta_mu2 = X_i*beta + mu2;
  double e_term = exp(Xbeta_mu2);
  double psi_Yi = psi + Y_i;
  double ll = -lgamma(psi) + psi*log(psi) + lgamma(psi_Yi) - log_factorial_Y_i - psi_Yi*log(psi + e_term) + Y_i*Xbeta_mu2;
  return ll;
}


// [[Rcpp::export]]

double Posterior_prob(double Y_i, double X_i, double beta, double psi, double mu1, double mu2, double alpha, double log_factorial_Y_i){
  
  double zero_prob = LogLikelihood_single(Y_i, X_i, beta, psi, mu2, log_factorial_Y_i);
  double e_term = 1/(exp(mu1 + alpha*X_i*beta) + 1);
  double prob1 = (1 - e_term)*exp(zero_prob);
  double pp = prob1/(e_term + prob1);
  return pp;
  
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

double PoissonMix_stepsize_for_beta(arma::vec Y, arma::vec X, double gra_beta, double ll, double beta, double psi,
                                    double mu1, double mu2, double alpha, arma::vec posterior, int n, double gamma, double down, arma::vec log_factorial_Y){
  
  double gra_beta2 = gra_beta*gra_beta*gamma;
  double start = 0.1;
  if(beta >= 0.001) start = sqrt(abs(beta/gra_beta))/2;
  
  double aa = start;
  double selected = beta;
  while(aa > 0){
    double aa2 = aa*aa;
    double beta_prime = beta + aa2*gra_beta;
    double lb_prime = Expected_logLikelihood(Y, X, beta_prime, psi, mu1, mu2, alpha, posterior, n, log_factorial_Y);
    if(lb_prime - ll - aa2*gra_beta2 > 0 ){// abs(lb_prime - ll - aa2*gra_beta2) < 0.0001) {
      selected = beta_prime;
      break;
    }
aa = aa - start*down;
//Rcpp::Rcout  << "aa" << aa << std::endl;
//Rcpp::Rcout  << "selected" << selected << std::endl;
    }
    // if(abs(selected) >= 10) selected = beta;
    return selected;
    
  }
  
  
  // [[Rcpp::depends("RcppArmadillo")]]
  // [[Rcpp::export]]
  
  double PoissonMix_stepsize_for_psi(arma::vec Y, arma::vec X, double gra_psi, double ll, double beta, double psi,
                                     double mu1, double mu2, double alpha, arma::vec posterior, int n, double gamma, double down, arma::vec log_factorial_Y){
    
    double gra_psi2 = gra_psi*gra_psi*gamma;
    double start = sqrt(abs(psi/gra_psi))/2;
    
    double aa = start;
    double selected = psi;
    while(aa > 0){
      double aa2 = aa*aa;
      double psi_prime = psi + aa2*gra_psi;
      if(psi_prime > 0){
        double lpsi_prime = Expected_logLikelihood(Y, X, beta, psi_prime, mu1, mu2, alpha, posterior, n, log_factorial_Y);
        if(lpsi_prime - ll - aa2*gra_psi2 > 0){// abs(lpsi_prime - ll - aa2*gra_psi2) < 0.0001) {
          selected = psi_prime;
          break;
        }
        }
        aa = aa - start*down;
      }
      // if(selected >= 20) selected = psi;
      return selected;
    }
    
    // [[Rcpp::depends("RcppArmadillo")]]
    // [[Rcpp::export]]
    
    double PoissonMix_stepsize_for_mu1(arma::vec Y, arma::vec X, double gra_mu1, double ll, double beta, double psi,
                                       double mu1, double mu2, double alpha, arma::vec posterior, int n, double gamma, double down, arma::vec log_factorial_Y){
      
      double gra_mu1_2 = gra_mu1*gra_mu1*gamma;
      double start = sqrt(abs(mu1/gra_mu1))/2;
      
      double aa = start;
      double selected = mu1;
      while(aa > 0){
        double aa2 = aa*aa;
        double mu1_prime = mu1 + aa2*gra_mu1;
        double lmu1_prime = Expected_logLikelihood(Y, X, beta, psi, mu1_prime, mu2, alpha, posterior, n, log_factorial_Y);
        if(lmu1_prime - ll - aa2*gra_mu1_2 > 0){// | abs(lmu1_prime - ll - aa2*gra_mu1_2) < 0.0001) {
          selected = mu1_prime;
          break;
        }

aa = aa - start*down;
        }
        // if(abs(selected) >= 10) selected = mu1;
        return selected;
      }
      
      // [[Rcpp::depends("RcppArmadillo")]]
      // [[Rcpp::export]]
      
      double PoissonMix_stepsize_for_mu2(arma::vec Y, arma::vec X, double gra_mu2, double ll, double beta, double psi,
                                         double mu1, double mu2, double alpha, arma::vec posterior, int n, double gamma, double down, arma::vec log_factorial_Y){
        
        double gra_mu2_2 = gra_mu2*gra_mu2*gamma;
        double start = sqrt(abs(mu2/gra_mu2))/2;
        
        double aa = start;
        double selected = mu2;
        while(aa > 0){
          double aa2 = aa*aa;
          double mu2_prime = mu2 + aa2*gra_mu2;
          double lmu2_prime = Expected_logLikelihood(Y, X, beta, psi, mu1, mu2_prime, alpha, posterior, n, log_factorial_Y);
          if(lmu2_prime - ll - aa2*gra_mu2_2 > 0 ){// abs(lmu2_prime - ll - aa2*gra_mu2_2) < 0.0001) {
            selected = mu2_prime;
            break;
          }
aa = aa - start*down;
          }
          
          //  if(abs(selected) >= 10) selected = mu2;
          return selected;
        }
        
        
        // [[Rcpp::depends("RcppArmadillo")]]
        // [[Rcpp::export]]
        
        double PoissonMix_stepsize_for_alpha(arma::vec Y, arma::vec X, double gra_alpha, double ll, double beta, double psi,
                                             double mu1, double mu2, double alpha, arma::vec posterior, int n, double gamma, double down, arma::vec log_factorial_Y){
          
          double gra_alpha_2 = gra_alpha*gra_alpha*gamma;
          double start = sqrt(abs(alpha/gra_alpha))/2;
          
          double aa = start;
          double selected = alpha;
          while(aa > 0){
            double aa2 = aa*aa;
            double alpha_prime = alpha + aa2*gra_alpha;
            double lalpha_prime = Expected_logLikelihood(Y, X, beta, psi, mu1, mu2, alpha_prime, posterior, n, log_factorial_Y);
            if(lalpha_prime - ll - aa2*gra_alpha_2 > 0 ){ //| abs(lalpha_prime - ll - aa2*gra_alpha_2) < 0.0001) {
              selected = alpha_prime;
              break;
            }
aa = aa - start*down;
            }
            
            //  if(abs(selected) >= 10) selected = alpha;
            return selected;
          }
          
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          arma::vec gradient_descent_PoissonGamma(arma::vec Y, arma::vec X, double beta, double psi, double mu1, double mu2, double alpha, arma::vec posterior, int n,
                                                  double gamma, int steps, double down, arma::vec log_factorial_Y){
            
            arma::vec gradient = gradient_all(Y, X, beta, psi, mu1, mu2, alpha, posterior, n);
            double ll = Expected_logLikelihood(Y, X, beta, psi, mu1, mu2, alpha, posterior, n, log_factorial_Y);
            
            double beta_prime = 0; double psi_prime = 0; double mu1_prime = 0; double mu2_prime = 0; double alpha_prime = 0;
            for(int i = 0; i < steps; ++i){
              if(abs(gradient(0)) >= 0.00001){
                beta_prime = PoissonMix_stepsize_for_beta(Y, X, gradient(0), ll, beta, psi, mu1, mu2, alpha, posterior, n, gamma, down, log_factorial_Y);
              } else {
                beta_prime = beta;
              }
              if(abs(gradient(1)) >= 0.00001){
                psi_prime = PoissonMix_stepsize_for_psi(Y, X, gradient(1), ll, beta, psi, mu1, mu2, alpha, posterior, n, gamma, down, log_factorial_Y);
              } else {
                psi_prime = psi;
              }
              if(abs(gradient(2)) >= 0.00001){
                mu1_prime = PoissonMix_stepsize_for_mu1(Y, X, gradient(2), ll, beta, psi, mu1, mu2, alpha, posterior, n, gamma, down, log_factorial_Y);
              } else {
                mu1_prime = mu1;
              }
              if(abs(gradient(3)) >= 0.00001){
                mu2_prime = PoissonMix_stepsize_for_mu2(Y, X, gradient(3), ll, beta, psi, mu1, mu2, alpha, posterior, n, gamma, down, log_factorial_Y);
              } else {
                mu2_prime = mu2;
              }
              if(abs(gradient(4)) >= 0.00001){
                alpha_prime = PoissonMix_stepsize_for_alpha(Y, X, gradient(4), ll, beta, psi, mu1, mu2, alpha, posterior, n, gamma, down, log_factorial_Y);
              } else {
                alpha_prime = alpha;
              }
              
              beta = beta_prime; psi = psi_prime; mu1 = mu1_prime; mu2 = mu2_prime; alpha = alpha_prime;
              
              gradient = gradient_all(Y, X, beta, psi, mu1, mu2, alpha, posterior, n);
              ll = Expected_logLikelihood(Y, X, beta, psi, mu1, mu2, alpha, posterior, n, log_factorial_Y);
            }
            
            arma::vec est = arma::zeros<arma::vec>(6);
            est(0) = beta;
            est(1) = psi;
            est(2) = mu1;
            est(3) = mu2;
            est(4) = alpha;
            est(5) = ll;
            
            
            return est ;
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          arma::vec gradient_descent_PoissonGamma_alt(arma::vec Y, arma::vec X, double psi, double mu1, double mu2, arma::vec posterior, int n,
                                                      double gamma, int steps, double down, arma::vec log_factorial_Y){
            
            arma::vec gradient = gradient_all(Y, X, 0, psi, mu1, mu2, 0, posterior, n);
            double ll = Expected_logLikelihood(Y, X, 0, psi, mu1, mu2, 0, posterior, n, log_factorial_Y);
            
            double psi_prime = 0; double mu1_prime = 0; double mu2_prime = 0;
            
            for(int i = 0; i < steps; ++i){
              if(abs(gradient(1)) >= 0.00001){
                psi_prime = PoissonMix_stepsize_for_psi(Y, X, gradient(1), ll, 0, psi, mu1, mu2, 0, posterior, n, gamma, down, log_factorial_Y);
              } else {
                psi_prime = psi;
              }
              if(abs(gradient(2)) >= 0.00001){
                mu1_prime = PoissonMix_stepsize_for_mu1(Y, X, gradient(2), ll, 0, psi, mu1, mu2, 0, posterior, n, gamma, down, log_factorial_Y);
              } else {
                mu1_prime = mu1;
              }
              if(abs(gradient(3)) >= 0.00001){
                mu2_prime = PoissonMix_stepsize_for_mu2(Y, X, gradient(3), ll, 0, psi, mu1, mu2, 0, posterior, n, gamma, down, log_factorial_Y);
              } else {
                mu2_prime = mu2;
              }
              
              psi = psi_prime; mu1 = mu1_prime; mu2 = mu2_prime;
              gradient = gradient_all(Y, X, 0, psi, mu1, mu2, 0, posterior, n);
              ll = Expected_logLikelihood(Y, X, 0, psi, mu1, mu2, 0, posterior, n, log_factorial_Y);
              
            }
            
            arma::vec est = arma::zeros<arma::vec>(6);
            est(1) = psi;
            est(2) = mu1;
            est(3) = mu2;
            est(5) = ll;
            
            return est;
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          arma::mat Fisher_information(arma::vec Y, arma::vec X, double beta, double psi, double mu1,  double mu2, double alpha, arma::vec posterior, int n){
            arma::mat Fisher = arma::zeros<arma::mat>(5, 5);
            arma::vec e_term_prime = exp(mu1 + alpha*beta*X);
            arma::vec e_term_square_inverse_prime = 1/pow(e_term_prime + 1, 2);
            arma::vec a_term = e_term_prime%e_term_square_inverse_prime;
            
            arma::vec Y_psi = Y + psi;
            arma::vec e_term = exp(X*beta + mu2);
            arma::vec e_term_fac_prime = 1/(e_term + psi);
            arma::vec e_term_fac = e_term%e_term_fac_prime;
            arma::vec e_term_fac_2 = 1/pow(e_term + psi, 2);
            arma::vec e_term_square_inverse = e_term%e_term_fac_2;
            arma::vec Y_i_psi_e_term_psi_inv2 = Y_psi%e_term_square_inverse;
            arma::vec X_square = pow(X, 2);
            arma::vec FX = posterior%X;
            double common_psi = - 1/psi + R::trigamma(psi);
            
            
            Fisher(0, 0) = sum(pow(alpha, 2)*a_term%X_square) + sum(psi*FX%Y_i_psi_e_term_psi_inv2%X); //beta
            Fisher(0, 1) = Fisher(1, 0) = sum(e_term_fac%FX) - sum(FX%Y_i_psi_e_term_psi_inv2); // beta, psi
            Fisher(0, 2) = Fisher(2, 0) = sum(alpha*a_term%X); //beta, mu1
            Fisher(0, 3) = Fisher(3, 0) = sum(psi*FX%Y_i_psi_e_term_psi_inv2); //beta, mu2
            Fisher(0, 4) = Fisher(4, 0) = sum(alpha*beta*a_term%X_square) + sum((1 - 1/(e_term_prime + 1))%X) - sum(FX); //beta, alpha
            
            arma::vec psi2_vec = arma::zeros<arma::vec>(n);
            for(int j = 0; j < n; ++j){
              psi2_vec(j) = - R::trigamma(Y_psi(j)) - Y_psi(j)*e_term_fac_2(j)  + 2*e_term_fac_prime(j);
            }
            Fisher(1, 1) = sum(posterior%psi2_vec) + common_psi*sum(posterior); //psi, psi
            Fisher(1, 3) = Fisher(3, 1) = sum(e_term_fac%posterior) - sum(posterior%Y_i_psi_e_term_psi_inv2);  //psi, mu2
            Fisher(2, 2) = sum(a_term); //mu1, mu1
            Fisher(2, 4) = Fisher(4, 2) = beta*sum(a_term%X); //mu1, alpha
            Fisher(3, 3) = psi*sum(posterior%Y_i_psi_e_term_psi_inv2); //mu2, mu2
            Fisher(4, 4) = pow(beta, 2)*sum(a_term%X_square) ; //alpha, alpha
            
            return Fisher;
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          
          Rcpp::List EM_PoissonGamma(arma::vec Y, arma::vec X, arma::uvec Y_ind_zero, double beta, double psi, double mu1, double mu2,
                                     double alpha, arma::vec posterior, double gamma, int steps, int EM_steps, int n, int n_1, double down,
                                     arma::vec log_factorial_Y, bool ReportAll){
            
            
            arma::vec est = arma::zeros<arma::vec>(6);
            arma::mat Fisher = arma::zeros<arma::mat>(5, 5);
            arma::mat all_est = arma::zeros<arma::mat>(6, EM_steps);
            arma::vec all_pvalue = arma::zeros<arma::vec>(EM_steps);
            arma::vec all_SE = arma::zeros<arma::vec>(EM_steps);
            
            for(int i = 0; i < EM_steps; ++i){
              
              arma::vec res = gradient_descent_PoissonGamma(Y, X, beta, psi, mu1, mu2, alpha, posterior, n, gamma, steps, down, log_factorial_Y);
              beta = res(0);
              psi = res(1);
              mu1 = res(2);
              mu2 = res(3);
              alpha = res(4);
              
              for(int j = 0; j < n_1; ++j){
                int id = Y_ind_zero(j);
                posterior(id) = Posterior_prob(Y(id), X(id), beta, psi, mu1, mu2, alpha, log_factorial_Y(id));
              }
              
              if(ReportAll == FALSE){
                if(i == EM_steps - 1)  {
                  est = res;
                  Fisher = Fisher_information(Y, X, beta, psi, mu1, mu2, alpha, posterior, n);
                  arma::mat score = gradient_all_cov(Y, X, beta, psi, mu1, mu2, alpha, posterior, n);
                  Fisher -= score;
                }
              } else {
                all_est.col(i) = res;
                Fisher = Fisher_information(Y, X, beta, psi, mu1, mu2, alpha, posterior, n);
                arma::mat score = gradient_all_cov(Y, X, beta, psi, mu1, mu2, alpha, posterior, n);
                Fisher -= score;
                arma::mat est_cov = arma::inv(Fisher);
                double SE = sqrt(abs(est_cov(0, 0)));
                all_SE(i) = SE;
                all_pvalue(i) = 2*R::pnorm(-abs(beta)/SE, 0, 1, TRUE, FALSE);
              }
              
              //    Rcpp::Rcout  << "ll" << i << ":" << res(5) << std::endl;
            }
            
            if(ReportAll == FALSE){
              return Rcpp::List::create(Rcpp::Named("est") = est,
                                        Rcpp::Named("Fisher") = Fisher);
            } else {
              
              return Rcpp::List::create(Rcpp::Named("all_est") = all_est,
                                        Rcpp::Named("all_SE") = all_SE,
                                        Rcpp::Named("all_pvalue") = all_pvalue);
              
            }
            
          }
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          
          arma::vec EM_PoissonGamma_alt(arma::vec Y, arma::vec X, arma::uvec Y_ind_zero, double psi, double mu1, double mu2,
                                        arma::vec posterior, double gamma, int steps, int EM_steps, int n, int n_1, double down, arma::vec log_factorial_Y){
            
            arma::vec est = arma::zeros<arma::vec>(6);
            
            for(int i = 0; i < EM_steps; ++i){
              
              arma::vec res = gradient_descent_PoissonGamma_alt(Y, X, psi, mu1, mu2,  posterior, n, gamma, steps, down,  log_factorial_Y);
              psi = res(1);
              mu1 = res(2);
              mu2 = res(3);
              
              for(int j = 0; j < n_1; ++j){
                int id = Y_ind_zero(j);
                posterior(id) = Posterior_prob(0, X(id), 0, psi, mu1, mu2, 0, log_factorial_Y(id));
              }
              
              if(i == EM_steps - 1)  {
                est = res;
              }
            }
            
            return est;
            
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          Rcpp::List PoissonGamma_Mix(arma::vec Y, arma::vec X, double psi, double gamma, int steps, int EM_steps, double down,
                                      bool LRT, bool group, bool ReportAll){
            
            arma::uvec Y_ind_zero = arma::find(Y == 0);
            
            int n = Y.n_elem;
            int n_1 = Y_ind_zero.n_elem;
            
            arma::vec calculated_values = log_factorial_calculated(Y.max());
            arma::vec log_factorial_Y = arma::zeros<arma::vec>(n);
            for(int i = 0; i < n; ++i){
              log_factorial_Y(i) = calculated_values(Y(i));
            }
            
            arma::vec coef1 = initialize_coef(Y, X);
            double mu2 = coef1(0);
            double beta = coef1(1);
            
            arma::vec coef2 = arma::zeros<arma::vec>(2);
            arma::vec X_beta = X*beta;
            
            if(group == TRUE){
              coef2 = initialize_mu_alpha_with_group(Y, X_beta, X);
            } else {
              coef2 = initialize_mu_alpha(Y, X_beta);
            }
            double mu1 = coef2(0);
            double alpha = coef2(1);
            
            arma::vec posterior = arma::ones<arma::vec>(n);
            
            for(int j = 0; j < n_1; ++j){
              int id = Y_ind_zero(j);
              posterior(id) = Posterior_prob(0, X(id), beta, psi, mu1, mu2, alpha, log_factorial_Y(id));
            }
            
            Rcpp::List res = EM_PoissonGamma(Y, X, Y_ind_zero, beta, psi, mu1, mu2, alpha, posterior, gamma, steps, EM_steps, n, n_1, down, log_factorial_Y, ReportAll);
            
            if(ReportAll == FALSE){
              arma::mat Fisher = res["Fisher"];
              arma::mat est_cov = arma::inv(Fisher);
              arma::vec est = res["est"];
              
              double SE = sqrt(abs(est_cov(0, 0)));
              double zscore = abs(est(0))/SE;
              double WT_pvalue = 2*R::pnorm(-zscore, 0, 1, TRUE, FALSE);
              
              if(LRT == TRUE){
                
                arma::vec est_alt = EM_PoissonGamma_alt(Y, X, Y_ind_zero, psi, mu1, mu2, posterior, gamma, steps, EM_steps, n, n_1, down, log_factorial_Y);
                
                double test_statistics = est(5) - est_alt(5);
                double p_value = 1 - R::pchisq(2*test_statistics, 2, TRUE, FALSE);
                
                return Rcpp::List::create(Rcpp::Named("beta") = est(0), Rcpp::Named("LRT_pvalue") = p_value,
                                          Rcpp::Named("psi") = est(1), Rcpp::Named("alt_psi") = est_alt(1),
                                          Rcpp::Named("mu1") = est(2), Rcpp::Named("alt_mu1") = est_alt(2),
                                          Rcpp::Named("mu2") = est(3), Rcpp::Named("alt_mu2") = est_alt(3),
                                          Rcpp::Named("alpha") = est(4),
                                          Rcpp::Named("likelihood") = est(5), Rcpp::Named("alt_likelihood") = est_alt(5),
                                          Rcpp::Named("std") = SE, Rcpp::Named("WT_pvalue") = WT_pvalue);
                
              } else {
                return Rcpp::List::create(Rcpp::Named("beta") = est(0),
                                          Rcpp::Named("psi") = est(1),
                                          Rcpp::Named("mu1") = est(2),
                                          Rcpp::Named("mu2") = est(3),
                                          Rcpp::Named("alpha") = est(4),
                                          Rcpp::Named("likelihood") = est(5),
                                          Rcpp::Named("std") = SE,
                                          Rcpp::Named("WT_pvalue") = WT_pvalue);
              }
            } else {
              return Rcpp::List::create(Rcpp::Named("all_est") = res["all_est"],
                                        Rcpp::Named("std") = res["all_SE"],
                                        Rcpp::Named("WT_pvalue") = res["all_pvalue"]);
            }
          }
          
          
          
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          arma::vec gradient_multiple_beta_all(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, arma::vec beta,
                                               double psi, double delta, double mu1, double mu2, arma::vec alpha, arma::vec posterior, int n, int k){
            
            arma::vec gradient = arma::zeros<arma::vec>(4 + k*2);
            double common_term = log(psi) - R::digamma(psi);
            double sum_posterior = sum(posterior);
            
            for(int i = 0; i < n; ++i){
              
              double e_term = exp(Xbeta(i) + mu2);
              double e_term_psi = e_term + psi;
              double Y_i_psi = Y(i) + psi;
              double e_term_fac = Y_i_psi/e_term_psi;
              double f_term = Y(i) - e_term*e_term_fac;
              
              double e_term_prime = exp(alphaXbeta(i) + mu1 + delta*Z(i));
              double e_term_prime_frac = e_term_prime/(e_term_prime + 1);
              
              double gradient_psi_i =  -log(e_term_psi) - e_term_fac + R::digamma(Y_i_psi);
              gradient(0) += gradient_psi_i*posterior(i); //psi
              double g_term = posterior(i) -e_term_prime_frac;
              gradient(1) += g_term; //mu1
              gradient(2) += f_term*posterior(i); //mu2
              gradient(3) += g_term*Z(i); //delta
              
              for(int j = 0; j < k; ++j){
                double term1 = (f_term + alpha(j))*posterior(i);
                double term2 = e_term_prime_frac*alpha(j);
                gradient(j + 4) +=  term1*X(i, j) - term2*X(i, j); //beta_k
              }
              
              for(int j = 0; j < k; ++j){
                gradient(j + 4 + k) +=  g_term*X(i, j)*beta(j); //alpha_k
              }
              
            }
            gradient(0) += sum_posterior*(common_term + 1);
            
            return gradient;
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          arma::mat gradient_multiple_beta_all_cov(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, arma::vec beta,
                                                   double psi, double delta, double mu1, double mu2, arma::vec alpha, arma::vec posterior, int n, int k){
            
            arma::mat gradient = arma::zeros<arma::mat>(n, 4 + k*2);
            double common_term = log(psi) - R::digamma(psi);
            
            for(int i = 0; i < n; ++i){
              
              double e_term = exp(Xbeta(i) + mu2);
              double e_term_psi = e_term + psi;
              double Y_i_psi = Y(i) + psi;
              double e_term_fac = Y_i_psi/e_term_psi;
              double f_term = Y(i) - e_term*e_term_fac;
              
              double e_term_prime = exp(alphaXbeta(i) + mu1 + delta*Z(i));
              double e_term_prime_frac = e_term_prime/(e_term_prime + 1);
              
              double gradient_psi_i =  common_term -log(e_term_psi) - e_term_fac + R::digamma(Y_i_psi);
              gradient(i, 0) = gradient_psi_i*posterior(i) + posterior(i); //psi
              double g_term = posterior(i) -e_term_prime_frac;
              gradient(i, 1) = g_term; //mu1
              gradient(i, 2) = f_term*posterior(i); //mu2
              gradient(i, 3) = g_term*Z(i); //delta
              
              for(int j = 0; j < k; ++j){
                double term1 = (f_term + alpha(j))*posterior(i);
                double term2 = e_term_prime_frac*alpha(j);
                gradient(i, j + 4) =  term1*X(i, j) - term2*X(i, j); //beta_k
              }
              for(int j = 0; j < k; ++j){
                gradient(i, j + 4 + k) =  g_term*X(i, j)*beta(j); //alpha_k
              }
              
            }
            
            arma::mat est = arma::cov(gradient);
            
            return est;
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          double Expected_logLikelihood_multiple_beta(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, double psi,
                                                      double delta, double mu1, double mu2, arma::vec alphaXbeta, arma::vec posterior, int n, arma::vec log_factorial_Y){
            
            double Likelihood = 0;
            double common_term = -lgamma(psi) + psi*log(psi);
            double sum_posterior = sum(posterior);
            
            for(int i = 0; i < n; ++i){
              
              double Xbeta_mu2 = Xbeta(i) + mu2;
              double e_term = exp(Xbeta_mu2);
              double l_term = alphaXbeta(i) + mu1 + delta*Z(i);
              double psi_Yi = psi + Y(i);
              double h_term = lgamma(psi_Yi) - log_factorial_Y(i) - psi_Yi*log(psi + e_term) + Y(i)*(Xbeta_mu2);
              Likelihood += -log(exp(l_term) + 1) + posterior(i)*l_term + posterior(i)*h_term;
              
            }
            Likelihood += sum_posterior*common_term;
            return Likelihood;
            
          }
          
          
          // [[Rcpp::export]]
          
          double LogLikelihood_single_multiple_beta(double Y_i, double Xbeta_i, double psi, double mu2, double log_factorial_Y_i){
            double Xbeta_mu2 = Xbeta_i + mu2;
            double e_term = exp(Xbeta_mu2);
            double psi_Yi = psi + Y_i;
            double ll = -lgamma(psi) + psi*log(psi) + lgamma(psi_Yi) - log_factorial_Y_i - psi_Yi*log(psi + e_term) + Y_i*Xbeta_mu2;
            return ll;
          }
          
          
          // [[Rcpp::export]]
          
          double Posterior_prob_multiple_beta(double Y_i, double Xbeta_i, double Z_i, double psi, double delta,
                                              double mu1, double mu2, double alphaXbeta_i, double log_factorial_Y_i){
            
            double zero_prob = LogLikelihood_single_multiple_beta(Y_i, Xbeta_i, psi, mu2, log_factorial_Y_i);
            double e_term = 1/(exp(mu1 +  alphaXbeta_i + Z_i*delta) + 1);
            double prob1 = (1 - e_term)*exp(zero_prob);
            double pp = prob1/(e_term + prob1);
            return pp;
            
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          double PoissonMix_stepsize_for_mutiple_beta(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, arma::vec betas,
                                                      int ind, arma::vec m_gra_beta, double ll, double psi, double delta, double mu1, double mu2,
                                                      arma::vec posterior, int n, double gamma, double down, arma::vec log_factorial_Y){
            
            double gra_beta = m_gra_beta(ind);
            double gra_beta2 = gra_beta*gra_beta*gamma;
            double start = 0.1;
            double beta = betas(ind);
            arma::vec Xbeta_remain = Xbeta - X.col(ind)*beta;
            if(beta >= 0.05) start = sqrt(abs(beta/gra_beta))/2;
            
            double aa = start;
            double selected = beta;
            while(aa > 0){
              double aa2 = aa*aa;
              double beta_prime = beta + aa2*gra_beta;
              arma::vec Xbeta_prime = Xbeta_remain + X.col(ind)*beta_prime;
              double lb_prime = Expected_logLikelihood_multiple_beta(Y, X, Z, Xbeta_prime, psi, delta, mu1, mu2, alphaXbeta, posterior, n, log_factorial_Y);
              if(lb_prime - ll - aa2*gra_beta2 > 0 ){
                selected = beta_prime;
                break;
              }
              aa = aa - start*down;
              
            }
            
            return selected;
            
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          double PoissonMix_stepsize_mutiple_beta_for_psi(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, double gra_psi, double ll, double psi, double delta,
                                                          double mu1, double mu2, arma::vec posterior, int n, double gamma, double down, arma::vec log_factorial_Y){
            
            double gra_psi2 = gra_psi*gra_psi*gamma;
            double start = sqrt(abs(psi/gra_psi))/2;
            
            double aa = start;
            double selected = psi;
            while(aa > 0){
              double aa2 = aa*aa;
              double psi_prime = psi + aa2*gra_psi;
              if(psi_prime > 0){
                double lpsi_prime = Expected_logLikelihood_multiple_beta(Y, X, Z, Xbeta, psi_prime, delta, mu1, mu2, alphaXbeta, posterior, n, log_factorial_Y);
                if(lpsi_prime - ll - aa2*gra_psi2 > 0){
                  selected = psi_prime;
                  break;
                }
              }
              aa = aa - start*down;
            }
            return selected;
          }
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          double PoissonMix_stepsize_mutiple_beta_for_mu1(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, double gra_mu1, double ll, double psi, double delta,
                                                          double mu1, double mu2, arma::vec posterior, int n, double gamma, double down, arma::vec log_factorial_Y){
            
            double gra_mu1_2 = gra_mu1*gra_mu1*gamma;
            double start = 0.1;
            if(abs(mu1) >= 0.05) start = sqrt(abs(mu1/gra_mu1))/2;
            
            double aa = start;
            double selected = mu1;
            while(aa > 0){
              double aa2 = aa*aa;
              double mu1_prime = mu1 + aa2*gra_mu1;
              double lmu1_prime = Expected_logLikelihood_multiple_beta(Y, X, Z, Xbeta, psi, delta, mu1_prime, mu2, alphaXbeta, posterior, n, log_factorial_Y);
              if(lmu1_prime - ll - aa2*gra_mu1_2 > 0){
                selected = mu1_prime;
                break;
              }
              
              aa = aa - start*down;
            }
            
            return selected;
          }
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          double PoissonMix_stepsize_mutiple_beta_for_mu2(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, double gra_mu2, double ll, double psi, double delta,
                                                          double mu1, double mu2, arma::vec posterior, int n, double gamma, double down, arma::vec log_factorial_Y){
            
            double gra_mu2_2 = gra_mu2*gra_mu2*gamma;
            
            double start = 0.1;
            if(abs(mu2) >= 0.05) start = sqrt(abs(mu2/gra_mu2))/2;
            
            double aa = start;
            double selected = mu2;
            while(aa > 0){
              double aa2 = aa*aa;
              double mu2_prime = mu2 + aa2*gra_mu2;
              double lmu2_prime = Expected_logLikelihood_multiple_beta(Y, X, Z, Xbeta, psi, delta, mu1, mu2_prime, alphaXbeta, posterior, n, log_factorial_Y);
              if(lmu2_prime - ll - aa2*gra_mu2_2 > 0 ){
                selected = mu2_prime;
                break;
              }
              aa = aa - start*down;
            }
            
            return selected;
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          double PoissonMix_stepsize_mutiple_beta_for_delta(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, double gra_delta, double ll, double psi, double delta,
                                                            double mu1, double mu2, arma::vec posterior, int n, double gamma, double down, arma::vec log_factorial_Y){
            
            double gra_delta_2 = gra_delta*gra_delta*gamma;
            double start = 0.1;
            if(abs(delta) >= 0.05) start = sqrt(abs(delta/gra_delta))/2;
            
            double aa = start;
            double selected = delta;
            while(aa > 0){
              double aa2 = aa*aa;
              double delta_prime = delta + aa2*gra_delta;
              double ldelta_prime = Expected_logLikelihood_multiple_beta(Y, X, Z, Xbeta, psi, delta_prime, mu1, mu2, alphaXbeta, posterior, n, log_factorial_Y);
              if(ldelta_prime - ll - aa2*gra_delta_2 > 0 ){
                selected = delta_prime;
                break;
              }
              aa = aa - start*down;
            }
            
            return selected;
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          double PoissonMix_stepsize_for_mutiple_alpha(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::mat Xbeta_mat, arma::vec alphaXbeta, arma::vec alphas, int ind,  arma::vec m_gra_alpha,
                                                       double ll, double psi, double delta, double mu1, double mu2, arma::vec posterior, int n, double gamma, double down, arma::vec log_factorial_Y){
            
            double gra_alpha = m_gra_alpha(ind);
            double gra_alpha_2 = gra_alpha*gra_alpha*gamma;
            double alpha = alphas(ind);
            
            arma::vec alphaXbeta_remain = alphaXbeta - Xbeta_mat.col(ind)*alpha;
            double start = 0.1;
            if(alpha >= 0.05) start = sqrt(abs(alpha/gra_alpha))/2;
            
            double aa = start;
            double selected = alpha;
            while(aa > 0){
              double aa2 = aa*aa;
              double alpha_prime = alpha + aa2*gra_alpha;
              arma::vec alphaXbeta_prime = alphaXbeta_remain + Xbeta_mat.col(ind)*alpha_prime;
              double lalpha_prime = Expected_logLikelihood_multiple_beta(Y, X, Z, Xbeta, psi, delta, mu1, mu2, alphaXbeta_prime, posterior, n, log_factorial_Y);
              if(lalpha_prime - ll - aa2*gra_alpha_2 > 0){
                selected = alpha_prime;
                break;
              }
              aa = aa - start*down;
            }
            
            return selected;
          }
          
          
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          arma::vec gradient_descent_PoissonGamma_mutiple_beta(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, arma::vec betas, double psi, double delta,
                                                               double mu1, double mu2, arma::vec alphas, arma::vec posterior, int n, int k,
                                                               double gamma, int steps, double down, arma::vec log_factorial_Y){
            
            arma::vec gradient = gradient_multiple_beta_all(Y, X, Z, Xbeta, alphaXbeta, betas, psi, delta, mu1, mu2, alphas, posterior, n, k);
            double ll = Expected_logLikelihood_multiple_beta(Y, X, Z, Xbeta, psi, delta, mu1, mu2, alphaXbeta, posterior, n, log_factorial_Y);
            
            
            double psi_prime = 0; double mu1_prime = 0; double mu2_prime = 0; double delta_prime = 0;
            arma::vec beta_prime = arma::zeros<arma::vec>(k); arma::vec alpha_prime = arma::zeros<arma::vec>(k);
            arma::vec m_gra_beta = gradient(arma::span(4, 3 + k));
            arma::vec m_gra_alpha = gradient(arma::span(4 + k, 3 + 2*k));
            
            arma::mat Xbeta_mat = X;
            for(int j = 0; j < k; ++j){
              Xbeta_mat.col(j) = X.col(j)*betas(j);
            }
            
            for(int i = 0; i < steps; ++i){
              
              if(abs(gradient(0)) >= 0.00001){
                psi_prime = PoissonMix_stepsize_mutiple_beta_for_psi(Y, X, Z, Xbeta, alphaXbeta, gradient(0), ll, psi, delta, mu1, mu2, posterior, n, gamma, down, log_factorial_Y);
              } else {
                psi_prime = psi;
              }
              if(abs(gradient(1)) >= 0.00001){
                mu1_prime = PoissonMix_stepsize_mutiple_beta_for_mu1(Y, X, Z, Xbeta, alphaXbeta, gradient(1), ll, psi, delta, mu1, mu2, posterior, n, gamma, down, log_factorial_Y);
              } else {
                mu1_prime = mu1;
              }
              if(abs(gradient(2)) >= 0.00001){
                mu2_prime = PoissonMix_stepsize_mutiple_beta_for_mu2(Y, X, Z, Xbeta, alphaXbeta, gradient(2), ll, psi, delta, mu1, mu2, posterior, n, gamma, down, log_factorial_Y);
              } else {
                mu2_prime = mu2;
              }
              if(abs(gradient(3)) >= 0.00001){
                delta_prime = PoissonMix_stepsize_mutiple_beta_for_delta(Y, X, Z, Xbeta, alphaXbeta, gradient(3), ll, psi, delta, mu1, mu2, posterior, n, gamma, down, log_factorial_Y);
              } else {
                delta_prime = delta;
              }
              
              for(int j = 0; j < k; ++j){
                if(abs(gradient(j + 4)) >= 0.00001){
                  beta_prime(j) = PoissonMix_stepsize_for_mutiple_beta(Y, X, Z, Xbeta, alphaXbeta, betas, j, m_gra_beta, ll, psi, delta, mu1, mu2,
                                                                       posterior, n, gamma, down, log_factorial_Y);
                } else {
                  beta_prime(j) = betas(j);
                }
              }
              for(int j = 0; j < k; ++j){
                if(abs(gradient(j + 4 + k)) >= 0.00001){
                  alpha_prime(j) = PoissonMix_stepsize_for_mutiple_alpha(Y, X, Z, Xbeta, Xbeta_mat, alphaXbeta, alphas, j, m_gra_alpha, ll, psi, delta, mu1, mu2,
                                                                         posterior, n, gamma, down, log_factorial_Y);
                  
                } else {
                  alpha_prime(j) = alphas(j);
                }
              }
              
              psi = psi_prime; mu1 = mu1_prime; mu2 = mu2_prime; delta = delta_prime;
              betas = beta_prime; alphas = alpha_prime;
              for(int j = 0; j < k; ++j){
                Xbeta_mat.col(j) = X.col(j)*betas(j);
              }
              
              Xbeta = X*betas;
              arma::vec alpha_beta = alphas%betas;
              alphaXbeta = X*alpha_beta;
              
              gradient = gradient_multiple_beta_all(Y, X, Z, Xbeta, alphaXbeta, betas, psi, delta, mu1, mu2, alphas, posterior, n, k);
              ll = Expected_logLikelihood_multiple_beta(Y, X, Z, Xbeta, psi, delta, mu1, mu2, alphaXbeta, posterior, n, log_factorial_Y);
              
              
              m_gra_beta = gradient(arma::span(4, 3 + k));
              m_gra_alpha = gradient(arma::span(4 + k, 3 + 2*k));
              
            }
            
            arma::vec est = arma::zeros<arma::vec>(4 + 2*k + 1);
            est(0) = psi;
            est(1) = mu1;
            est(2) = mu2;
            est(3) = delta;
            est(arma::span(4, 3 + k)) = betas;
            est(arma::span(4 + k, 3 + 2*k)) = alphas;
            est(4 + 2*k) = ll;
            
            //Rcpp::Rcout  << "est" << est << std::endl;
            return est ;
          }
          
          
          
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          arma::mat Fisher_information_multiple_beta(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, arma::vec betas, arma::vec alphas,
                                                     double psi, double delta, double mu1, double mu2, arma::vec posterior, int n, int k){
            
            arma::mat Fisher = arma::zeros<arma::mat>(4 + 2*k, 4 + 2*k);
            arma::vec e_term_prime = exp(mu1 + alphaXbeta + delta*Z);
            arma::vec e_term_square_inverse_prime = 1/pow(e_term_prime + 1, 2);
            arma::vec a_term = e_term_prime%e_term_square_inverse_prime;
            
            arma::vec Y_psi = Y + psi;
            arma::vec e_term = exp(Xbeta + mu2);
            arma::vec e_term_fac_prime = 1/(e_term + psi);
            arma::vec e_term_fac = e_term%e_term_fac_prime;
            arma::vec e_term_fac_2 = 1/pow(e_term + psi, 2);
            arma::vec e_term_square_inverse = e_term%e_term_fac_2;
            arma::vec Y_i_psi_e_term_psi_inv2 = Y_psi%e_term_square_inverse;
            arma::mat X_square = pow(X, 2);
            
            arma::mat FX = X;
            for(int j = 0; j < k; ++j){
              FX.col(j) = FX.col(j)%posterior;
            }
            
            double common_psi = - 1/psi + R::trigamma(psi);
            
            // 0, 1, 2, 3, 4, ..., 4 + k, ... 4 + 2k -1
            // psi, mu1, mu2, delta, betas, alphas
            
            arma::vec psi2_vec = arma::zeros<arma::vec>(n);
            for(int j = 0; j < n; ++j){
              psi2_vec(j) = - R::trigamma(Y_psi(j)) - Y_psi(j)*e_term_fac_2(j)  + 2*e_term_fac_prime(j);
            }
            
            arma::vec a_term_Z = a_term%Z;
            Fisher(0, 0) = sum(posterior%psi2_vec) + common_psi*sum(posterior); //psi, psi
            Fisher(1, 1) = sum(a_term); //mu1, mu1
            Fisher(1, 3) = Fisher(3, 1) = sum(a_term_Z); //mu1, delta
            Fisher(0, 2) = Fisher(2, 0) = sum(e_term_fac%posterior) - sum(posterior%Y_i_psi_e_term_psi_inv2);  //psi, mu2
            Fisher(2, 2) = psi*sum(posterior%Y_i_psi_e_term_psi_inv2); //mu2, mu2
            Fisher(3, 3) = sum(a_term_Z%Z); //delta, delta
            
            for(int j = 0; j < k; ++j){
              
              arma::vec comm_term = a_term%X.col(j);
              double sum_comm_term_2 = sum(comm_term%X.col(j));
              double sum_comm_term = sum(comm_term);
              double sum_comm_term_Z = sum(comm_term%Z);
              Fisher(0, 4 + j) = Fisher(4 + j, 0) = sum(e_term_fac%FX.col(j)) - sum(FX.col(j)%Y_i_psi_e_term_psi_inv2); // beta, psi
              Fisher(1, 4 + j) = Fisher(4 + j, 1) = alphas(j)*sum_comm_term; //beta, mu1
              Fisher(1, 4 + k + j) = Fisher(4 + k + j, 1) = betas(j)*sum_comm_term; //mu1, alpha_k
              Fisher(2, 4 + j) = Fisher(4 + j, 2) = psi*sum(FX.col(j)%Y_i_psi_e_term_psi_inv2); //beta, mu2
              Fisher(3, 4 + j) = Fisher(4 + j, 3) = alphas(j)*sum_comm_term_Z; //beta, delta
              Fisher(3, 4 + j + k) = Fisher(4 + j + k, 3) = betas(j)*sum_comm_term_Z; //alpha, delta
              
              Fisher(4 + j, 4 + j) = pow(alphas(j), 2)*sum_comm_term_2 + psi*sum(FX.col(j)%Y_i_psi_e_term_psi_inv2%X.col(j)); //beta_k, beta_k
              Fisher(4 + k + j, 4 + k + j) = pow(betas(j), 2)*sum_comm_term_2 ; //alpha_k, alpha_k
              Fisher(4 + j, 4 + k + j) = Fisher(4 + k + j, 4 + j) = alphas(j)*betas(j)*sum_comm_term_2 + sum((1 - 1/(e_term_prime + 1))%X.col(j)) - sum(FX.col(j)); //beta_k, alpha_k
              
              for(int s = 0; s < k; ++s){
                if(s != j){
                  double sumcomm_term_s = alphas(j)*alphas(s)*sum(comm_term%X.col(s));
                  Fisher(4 + j, 4 + s) = Fisher(4 + s, 4 + j) = sumcomm_term_s + psi*sum(FX.col(j)%Y_i_psi_e_term_psi_inv2%X.col(s)); //beta_k, beta_s
                  Fisher(4 + j, 4 + k + s) = Fisher(4 + k + s, 4 + j)= sumcomm_term_s; //beta_k, alpha_s
                  Fisher(4 + k + j, 4 + k + s) = Fisher(4 + k + s, 4 + k + j) = betas(j)*betas(s)*sum(comm_term%X.col(s)); //alpha_k, alpha_s
                }
              }
            }
            
            return Fisher;
          }
          
          
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          
          Rcpp::List EM_PoissonGamma_multiple_beta(arma::vec Y, arma::mat X, arma::vec Z, arma::uvec Y_ind_zero, arma::vec betas, double psi, double mu1, double mu2, double delta,
                                                   arma::vec alphas, arma::vec posterior, double gamma, int steps, int EM_steps, int n, int k, int n_1, double down,
                                                   arma::vec log_factorial_Y, bool ReportAll){
            
            arma::vec est = arma::zeros<arma::vec>(4 + 2*k);
            arma::mat Fisher = arma::zeros<arma::mat>(4 + 2*k, 4 + 2*k);
            arma::mat all_est = arma::zeros<arma::mat>(4 + 2*k, EM_steps);
            arma::mat all_pvalue = arma::zeros<arma::mat>(k, EM_steps);
            arma::mat all_SE = arma::zeros<arma::mat>(k, EM_steps);
            arma::vec Xbeta = X*betas;
            arma::vec alpha_beta = alphas%betas;
            arma::vec alphaXbeta = X*alpha_beta;
            
            for(int i = 0; i < EM_steps; ++i){
              //  Rcpp::Rcout << "i" << i << std::endl;
              arma::vec res = gradient_descent_PoissonGamma_mutiple_beta(Y, X, Z, Xbeta, alphaXbeta, betas, psi, delta, mu1, mu2, alphas, posterior, n, k,
                                                                         gamma, steps, down, log_factorial_Y);
              psi = res(0);
              mu1 = res(1);
              mu2 = res(2);
              delta = res(3);
              betas = res(arma::span(4, 3 + k));
              alphas = res(arma::span(4 + k, 3 + 2*k));
              
              //  Rcpp::Rcout << "2"  << std::endl;
              
              Xbeta = X*betas;
              alpha_beta = alphas%betas;
              alphaXbeta = X*alpha_beta;
              
              for(int j = 0; j < n_1; ++j){
                int id = Y_ind_zero(j);
                posterior(id) = Posterior_prob_multiple_beta(Y(id), Xbeta(id), Z(id), psi, delta, mu1, mu2, alphaXbeta(id), log_factorial_Y(id));
              }
              
              if(ReportAll == FALSE){
                if(i == EM_steps - 1)  {
                  est = res;
                  Fisher = Fisher_information_multiple_beta(Y, X, Z, Xbeta, alphaXbeta, betas, alphas, psi, delta, mu1, mu2, posterior, n, k);
                  arma::mat score = gradient_multiple_beta_all_cov(Y, X, Z, Xbeta, alphaXbeta, betas, psi, delta, mu1, mu2, alphas, posterior, n, k);
                  Fisher -= score;
                }
              } else {
                all_est.col(i) = res;
                Fisher = Fisher_information_multiple_beta(Y, X, Z, Xbeta, alphaXbeta, betas, alphas, psi, delta, mu1, mu2, posterior, n, k);
                arma::mat score = gradient_multiple_beta_all_cov(Y, X, Z, Xbeta, alphaXbeta, betas, psi, delta, mu1, mu2, alphas, posterior, n, k);
                Fisher -= score;
                arma::mat est_cov = arma::inv(Fisher);
                
                for(int j = 0; j < k; ++j){
                  double SE = sqrt(abs(est_cov(4 + j, 4 + j)));
                  all_SE(j, i) = SE;
                  all_pvalue(j, i) = 2*R::pnorm(-abs(betas(j))/SE, 0, 1, TRUE, FALSE);
                }
                
              }
              //    Rcpp::Rcout  << "ll" << i << ":" << res(5) << std::endl;
            }
            
            if(ReportAll == FALSE){
              return Rcpp::List::create(Rcpp::Named("est") = est,
                                        Rcpp::Named("Fisher") = Fisher);
            } else {
              
              return Rcpp::List::create(Rcpp::Named("all_est") = all_est,
                                        Rcpp::Named("all_SE") = all_SE,
                                        Rcpp::Named("all_pvalue") = all_pvalue);
              
            }
            
          }
          
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          Rcpp::List PoissonGamma_Mix_multiple_beta(arma::vec Y,  arma::mat X, arma::vec Z, double psi, double gamma, int steps, int EM_steps, double down, bool group, bool ReportAll){
            
            arma::uvec Y_ind_zero = arma::find(Y == 0);
            
            int n = Y.n_elem;
            int n_1 = Y_ind_zero.n_elem;
            int k = X.n_cols;
            
            arma::vec calculated_values = log_factorial_calculated(Y.max());
            arma::vec log_factorial_Y = arma::zeros<arma::vec>(n);
            for(int i = 0; i < n; ++i){
              log_factorial_Y(i) = calculated_values(Y(i));
            }
            
            arma::vec coef1 = initialize_coef_multiple_beta(Y, X);
            double mu2 = coef1(0);
            arma::vec betas = coef1(arma::span(1, k));
            arma::vec coef2 = arma::zeros<arma::vec>(2 + k);
            
            if(group == TRUE){
              coef2 = initialize_mu_alpha_with_group_multiple_beta(Y, X, betas, Z, X.col(0));
            } else {
              coef2 = initialize_mu_alpha_multiple_beta(Y, X, betas, Z);
            }
            
            double mu1 = coef2(0);
            double delta = 0.1;
            arma::vec alphas = coef2(arma::span(2, 1 + k));
            
            arma::vec Xbeta = X*betas;
            arma::vec alpha_beta = alphas%betas;
            arma::vec alphaXbeta = X*alpha_beta;
            
            arma::vec posterior = arma::ones<arma::vec>(n);
            
            // Rcpp::Rcout << "1" << std::endl;
            for(int j = 0; j < n_1; ++j){
              int id = Y_ind_zero(j);
              posterior(id) = Posterior_prob_multiple_beta(Y(id), Xbeta(id), Z(id), psi, delta, mu1, mu2, alphaXbeta(id), log_factorial_Y(id));
            }
            
            Rcpp::List res = EM_PoissonGamma_multiple_beta(Y, X, Z, Y_ind_zero, betas, psi, mu1, mu2, delta, alphas, posterior, gamma,
                                                           steps, EM_steps, n, k, n_1, down, log_factorial_Y, ReportAll);
            
            if(ReportAll == FALSE){
              arma::mat Fisher = res["Fisher"];
              arma::mat est_cov = arma::inv(Fisher);
              arma::vec est = res["est"];
              betas = est(arma::span(4, 3 + k));
              
              arma::vec SE = arma::zeros<arma::vec>(k);
              arma::vec pvalue = arma::zeros<arma::vec>(k);
              
              for(int j = 0; j < k; ++j){
                SE(j) = sqrt(abs(est_cov(4 + j, 4 + j)));
                pvalue(j) = 2*R::pnorm(-abs(betas(j))/SE(j), 0, 1, TRUE, FALSE);
              }
              
              alphas = est(arma::span(4 + k, 3 + 2*k));
              return Rcpp::List::create(Rcpp::Named("psi") = est(0),
                                        Rcpp::Named("mu1") = est(1),
                                        Rcpp::Named("mu2") = est(2),
                                        Rcpp::Named("delta") = est(3),
                                        Rcpp::Named("betas") = betas,
                                        Rcpp::Named("alphas") = alphas,
                                        Rcpp::Named("likelihood") = est(4 + 2*k),
                                        Rcpp::Named("std") = SE,
                                        Rcpp::Named("pvalue") = pvalue);
              
            } else {
              return Rcpp::List::create(Rcpp::Named("all_est") = res["all_est"],
                                        Rcpp::Named("std") = res["all_SE"],
                                        Rcpp::Named("WT_pvalue") = res["all_pvalue"]);
            }
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          double PoissonMix_stepsize_for_mutiple_alpha_nowith_alpha(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta,  arma::vec alphaXbeta, arma::vec alphas, int ind,  arma::vec m_gra_alpha,
                                                                    double ll, double psi, double delta, double mu1, double mu2, arma::vec posterior, int n, double gamma, double down, arma::vec log_factorial_Y){
            
            double gra_alpha = m_gra_alpha(ind);
            double gra_alpha_2 = gra_alpha*gra_alpha*gamma;
            double alpha = alphas(ind);
            
            arma::vec alphaXbeta_remain = alphaXbeta - X.col(ind)*alpha;
            double start = 0.1;
            if(alpha >= 0.05) start = sqrt(abs(alpha/gra_alpha))/2;
            
            double aa = start;
            double selected = alpha;
            while(aa > 0){
              double aa2 = aa*aa;
              double alpha_prime = alpha + aa2*gra_alpha;
              arma::vec alphaXbeta_prime = alphaXbeta_remain + X.col(ind)*alpha_prime;
              double lalpha_prime = Expected_logLikelihood_multiple_beta(Y, X, Z, Xbeta, psi, delta, mu1, mu2, alphaXbeta_prime, posterior, n, log_factorial_Y);
              if(lalpha_prime - ll - aa2*gra_alpha_2 > 0){
                selected = alpha_prime;
                break;
              }
              aa = aa - start*down;
            }
            
            return selected;
          }
          
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          arma::vec gradient_multiple_beta_all_nowith_alpha(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, arma::vec beta,
                                                            double psi, double delta, double mu1, double mu2, arma::vec alpha, arma::vec posterior, int n, int k){
            
            arma::vec gradient = arma::zeros<arma::vec>(4 + k*2);
            double common_term = log(psi) - R::digamma(psi);
            double sum_posterior = sum(posterior);
            
            for(int i = 0; i < n; ++i){
              
              double e_term = exp(Xbeta(i) + mu2);
              double e_term_psi = e_term + psi;
              double Y_i_psi = Y(i) + psi;
              double e_term_fac = Y_i_psi/e_term_psi;
              double f_term = Y(i) - e_term*e_term_fac;
              
              double e_term_prime = exp(alphaXbeta(i) + mu1 + delta*Z(i));
              double e_term_prime_frac = e_term_prime/(e_term_prime + 1);
              
              double gradient_psi_i =  -log(e_term_psi) - e_term_fac + R::digamma(Y_i_psi);
              gradient(0) += gradient_psi_i*posterior(i); //psi
              double g_term = posterior(i) -e_term_prime_frac;
              gradient(1) += g_term; //mu1
              gradient(2) += f_term*posterior(i); //mu2
              gradient(3) += g_term*Z(i); //delta
              
              for(int j = 0; j < k; ++j){
                double term1 = f_term*posterior(i);//(f_term + alpha(j))*posterior(i);
                //double term2 = e_term_prime_frac;//*alpha(j);
                gradient(j + 4) +=  term1*X(i, j); //- term2*X(i, j); //beta_k
              }
              
              for(int j = 0; j < k; ++j){
                gradient(j + 4 + k) +=  g_term*X(i, j); //alpha_k
              }
              
            }
            gradient(0) += sum_posterior*(common_term + 1);
            
            return gradient;
          }
          
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          arma::mat gradient_multiple_beta_all_cov_nowith_alpha(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, arma::vec beta,
                                                                double psi, double delta, double mu1, double mu2, arma::vec alpha, arma::vec posterior, int n, int k){
            
            arma::mat gradient = arma::zeros<arma::mat>(n, 4 + k*2);
            double common_term = log(psi) - R::digamma(psi);
            
            for(int i = 0; i < n; ++i){
              
              double e_term = exp(Xbeta(i) + mu2);
              double e_term_psi = e_term + psi;
              double Y_i_psi = Y(i) + psi;
              double e_term_fac = Y_i_psi/e_term_psi;
              double f_term = Y(i) - e_term*e_term_fac;
              
              double e_term_prime = exp(alphaXbeta(i) + mu1 + delta*Z(i));
              double e_term_prime_frac = e_term_prime/(e_term_prime + 1);
              
              double gradient_psi_i =  common_term -log(e_term_psi) - e_term_fac + R::digamma(Y_i_psi);
              gradient(i, 0) = gradient_psi_i*posterior(i) + posterior(i); //psi
              double g_term = posterior(i) -e_term_prime_frac;
              gradient(i, 1) = g_term; //mu1
              gradient(i, 2) = f_term*posterior(i); //mu2
              gradient(i, 3) = g_term*Z(i); //delta
              for(int j = 0; j < k; ++j){
                double term1 = f_term*posterior(i);//(f_term + alpha(j))*posterior(i);
                //double term2 = e_term_prime_frac*alpha(j);
                gradient(i, j + 4) =  term1*X(i, j);// - term2*X(i, j); //beta_k
              }
              for(int j = 0; j < k; ++j){
                gradient(i, j + 4 + k) = g_term*X(i, j); //alpha_k
              }
              
            }
            arma::mat est = arma::cov(gradient);
            
            return est;
          }
          
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          arma::vec gradient_descent_PoissonGamma_mutiple_beta_nowith_alpha(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, arma::vec betas, double psi, double delta,
                                                                            double mu1, double mu2, arma::vec alphas, arma::vec posterior, int n, int k,
                                                                            double gamma, int steps, double down, arma::vec log_factorial_Y){
            
            
            arma::vec gradient = gradient_multiple_beta_all_nowith_alpha(Y, X, Z, Xbeta, alphaXbeta, betas, psi, delta, mu1, mu2, alphas, posterior, n, k);
            double ll = Expected_logLikelihood_multiple_beta(Y, X, Z, Xbeta, psi, delta, mu1, mu2, alphaXbeta, posterior, n, log_factorial_Y);
            double psi_prime = 0; double mu1_prime = 0; double mu2_prime = 0; double delta_prime = 0;
            arma::vec beta_prime = arma::zeros<arma::vec>(k); arma::vec alpha_prime = arma::zeros<arma::vec>(k);
            arma::vec m_gra_beta = gradient(arma::span(4, 3 + k));
            arma::vec m_gra_alpha = gradient(arma::span(4 + k, 3 + 2*k));
            
            for(int i = 0; i < steps; ++i){
              
              if(abs(gradient(0)) >= 0.00001){
                psi_prime = PoissonMix_stepsize_mutiple_beta_for_psi(Y, X, Z, Xbeta, alphaXbeta, gradient(0), ll, psi, delta, mu1, mu2, posterior, n, gamma, down, log_factorial_Y);
              } else {
                psi_prime = psi;
              }
              if(abs(gradient(1)) >= 0.00001){
                mu1_prime = PoissonMix_stepsize_mutiple_beta_for_mu1(Y, X, Z, Xbeta, alphaXbeta, gradient(1), ll, psi, delta, mu1, mu2, posterior, n, gamma, down, log_factorial_Y);
              } else {
                mu1_prime = mu1;
              }
              if(abs(gradient(2)) >= 0.00001){
                mu2_prime = PoissonMix_stepsize_mutiple_beta_for_mu2(Y, X, Z, Xbeta, alphaXbeta, gradient(2), ll, psi, delta, mu1, mu2, posterior, n, gamma, down, log_factorial_Y);
              } else {
                mu2_prime = mu2;
              }
              if(abs(gradient(3)) >= 0.00001){
                delta_prime = PoissonMix_stepsize_mutiple_beta_for_delta(Y, X, Z, Xbeta, alphaXbeta, gradient(3), ll, psi, delta, mu1, mu2, posterior, n, gamma, down, log_factorial_Y);
              } else {
                delta_prime = delta;
              }
              
              for(int j = 0; j < k; ++j){
                if(abs(gradient(j + 4)) >= 0.00001){
                  beta_prime(j) = PoissonMix_stepsize_for_mutiple_beta(Y, X, Z, Xbeta, alphaXbeta, betas, j, m_gra_beta, ll, psi, delta, mu1, mu2,
                                                                       posterior, n, gamma, down, log_factorial_Y);
                } else {
                  beta_prime(j) = betas(j);
                }
              }
              for(int j = 0; j < k; ++j){
                if(abs(gradient(j + 4 + k)) >= 0.00001){
                  alpha_prime(j) = PoissonMix_stepsize_for_mutiple_alpha_nowith_alpha(Y, X, Z, Xbeta, alphaXbeta, alphas, j, m_gra_alpha, ll, psi, delta, mu1, mu2,
                                                                                      posterior, n, gamma, down, log_factorial_Y);
                  
                } else {
                  alpha_prime(j) = alphas(j);
                }
              }
              
              psi = psi_prime; mu1 = mu1_prime; mu2 = mu2_prime; delta = delta_prime;
              betas = beta_prime; alphas = alpha_prime;
              
              Xbeta = X*betas;
              arma::vec alpha_beta = alphas;
              alphaXbeta = X*alpha_beta;
              
              gradient = gradient_multiple_beta_all_nowith_alpha(Y, X, Z, Xbeta, alphaXbeta, betas, psi, delta, mu1, mu2, alphas, posterior, n, k);
              ll = Expected_logLikelihood_multiple_beta(Y, X, Z, Xbeta, psi, delta, mu1, mu2, alphaXbeta, posterior, n, log_factorial_Y);
              
              
              m_gra_beta = gradient(arma::span(4, 3 + k));
              m_gra_alpha = gradient(arma::span(4 + k, 3 + 2*k));
              
            }
            
            arma::vec est = arma::zeros<arma::vec>(4 + 2*k + 1);
            est(0) = psi;
            est(1) = mu1;
            est(2) = mu2;
            est(3) = delta;
            est(arma::span(4, 3 + k)) = betas;
            est(arma::span(4 + k, 3 + 2*k)) = alphas;
            est(4 + 2*k) = ll;
            
            //Rcpp::Rcout  << "est" << est << std::endl;
            return est ;
          }
          
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          arma::mat Fisher_information_multiple_beta_nowith_alpha(arma::vec Y, arma::mat X, arma::vec Z, arma::vec Xbeta, arma::vec alphaXbeta, arma::vec betas, arma::vec alphas,
                                                                  double psi, double delta, double mu1, double mu2, arma::vec posterior, int n, int k){
            
            arma::mat Fisher = arma::zeros<arma::mat>(4 + 2*k, 4 + 2*k);
            arma::vec e_term_prime = exp(mu1 + alphaXbeta + delta*Z);
            arma::vec e_term_square_inverse_prime = 1/pow(e_term_prime + 1, 2);
            arma::vec a_term = e_term_prime%e_term_square_inverse_prime;
            
            arma::vec Y_psi = Y + psi;
            arma::vec e_term = exp(Xbeta + mu2);
            arma::vec e_term_fac_prime = 1/(e_term + psi);
            arma::vec e_term_fac = e_term%e_term_fac_prime;
            arma::vec e_term_fac_2 = 1/pow(e_term + psi, 2);
            arma::vec e_term_square_inverse = e_term%e_term_fac_2;
            arma::vec Y_i_psi_e_term_psi_inv2 = Y_psi%e_term_square_inverse;
            arma::mat X_square = pow(X, 2);
            
            arma::mat FX = X;
            for(int j = 0; j < k; ++j){
              FX.col(j) = FX.col(j)%posterior;
            }
            
            double common_psi = - 1/psi + R::trigamma(psi);
            
            // 0, 1, 2, 3, 4, ..., 4 + k, ... 4 + 2k -1
            // psi, mu1, mu2, delta, betas, alphas
            
            arma::vec psi2_vec = arma::zeros<arma::vec>(n);
            for(int j = 0; j < n; ++j){
              psi2_vec(j) = - R::trigamma(Y_psi(j)) - Y_psi(j)*e_term_fac_2(j)  + 2*e_term_fac_prime(j);
            }
            
            arma::vec a_term_Z = a_term%Z;
            Fisher(0, 0) = sum(posterior%psi2_vec) + common_psi*sum(posterior); //psi, psi
            Fisher(1, 1) = sum(a_term); //mu1, mu1
            Fisher(1, 3) = Fisher(3, 1) = sum(a_term_Z); //mu1, delta
            Fisher(0, 2) = Fisher(2, 0) = sum(e_term_fac%posterior) - sum(posterior%Y_i_psi_e_term_psi_inv2);  //psi, mu2
            Fisher(2, 2) = psi*sum(posterior%Y_i_psi_e_term_psi_inv2); //mu2, mu2
            Fisher(3, 3) = sum(a_term_Z%Z); //delta, delta
            
            for(int j = 0; j < k; ++j){
              
              arma::vec comm_term = a_term%X.col(j);
              double sum_comm_term_2 = sum(comm_term%X.col(j));
              double sum_comm_term = sum(comm_term);
              double sum_comm_term_Z = sum(comm_term%Z);
              Fisher(0, 4 + j) = Fisher(4 + j, 0) = sum(e_term_fac%FX.col(j)) - sum(FX.col(j)%Y_i_psi_e_term_psi_inv2); // beta, psi
              // Fisher(1, 4 + j) = Fisher(4 + j, 1) = 0; //beta, mu1
              Fisher(1, 4 + k + j) = Fisher(4 + k + j, 1) = sum_comm_term; //mu1, alpha_k
              Fisher(2, 4 + j) = Fisher(4 + j, 2) = psi*sum(FX.col(j)%Y_i_psi_e_term_psi_inv2); //beta, mu2
              // Fisher(3, 4 + j) = Fisher(4 + j, 3) = 0; //beta, delta
              Fisher(3, 4 + j + k) = Fisher(4 + j + k, 3) = sum_comm_term_Z; //alpha, delta
              
              Fisher(4 + j, 4 + j) =  psi*sum(FX.col(j)%Y_i_psi_e_term_psi_inv2%X.col(j)); //beta_k, beta_k
              Fisher(4 + k + j, 4 + k + j) = sum_comm_term_2 ;  //alpha_k, alpha_k
              // Fisher(4 + j, 4 + k + j) = Fisher(4 + k + j, 4 + j) = 0;//beta_k, alpha_k
              
              for(int s = 0; s < k; ++s){
                if(s != j){
                  Fisher(4 + j, 4 + s) = Fisher(4 + s, 4 + j) =  psi*sum(FX.col(j)%Y_i_psi_e_term_psi_inv2%X.col(s)); //beta_k, beta_s
                  Fisher(4 + k + j, 4 + k + s) = Fisher(4 + k + s, 4 + k + j) = sum(comm_term%X.col(s)); //alpha_k, alpha_s
                  //  Fisher(4 + j, 4 + k + s) = Fisher(4 + k + s, 4 + j)= 0; //beta_k, alpha_s
                }
              }
            }
            
            return Fisher;
          }
          
          
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          
          Rcpp::List EM_PoissonGamma_multiple_beta_nowith_alpha(arma::vec Y, arma::mat X, arma::vec Z, arma::uvec Y_ind_zero, arma::vec betas, double psi, double mu1, double mu2, double delta,
                                                                arma::vec alphas, arma::vec posterior, double gamma, int steps, int EM_steps, int n, int k, int n_1, double down,
                                                                arma::vec log_factorial_Y, bool ReportAll){
            
            arma::vec est = arma::zeros<arma::vec>(4 + 2*k);
            arma::mat Fisher = arma::zeros<arma::mat>(4 + 2*k, 4 + 2*k);
            arma::mat all_est = arma::zeros<arma::mat>(4 + 2*k, EM_steps);
            arma::mat all_pvalue = arma::zeros<arma::mat>(k, EM_steps);
            arma::mat all_SE = arma::zeros<arma::mat>(k, EM_steps);
            arma::vec Xbeta = X*betas;
            arma::vec alpha_beta = alphas;
            arma::vec alphaXbeta = X*alpha_beta;
            
            for(int i = 0; i < EM_steps; ++i){
              //   Rcpp::Rcout << "i" << i << std::endl;
              arma::vec res = gradient_descent_PoissonGamma_mutiple_beta_nowith_alpha(Y, X, Z, Xbeta, alphaXbeta, betas, psi, delta, mu1, mu2, alphas, posterior, n, k,
                                                                                      gamma, steps, down, log_factorial_Y);
              psi = res(0);
              mu1 = res(1);
              mu2 = res(2);
              delta = res(3);
              betas = res(arma::span(4, 3 + k));
              alphas = res(arma::span(4 + k, 3 + 2*k));
              
              //  Rcpp::Rcout << "2"  << std::endl;
              
              Xbeta = X*betas;
              alpha_beta = alphas;
              alphaXbeta = X*alpha_beta;
              
              for(int j = 0; j < n_1; ++j){
                int id = Y_ind_zero(j);
                posterior(id) = Posterior_prob_multiple_beta(Y(id), Xbeta(id), Z(id), psi, delta, mu1, mu2, alphaXbeta(id), log_factorial_Y(id));
              }
              
              if(ReportAll == FALSE){
                if(i == EM_steps - 1)  {
                  est = res;
                  Fisher = Fisher_information_multiple_beta_nowith_alpha(Y, X, Z, Xbeta, alphaXbeta, betas, alphas, psi, delta, mu1, mu2, posterior, n, k);
                  arma::mat score = gradient_multiple_beta_all_cov_nowith_alpha(Y, X, Z, Xbeta, alphaXbeta, betas, psi, delta, mu1, mu2, alphas, posterior, n, k);
                  Fisher -= score;
                }
              } else {
                all_est.col(i) = res;
                Fisher = Fisher_information_multiple_beta_nowith_alpha(Y, X, Z, Xbeta, alphaXbeta, betas, alphas, psi, delta, mu1, mu2, posterior, n, k);
                arma::mat score = gradient_multiple_beta_all_cov_nowith_alpha(Y, X, Z, Xbeta, alphaXbeta, betas, psi, delta, mu1, mu2, alphas, posterior, n, k);
                Fisher -= score;
                arma::mat est_cov = arma::inv(Fisher);
                
                for(int j = 0; j < k; ++j){
                  double SE = sqrt(abs(est_cov(4 + j, 4 + j)));
                  all_SE(j, i) = SE;
                  all_pvalue(j, i) = 2*R::pnorm(-abs(betas(j))/SE, 0, 1, TRUE, FALSE);
                }
                
              }
              //    Rcpp::Rcout  << "ll" << i << ":" << res(5) << std::endl;
            }
            
            if(ReportAll == FALSE){
              return Rcpp::List::create(Rcpp::Named("est") = est,
                                        Rcpp::Named("Fisher") = Fisher);
            } else {
              
              return Rcpp::List::create(Rcpp::Named("all_est") = all_est,
                                        Rcpp::Named("all_SE") = all_SE,
                                        Rcpp::Named("all_pvalue") = all_pvalue);
              
            }
            
          }
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          Rcpp::List PoissonGamma_Mix_multiple_beta_nowith_alpha(arma::vec Y,  arma::mat X, arma::vec Z, double psi, double gamma, int steps, int EM_steps, double down, bool group, bool ReportAll){
            
            arma::uvec Y_ind_zero = arma::find(Y == 0);
            
            int n = Y.n_elem;
            int n_1 = Y_ind_zero.n_elem;
            int k = X.n_cols;
            
            arma::vec calculated_values = log_factorial_calculated(Y.max());
            arma::vec log_factorial_Y = arma::zeros<arma::vec>(n);
            for(int i = 0; i < n; ++i){
              log_factorial_Y(i) = calculated_values(Y(i));
            }
            
            arma::vec coef1 = initialize_coef_multiple_beta(Y, X);
            double mu2 = coef1(0);
            arma::vec betas = coef1(arma::span(1, k));
            arma::vec coef2 = arma::zeros<arma::vec>(2 + k);
            
            if(group == TRUE){
              coef2 = initialize_mu_alpha_with_group_multiple_beta_nowith_alpha(Y, X, Z, X.col(0));
            } else {
              coef2 = initialize_mu_alpha_multiple_beta_nowith_alpha(Y, X, Z);
            }
            //Rcpp::Rcout << "1" << std::endl;
            double mu1 = coef2(0);
            double delta = 0.1;
            arma::vec alphas = coef2(arma::span(2, 1 + k));
            
            arma::vec Xbeta = X*betas;
            arma::vec alpha_beta = alphas;
            arma::vec alphaXbeta = X*alpha_beta;
            
            arma::vec posterior = arma::ones<arma::vec>(n);
            
            
            for(int j = 0; j < n_1; ++j){
              int id = Y_ind_zero(j);
              posterior(id) = Posterior_prob_multiple_beta(Y(id), Xbeta(id), Z(id), psi, delta, mu1, mu2, alphaXbeta(id), log_factorial_Y(id));
            }
            
            Rcpp::List res = EM_PoissonGamma_multiple_beta_nowith_alpha(Y, X, Z, Y_ind_zero, betas, psi, mu1, mu2, delta, alphas, posterior, gamma,
                                                                        steps, EM_steps, n, k, n_1, down, log_factorial_Y, ReportAll);
            
            
            if(ReportAll == FALSE){
              arma::mat Fisher = res["Fisher"];
              arma::mat est_cov = arma::inv(Fisher);
              arma::vec est = res["est"];
              betas = est(arma::span(4, 3 + k));
              
              arma::vec SE = arma::zeros<arma::vec>(k);
              arma::vec pvalue = arma::zeros<arma::vec>(k);
              
              for(int j = 0; j < k; ++j){
                SE(j) = sqrt(abs(est_cov(4 + j, 4 + j)));
                pvalue(j) = 2*R::pnorm(-abs(betas(j))/SE(j), 0, 1, TRUE, FALSE);
              }
              
              alphas = est(arma::span(4 + k, 3 + 2*k));
              return Rcpp::List::create(Rcpp::Named("psi") = est(0),
                                        Rcpp::Named("mu1") = est(1),
                                        Rcpp::Named("mu2") = est(2),
                                        Rcpp::Named("delta") = est(3),
                                        Rcpp::Named("betas") = betas,
                                        Rcpp::Named("alphas") = alphas,
                                        Rcpp::Named("likelihood") = est(4 + 2*k),
                                        Rcpp::Named("std") = SE,
                                        Rcpp::Named("pvalue") = pvalue);
              
            } else {
              return Rcpp::List::create(Rcpp::Named("all_est") = res["all_est"],
                                        Rcpp::Named("std") = res["all_SE"],
                                        Rcpp::Named("WT_pvalue") = res["all_pvalue"]);
            }
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          Rcpp::List Mix_gradient_and_LogLikelihood_for_individual_sample(arma::vec Y, arma::vec W, arma::vec V, arma::vec WY, arma::vec WWY, arma::vec W3Y, arma::vec W4Y,
                                                                          arma::vec VY, arma::vec VVY, arma::vec V3Y, arma::vec V4Y,
                                                                          arma::vec WW, arma::vec W3, arma::vec W4, arma::vec VV, arma::vec V3, arma::vec V4,
                                                                          double a0, double a1, double a2, double a3, double a4,
                                                                          double b1, double b2, double b3, double b4, double psi, int n, double sum_log_factorial_Y){
            
            arma::vec gradient = arma::zeros<arma::vec>(10);
            double Likelihood = 0;
            double common_term = -lgamma(psi) + psi*log(psi);
            double common_term2 = log(psi) - R::digamma(psi);
            
            for(int i = 0; i < n; ++i){
              
              double A_i = a0 + a1*W(i) + a2*WW(i) + a3*W3(i) + a4*W4(i);
              double B_i = b1*V(i) + b2*VV(i) + b3*V3(i) + b4*V4(i);
              double exp_Ai = exp(A_i);
              double pr =  exp_Ai*B_i;
              double ll =  pr + psi;
              double psi_Yi = psi + Y(i);
              Likelihood += lgamma(psi_Yi) - psi_Yi*log(ll) + Y(i)*(A_i + log(B_i));
              
              double term1 = -pr*psi_Yi/ll;
              gradient(0) += term1 + Y(i);
              gradient(1) += term1*W(i) + WY(i);
              gradient(2) += term1*WW(i) + WWY(i);
              gradient(3) += term1*W3(i) + W3Y(i);
              gradient(4) += term1*W4(i) + W4Y(i);
              
              double term2 = 1/B_i;
              gradient(5) += (term1*V(i) + VY(i))*term2;
              gradient(6) += (term1*VV(i) + VVY(i))*term2;
              gradient(7) += (term1*V3(i) + V3Y(i))*term2;
              gradient(8) += (term1*V4(i) + V4Y(i))*term2;
              gradient(9) += - psi_Yi/ll - log(ll) + R::digamma(psi_Yi);
            }
            
            gradient(9) += n*common_term2 + n;
            Likelihood -= sum_log_factorial_Y;
            Likelihood += n*common_term;
            return Rcpp::List::create(Rcpp::Named("gradient") = gradient,
                                      Rcpp::Named("Likelihood") = Likelihood);
            
          }
          
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          
          
          double Mix_LogLikelihood_for_individual_sample(arma::vec Y, arma::vec W, arma::vec V, arma::vec WW, arma::vec VV,
                                                         arma::vec W3, arma::vec V3, arma::vec W4, arma::vec V4,
                                                         double a0, double a1, double a2, double a3, double a4,
                                                         double b1, double b2, double b3, double b4, double psi,
                                                         int n, double sum_log_factorial_Y){
            
            double Likelihood = 0;
            double common_term = -lgamma(psi) + psi*log(psi);
            for(int i = 0; i < n; ++i){
              
              double A_i = a0 + a1*W(i) + a2*WW(i) + a3*W3(i) + a4*W4(i);
              double B_i = b1*V(i) + b2*VV(i) + b3*V3(i) + b4*V4(i);
              double exp_Ai = exp(A_i);
              double pr =  exp_Ai*B_i;
              double ll =  pr + psi;
              double psi_Yi = psi + Y(i);
              Likelihood += lgamma(psi_Yi) - psi_Yi*log(ll) + Y(i)*(A_i + log(B_i));
              
            }
            
            Likelihood -= sum_log_factorial_Y;
            Likelihood += n*common_term;
            return Likelihood;
          }
          
          
          // [[Rcpp::depends("RcppArmadillo")]]
          // [[Rcpp::export]]
          double Mix_select_stepsize_for_a_parameter(arma::vec Y, arma::vec W, arma::vec V, arma::vec WW, arma::vec VV,
                                                     arma::vec W3, arma::vec V3, arma::vec W4, arma::vec V4, double ll, double sum_log_factorial_Y,
                                                     arma::vec gradient, arma::vec parameters, int ind, double gamma, int n, double down){
            
            //if(ind < 5) gamma = 0.75;
            double gra = gradient(ind);
            double gra_2 = gra*gra*gamma;
            double para = parameters(ind);
            double start = sqrt(abs(para/gra))/5;
            //if(ind < 5) start = start/10;
            double a0 = parameters(0);
            double a1 = parameters(1);
            double a2 = parameters(2);
            double a3 = parameters(3);
            double a4 = parameters(4);
            double b1 = parameters(5);
            double b2 = parameters(6);
            double b3 = parameters(7);
            double b4 = parameters(8);
            double psi = parameters(9);
            
            double aa = start;
            double selected = para;
            double ll_prime = ll;
            while(aa > 0){
              double aa2 = aa*aa;
              double para_prime = para + aa2*gra;
              if(ind == 0){
                ll_prime = Mix_LogLikelihood_for_individual_sample(Y, W, V, WW, VV, W3, V3, W4, V4, para_prime, a1, a2, a3, a4, b1, b2, b3, b4, psi, n, sum_log_factorial_Y);
              }
              if(ind == 1){
                ll_prime = Mix_LogLikelihood_for_individual_sample(Y, W, V, WW, VV, W3, V3, W4, V4, a0, para_prime, a2, a3, a4, b1, b2, b3, b4, psi, n, sum_log_factorial_Y);
              }
              if(ind == 2){
                ll_prime = Mix_LogLikelihood_for_individual_sample(Y, W, V, WW, VV, W3, V3, W4, V4, a0, a1, para_prime, a3, a4, b1, b2, b3, b4, psi, n, sum_log_factorial_Y);
              }
              if(ind == 3){
                ll_prime = Mix_LogLikelihood_for_individual_sample(Y, W, V, WW, VV, W3, V3, W4, V4, a0, a1, a2, para_prime, a4, b1, b2, b3, b4, psi, n, sum_log_factorial_Y);
              }
              if(ind == 4){
                ll_prime = Mix_LogLikelihood_for_individual_sample(Y, W, V, WW, VV, W3, V3, W4, V4, a0, a1, a2, a3, para_prime, b1, b2, b3, b4, psi, n, sum_log_factorial_Y);
              }
              if(ind == 5){
                ll_prime = Mix_LogLikelihood_for_individual_sample(Y, W, V, WW, VV, W3, V3, W4, V4, a0, a1, a2, a3, a4, para_prime, b2, b3, b4, psi, n, sum_log_factorial_Y);
              }
              if(ind == 6){
                ll_prime = Mix_LogLikelihood_for_individual_sample(Y, W, V, WW, VV, W3, V3, W4, V4, a0, a1, a2, a3, a4, b1, para_prime, b3, b4, psi, n, sum_log_factorial_Y);
              }
              if(ind == 7){
                ll_prime = Mix_LogLikelihood_for_individual_sample(Y, W, V, WW, VV, W3, V3, W4, V4, a0, a1, a2, a3, a4, b1, b2, para_prime, b4, psi, n, sum_log_factorial_Y);
              }
              if(ind == 8){
                ll_prime = Mix_LogLikelihood_for_individual_sample(Y, W, V, WW, VV, W3, V3, W4, V4, a0, a1, a2, a3, a4, b1, b2, b3, para_prime, psi, n, sum_log_factorial_Y);
              }
              if(ind == 9){
                ll_prime = Mix_LogLikelihood_for_individual_sample(Y, W, V, WW, VV, W3, V3, W4, V4, a0, a1, a2, a3, a4, b1, b2, b3, b4, para_prime, n, sum_log_factorial_Y);
              }
              if(ll_prime - ll - aa2*gra_2 > 0 ) { //| abs(ll_prime - ll - aa2*gra_2) < 0.0001) {
                selected = para_prime;
                break;
              }
aa = aa - start*down;
              }
              
              return selected;
            }
            
            
            
            // [[Rcpp::depends("RcppArmadillo")]]
            // [[Rcpp::export]]
            
            Rcpp::List Mix_gradient_descent_for_individual_sample(arma::vec Y, arma::vec W, arma::vec V, double a0, double a1, double a2, double a3, double a4,
                                                                  double b1, double b2, double b3, double b4, double psi, double gamma, int steps, double down){
              
              int n = Y.n_elem;
              
              arma::vec calculated_values = log_factorial_calculated(Y.max());
              arma::vec log_factorial_Y = arma::zeros<arma::vec>(n);
              for(int i = 0; i < n; ++i){
                log_factorial_Y(i) = calculated_values(Y(i));
              }
              
              double sum_log_factorial_Y = sum(log_factorial_Y);
              
              arma::vec WW = W%W;
              arma::vec W3 = W%WW;
              arma::vec W4 = W%W3;
              arma::vec WY = W%Y;
              arma::vec WWY = W%WY;
              arma::vec W3Y = W%WWY;
              arma::vec W4Y = W%W3Y;
              
              arma::vec VV = V%V;
              arma::vec V3 = V%VV;
              arma::vec V4 = V%V3;
              arma::vec VY = V%Y;
              arma::vec VVY = V%VY;
              arma::vec V3Y = V%VVY;
              arma::vec V4Y = V%V3Y;
              Rcpp::List res = Mix_gradient_and_LogLikelihood_for_individual_sample(Y, W, V, WY, WWY, W3Y, W4Y, VY, VVY, V3Y, V4Y,
                                                                                    WW, W3, W4, VV, V3, V4, a0, a1, a2, a3, a4, b1, b2, b3, b4, psi, n, sum_log_factorial_Y);
              arma::vec gradient = res["gradient"];
              double ll = res["Likelihood"];
              
              arma::vec parameters = arma::zeros<arma::vec>(10);
              parameters(0) = a0;
              parameters(1) = a1;
              parameters(2) = a2;
              parameters(3) = a3;
              parameters(4) = a4;
              parameters(5) = b1;
              parameters(6) = b2;
              parameters(7) = b3;
              parameters(8) = b4;
              parameters(9) = psi;
              
              double a0_prime = 0; double a1_prime = 0; double a2_prime = 0; double a3_prime = 0; double a4_prime = 0;
              double b1_prime = 0; double b2_prime = 0; double b3_prime = 0; double b4_prime = 0; double psi_prime = 1;
              
              for(int i = 0; i < steps; ++i){
                if(abs(gradient(0)) >= 0.0001){
                  a0_prime = Mix_select_stepsize_for_a_parameter(Y, W, V, WW, VV, W3, V3, W4, V4, ll, sum_log_factorial_Y, gradient, parameters, 0, gamma, n, down);
                } else {
                  a0_prime = a0;
                }
                if(abs(gradient(1)) >= 0.0001){
                  a1_prime = Mix_select_stepsize_for_a_parameter(Y, W, V, WW, VV, W3, V3, W4, V4, ll, sum_log_factorial_Y, gradient, parameters, 1, gamma, n, down);
                } else {
                  a1_prime = a1;
                }
                if(abs(gradient(2)) >= 0.0001){
                  a2_prime = Mix_select_stepsize_for_a_parameter(Y, W, V, WW, VV, W3, V3, W4, V4, ll, sum_log_factorial_Y,  gradient, parameters, 2, gamma, n, down);
                } else {
                  a2_prime = a2;
                }
                if(abs(gradient(3)) >= 0.0001){
                  a3_prime = Mix_select_stepsize_for_a_parameter(Y, W, V, WW, VV, W3, V3, W4, V4, ll, sum_log_factorial_Y, gradient, parameters, 3, gamma, n, down);
                } else {
                  a3_prime = a3;
                }
                if(abs(gradient(4)) >= 0.0001){
                  a4_prime = Mix_select_stepsize_for_a_parameter(Y, W, V, WW, VV, W3, V3, W4, V4, ll, sum_log_factorial_Y, gradient, parameters, 4, gamma, n, down);
                } else {
                  a4_prime = a4;
                }
                if(abs(gradient(5)) >= 0.0001){
                  b1_prime = Mix_select_stepsize_for_a_parameter(Y, W, V, WW, VV, W3, V3, W4, V4, ll, sum_log_factorial_Y, gradient, parameters, 5, gamma, n, down);
                } else {
                  b1_prime = b1;
                }
                if(abs(gradient(6)) >= 0.0001){
                  b2_prime = Mix_select_stepsize_for_a_parameter(Y, W, V, WW, VV, W3, V3, W4, V4, ll, sum_log_factorial_Y,  gradient, parameters, 6, gamma, n, down);
                } else {
                  b2_prime = b2;
                }
                if(abs(gradient(7)) >= 0.0001){
                  b3_prime = Mix_select_stepsize_for_a_parameter(Y, W, V, WW, VV, W3, V3, W4, V4, ll, sum_log_factorial_Y, gradient, parameters, 7, gamma, n, down);
                } else {
                  b3_prime = b3;
                }
                if(abs(gradient(8)) >= 0.0001){
                  b4_prime = Mix_select_stepsize_for_a_parameter(Y, W, V, WW, VV, W3, V3, W4, V4, ll, sum_log_factorial_Y, gradient, parameters, 8, gamma, n, down);
                } else {
                  b4_prime = b4;
                }
                if(abs(gradient(9)) >= 0.0001){
                  psi_prime = Mix_select_stepsize_for_a_parameter(Y, W, V, WW, VV, W3, V3, W4, V4, ll, sum_log_factorial_Y, gradient, parameters, 9, gamma, n, down);
                } else {
                  psi_prime = psi;
                }
                
                a0 = a0_prime; a1 = a1_prime; a2 = a2_prime; a3 = a3_prime; a4 = a4_prime;
                b1 = b1_prime; b2 = b2_prime; b3 = b3_prime; b4 = b4_prime; psi = psi_prime;
                
                Rcpp::List res = Mix_gradient_and_LogLikelihood_for_individual_sample(Y, W, V, WY, WWY, W3Y, W4Y, VY, VVY, V3Y, V4Y,
                                                                                      WW, W3, W4, VV, V3, V4, a0, a1, a2, a3, a4, b1, b2, b3, b4, psi, n, sum_log_factorial_Y);
                arma::vec gradient_1 = res["gradient"];
                gradient = gradient_1;
                ll = res["Likelihood"];
                
                parameters(0) = a0;
                parameters(1) = a1;
                parameters(2) = a2;
                parameters(3) = a3;
                parameters(4) = a4;
                parameters(5) = b1;
                parameters(6) = b2;
                parameters(7) = b3;
                parameters(8) = b4;
                parameters(9) = psi;
                
                // Rcpp::Rcout << gradient << std::endl;
              }
              
              arma::vec corrected = Y;
              for(int i = 0; i < n; ++i){
                
                double A_i = a0 + a1*W(i) + a2*WW(i) + a3*W3(i) + a4*W4(i);
                double B_i = b1*V(i) + b2*VV(i) + b3*V3(i) + b4*V4(i);
                corrected(i) = B_i*exp(A_i);
              }
              
              return Rcpp::List::create(Rcpp::Named("parameters") = parameters,
                                        Rcpp::Named("corrected") = corrected);
            }
            
            
            // [[Rcpp::depends("RcppArmadillo")]]
            // [[Rcpp::export]]
            
            arma::vec Predict_for_individual_sample(arma::vec W, arma::vec V, double a0, double a1, double a2, double a3, double a4,
                                                    double b1, double b2, double b3, double b4){
              
              arma::vec WW = W%W;
              arma::vec W3 = W%WW;
              arma::vec W4 = W%W3;
              arma::vec VV = V%V;
              arma::vec V3 = V%VV;
              arma::vec V4 = V%V3;
              
              arma::vec corrected = W;
              int n = W.n_elem;
              for(int i = 0; i < n; ++i){
                
                double A_i = a0 + a1*W(i) + a2*WW(i) + a3*W3(i) + a4*W4(i);
                double B_i = b1*V(i) + b2*VV(i) + b3*V3(i) + b4*V4(i);
                corrected(i) = B_i*exp(A_i);
              }
              
              return(corrected);
              
            }
            
            
            
            // [[Rcpp::depends("RcppArmadillo")]]
            // [[Rcpp::export]]
            
            arma::vec reweighting_sum_C(arma::mat Ymat, arma::mat Yflagmat, arma::vec Y, arma::vec Yflag, arma::vec prior_weight, bool ImputeAll){
              
              int p = Ymat.n_cols;
              int k = Ymat.n_rows;
              double k_cut = 0.9*k;
              
              arma::vec res = Y;
              arma::mat Ymat_prod = Ymat%Yflagmat;
              
              if(ImputeAll)  {
                
                
                for(int i = 0; i < p; ++i){
                  
                  arma::vec Ymat_weights = arma::ones<arma::vec>(k);
                  
                  arma::uvec Y_ind_zero = arma::find(Yflagmat.col(i) == 0);
                  double zero_num = Y_ind_zero.n_elem;
                  double nonzero_num = k - zero_num;
                  
                  if(zero_num < k_cut & zero_num != 0){
                    
                    double mean_Y = sum(Ymat_prod.col(i))/nonzero_num;
                    double var_Y = 1;
                    if(nonzero_num != 1){
                      var_Y = sum(Ymat_prod.col(i)%Ymat_prod.col(i))/nonzero_num - mean_Y*mean_Y;
                      var_Y *= nonzero_num/(nonzero_num-1);
                    }
                    if(var_Y == 0) {
                      var_Y = 1;
                    }
                    double pN = R::dnorm(0, mean_Y, sqrt(var_Y), FALSE);
                    double non_dropout = nonzero_num*pN/(zero_num + nonzero_num*pN);
                    for(int j = 0; j < zero_num; ++j){
                      int zero_id = Y_ind_zero(j);
                      Ymat_weights(zero_id) = non_dropout;
                    }
                    Ymat_weights %= prior_weight;
                    Ymat_weights /= sum(Ymat_weights);
                    
                  } else {
                    
                    Ymat_weights = prior_weight;
                    
                  }
                  res(i) = sum(Ymat_weights%Ymat.col(i));
                }
                
              } else {
                
                arma::uvec to_impute = arma::find(Yflag == 0);
                double to_impute_num = to_impute.n_elem;
                //
                  for(int i = 0; i < to_impute_num; ++i){
                    
                    arma::vec Ymat_weights = prior_weight;
                    
                    int id = to_impute(i);
                    
                    arma::uvec Y_ind_zero = arma::find(Yflagmat.col(id) == 0);
                    double zero_num = Y_ind_zero.n_elem;
                    double nonzero_num = k - zero_num;
                    
                    if(zero_num < k_cut & zero_num != 0){
                      
                      double mean_Y = sum(Ymat_prod.col(id))/nonzero_num;
                      double var_Y = 1;
                      if(nonzero_num != 1){
                        var_Y = sum(Ymat_prod.col(i)%Ymat_prod.col(i))/nonzero_num - mean_Y*mean_Y;
                        var_Y *= nonzero_num/(nonzero_num-1);
                      }
                      if(var_Y == 0) {
                        var_Y = 1;
                      }
                      double pN = R::dnorm(0, mean_Y, sqrt(var_Y), FALSE);
                      double non_dropout = nonzero_num*pN/(zero_num + nonzero_num*pN);
                      
                      for(int j = 0; j < zero_num; ++j){
                        int zero_id = Y_ind_zero(j);
                        Ymat_weights(zero_id) = prior_weight(zero_id)*non_dropout;
                      }
                      
                      Ymat_weights = Ymat_weights/sum(Ymat_weights);
                    }
                    res(id) = sum(Ymat_weights%Ymat.col(id));
                  }
                
              }
              return res;
              
            }
            
            
            
            
            // [[Rcpp::depends("RcppArmadillo")]]
            // [[Rcpp::export]]
            
            arma::vec reweighting_C(arma::mat Ymat, arma::mat Yflagmat, arma::vec Y, arma::vec Yflag){
              
              int p = Ymat.n_cols;
              int k = Ymat.n_rows;
              double k_cut = 0.9*k;
              
              arma::vec res = Y;
              arma::mat Ymat_prod = Ymat%Yflagmat;
              
              for(int i = 0; i < p; ++i){
                
                arma::vec Ymat_weights = arma::ones<arma::vec>(k+1);
                arma::uvec Y_ind_zero = arma::find(Yflagmat.col(i) == 0);
                double zero_num = Y_ind_zero.n_elem;
                double nonzero_num = k - zero_num;
                if(Y(i) == 0) nonzero_num = nonzero_num + 1;
                
                if(zero_num < k_cut & zero_num != 0){
                  
                  double mean_Y = (sum(Ymat_prod.col(i)) + Y(i))/nonzero_num;
                  double pN = R::dpois(0, mean_Y, FALSE);
                  double non_dropout = nonzero_num*pN/(zero_num + nonzero_num*pN);
                  for(int j = 0; j < zero_num; ++j){
                    int zero_id = Y_ind_zero(j);
                    Ymat_weights(zero_id) = non_dropout;
                  }
                  if(Y(i) == 0) {
                    Ymat_weights(k) = non_dropout;
                  }
                }
                Ymat_weights /= sum(Ymat_weights);
                arma::vec Values = arma::ones<arma::vec>(k+1);
                Values(arma::span(0, k-1)) = Ymat.col(i);
                Values(k) = Y(i);
                res(i) = sum(Ymat_weights%Values);
              }
              
              return res*(k+1);
              
            }
            
            