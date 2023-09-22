data {
  int<lower=1> m; // Dimension of observation vector                                                 
  int<lower=1> p; // Order of the VAR model                                                              
  int<lower=1> N_obs; // Number of observed values                                                   
  int<lower=0> N_mis; // Number of missing values                                                    
  int<lower=0> N; // Total number of time series observations (NOTE: N is not simply N_obs + N_mis)
  
  vector[N_obs] y_obs; // Observed values of the time series
  vector[N] x_d; // Time series data for "Dumping"
  vector[N] x_dr; // Time series data for "Dredging"
  vector[N] x_w; // Time series data for "Wind"
  
  // Indices to keep track of observed and missing data
  array[N_obs] int i_obs;
  array[N_obs] int j_obs;
  array[N_mis] int i_mis;
  array[N_mis] int j_mis;

  // Hyperparameters in exchangeable multivariate normal prior for the phi_i                         
  vector[p] m_diag; // Prior mean for diagonal elements of AR coefficients
  vector<lower=0>[p] s_diag; // Prior standard deviation for diagonal elements of AR coefficients
  vector[p] m_offdiag; // Prior mean for off-diagonal elements of AR coefficients
  vector<lower=0>[p] s_offdiag; // Prior standard deviation for off-diagonal elements of AR coefficients
}

parameters {
  vector[m] beta1; // Coefficients for "Wind"
  vector[m] beta2; // Coefficients for "Dumping"
  vector[m] beta3; // Coefficients for "Dredging"
  vector[m] alpha; // Intercept terms for each time series dimension
  vector<lower=0, upper=100>[N_mis] y_mis; // Values for missing observations in the time series
  matrix[m, m] phi; // AR coefficients matrix
  vector<lower=0>[m] gamma0;
  vector<lower=0, upper=1>[m] gamma1;
}

transformed parameters {
  array[N] vector[m] y; // Complete time series (combining observed and imputed missing values)
  array[N] vector[m] mut_rest; // Conditional mean of the process
  array[N] vector[m] epsilon; // Error terms
  array[N] vector[m] sigma; // Volatility 
  
  // This bit is necessary to separate the observed values from the missing values, on which we place priors.
  for (i in 1 : N_obs) {
    y[i_obs[i]][j_obs[i]] = y_obs[i];
  }
  for (i in 1 : N_mis) {
    y[i_mis[i]][j_mis[i]] = y_mis[i];
  }
  
  // Calculate error for the first observation
  epsilon[1] = y[1] - (alpha + beta1 * x_w[1] + beta2 * x_d[1] + beta3 * x_dr[1]);
  
  // Calculate error, conditional mean, and volatility for the rest of the observations
  for (t in 2 : N) {
    epsilon[t] = y[t]
                 - (alpha + beta1 * x_w[t] + beta2 * x_d[t] + beta3 * x_dr[t]);
    mut_rest[t] = phi * epsilon[t - 1] + alpha + beta1 * x_w[t]
                  + beta2 * x_d[t] + beta3 * x_dr[t];
    sigma[t] = sqrt(gamma0
                    + diag_matrix(gamma1) * diag_matrix(y[t - 1]) * y[t - 1]);
  }
}

model {
  for (j in 1 : m) {
    for (t in 2 : N) {
      // Likelihood for the time series observations
      y[t][j] ~ normal(mut_rest[t][j], sigma[t][j]);
    }
  }
  
  // Prior distributions for the parameters
  for (i in 1 : m) {
    alpha[i] ~ normal(0, 100);
    beta1[i] ~ normal(0, 100);
    //beta2[i] ~ normal(0, 100);                                                                       
    //beta3[i] ~ normal(0, 100);                                                                       
    gamma0[i] ~ std_normal();
    gamma1[i] ~ beta(1, 5);
  }
  
  // Priors specific to "Dumping"
  for (i in 1 : 3) {
    beta2[i] ~ normal(0, 100);
    beta3[i] ~ normal(0, 0.3);
  }
  
  // Priors specific to "Dredging"
  for (i in 4 : 7) {
    beta3[i] ~ normal(0, 100);
    beta2[i] ~ normal(0, 0.3);
  }
  
  // Prior for the missing observations
  for (i in 1 : N_mis) {
    y_mis[i] ~ normal(0, 50);
  }
  
  // Priors for the AR coefficients
  for (s in 1 : p) {
    diagonal(phi) ~ normal(m_diag[s], s_diag[s]);
    for (i in 1 : m) {
      for (j in 1 : m) {
        if (i != j) {
          phi[i, j] ~ normal(m_offdiag[s], s_offdiag[s]);
        }
      }
    }
  }
}

// The generated quantities block computes the log likelihood (needed for WAIC and LOO)
generated quantities {
  array[N - 1] vector[m] y_sim;
  array[N - 1] vector[m] log_lik;
  
  for (j in 1 : m) {
    for (t in 1 : N - 1) {
      // Likelihood                                                                                    
      y_sim[t][j] = normal_rng(mut_rest[t + 1][j], sigma[t + 1][j]);
      log_lik[t][j] = normal_lpdf(y[t + 1][j] | mut_rest[t + 1][j], sigma[t + 1][j]);
    }
  }
}                                                                                 