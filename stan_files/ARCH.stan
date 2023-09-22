// ARCH Model for Time Series with Environmental Covariates and Missing Values
data {
  int<lower=1> m; // Dimension of observation vector
  int<lower=1> N_obs; // Number of observed time series values
  int<lower=0> N_mis; // Number of missing time series values
  int<lower=0> N; // Total number of time series observations (NOT equal to N_obs + N_mis)
  vector[N_obs] y_obs; // Observed time series values
  vector[N] x_d; // Time series of the dumping covariate
  vector[N] x_dr; // Time series of the dredging covariate
  vector[N] x_w;  // Time series of the wind covariate
  array[N_obs] int i_obs; // Array indicating the time indices of observed values
  array[N_obs] int j_obs; // Array indicating the variable indices of observed values
  array[N_mis] int i_mis; // Array indicating the time indices of missing values
  array[N_mis] int j_mis; // Array indicating the variable indices of missing values
}

parameters {
  vector[m] beta1; // Slope parameters for the wind covariate
  vector[m] beta2; // Slope parameters for the dumping covariate
  vector[m] beta3; // Slope parameters for the dredging covariate
  vector[m] alpha; // Intercept parameter for each dimension of the observation vector
  vector<lower=0, upper=100>[N_mis] y_mis; // Missing y values to be estimated
  vector<upper=1>[m] phi; // Autoregressive parameter for each dimension
  vector<lower=0>[m] gamma0; // Intercept for the volatility model for each dimension
  vector<lower=0, upper=1>[m] gamma1; // Slope for the volatility model for each dimension
}
transformed parameters {
  array[N] vector[m] y; // Complete time series (including observed and missing values)
  array[N] vector[m] mut_rest; // Conditional mean given previous values and covariates
  array[N] vector[m] epsilon; // Residual errors
  array[N] vector[m] sigma; // Volatility or standard deviation of residuals at each time
  
  // This part is necessary to define a prior on the missing values
  for (i in 1 : N_obs) {
    y[i_obs[i]][j_obs[i]] = y_obs[i];
  }
  for (i in 1 : N_mis) {
    y[i_mis[i]][j_mis[i]] = y_mis[i];
  }
  
  // Calculate residuals and conditional means for each time point
  epsilon[1] = y[1]
               - (alpha + beta1 * x_w[1] + beta2 * x_d[1] + beta3 * x_dr[1]);
  
  for (t in 2 : N) {
    epsilon[t] = y[t]
                 - (alpha + beta1 * x_w[t] + beta2 * x_d[t] + beta3 * x_dr[t]);
    mut_rest[t] = diag_matrix(phi) * epsilon[t - 1] + alpha + beta1 * x_w[t]
                  + beta2 * x_d[t] + beta3 * x_dr[t];
    sigma[t] = sqrt(gamma0
                    + diag_matrix(gamma1) * diag_matrix(y[t - 1]) * y[t - 1]);
  }
}
model {
  // Likelihood for the time series given model parameters
  for (j in 1 : m) {
    for (t in 2 : N) {
      y[t][j] ~ normal(mut_rest[t][j], sigma[t][j]);
    }
  }

  // Priors for the model parameters
  for (i in 1 : m) {
    alpha[i] ~ normal(0, 5);
    beta1[i] ~ normal(0, 100);
    //beta2[i] ~ normal(0, 100);
    //beta3[i] ~ normal(0, 100);
    gamma0[i] ~ std_normal();
    gamma1[i] ~ beta(1, 5);
    phi[i] ~ normal(0, 0.3);
  }
  
  // Special priors for dumping and dredging effects
  for (i in 1 : 3) {
    beta2[i] ~ normal(0, 100);
    beta3[i] ~ normal(0, 0.3);
  }
  
  // Prior for missing time series values
  for (i in 4 : 7) {
    beta3[i] ~ normal(0, 100);
    beta2[i] ~ normal(0, 0.3);
  }
  
  for (i in 1 : N_mis) {
    y_mis[i] ~ normal(0, 50);
  }
}
generated quantities {
  array[N - 1] vector[m] y_sim; // Simulated time series for out-of-sample prediction
  array[N - 1] vector[m] log_lik; // Log likelihood for the observed values given the model
  
  for (j in 1 : m) {
    for (t in 1 : N - 1) {
      // Likelihood
      y_sim[t][j] = normal_rng(mut_rest[t + 1][j], sigma[t + 1][j]);
      log_lik[t][j] = normal_lpdf(y[t + 1][j] | mut_rest[t + 1][j], sigma[t + 1][j]);
    }
  }
}