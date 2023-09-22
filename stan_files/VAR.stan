data {
  int<lower=1> m; // Dimension of the observation vector
  int<lower=1> p; // Order of the VAR (Vector AutoRegressive) model
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

  // Hyperparameters for the exchangeable multivariate normal prior for the phi_i
  vector[p] m_diag; // Prior mean for diagonal elements of AR coefficients
  vector<lower=0>[p] s_diag; // Prior standard deviation for diagonal elements of AR coefficients
  vector[p] m_offdiag; // Prior mean for off-diagonal elements of AR coefficients
  vector<lower=0>[p] s_offdiag; // Prior standard deviation for off-diagonal elements of AR coefficients
  
  real<lower=0> scale_diag; // Diagonal element of the scale matrix for prior on error covariance
  real<lower=-scale_diag / (m - 1)> scale_offdiag; // Off-diagonal element of the scale matrix
  real<lower=m + 3> df; // Degrees of freedom for inverse Wishart prior on error covariance
}

transformed data {
  matrix[m, m] scale_mat; // Constructing the scale-matrix for prior on Sigma
  
  // Fill in the scale matrix using scale_diag for diagonal and scale_offdiag for off-diagonal elements
  for (i in 1 : m) {
    for (j in 1 : m) {
      if (i == j) {
        scale_mat[i, j] = scale_diag;
      } else {
        scale_mat[i, j] = scale_offdiag;
      }
    }
  }
}

parameters {
  vector[m] beta1; // Coefficients for "Wind"
  vector[m] beta2; // Coefficients for "Dumping"
  vector[m] beta3; // Coefficients for "Dredging"
  vector[m] alpha; // Intercept terms for each time series dimension
  vector<lower=0, upper=100>[N_mis] y_mis; // Values for missing observations in the time series
  matrix[m, m] phi; // AR coefficients matrix
  cov_matrix[m] Sigma; // Error variance-covariance matrix
}

transformed parameters {
  array[N] vector[m] y; // Full time series (observed + imputed)
  array[N] vector[m] mut_rest; // Conditional means for VAR process
  array[N] vector[m] epsilon; // Residuals
  
  // Fill in observed values
  for (i in 1 : N_obs) {
    y[i_obs[i]][j_obs[i]] = y_obs[i];
  }
  
  // Fill in missing values
  for (i in 1 : N_mis) {
    y[i_mis[i]][j_mis[i]] = y_mis[i];
  }

  // Calculate residuals and conditional means
  epsilon[1] = y[1] - (alpha + beta1 * x_w[1] + beta2 * x_d[1] + beta3 * x_dr[1]);
  for (t in 2 : N) {
    epsilon[t] = y[t] - (alpha + beta1 * x_w[t] + beta2 * x_d[t] + beta3 * x_dr[t]);
    mut_rest[t] = phi * epsilon[t - 1] + alpha + beta1 * x_w[t] + beta2 * x_d[t] + beta3 * x_dr[t];
  }
}

model {
  // Likelihood for the VAR process
  for (i in 1 + p : N) {
    y[i] ~ multi_normal(mut_rest[i], Sigma);
  }
  
  // Prior specifications
  Sigma ~ inv_wishart(df, scale_mat);
  for (i in 1 : m) {
    alpha[i] ~ normal(0, 100);
    beta1[i] ~ normal(0, 100);
  }
  
  // Priors for "Dumping"
  for (i in 1 : 3) {
    beta2[i] ~ normal(0, 100);
    beta3[i] ~ normal(0, 0.3);
  }
  
  // Priors for "Dredging"
  for (i in 4 : 7) {
    beta3[i] ~ normal(0, 100);
    beta2[i] ~ normal(0, 0.3);
  }
  
  // Priors for missing values
  for (i in 1 : N_mis) {
    y_mis[i] ~ normal(0, 50);
  }
  
  // Priors for AR coefficients
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

generated quantities {
  array[N - p] vector[m] y_sim; // Simulated values for posterior predictive checks
  array[N - p] real log_lik; // Log likelihood for model comparison
  
  // Calculate likelihood and simulated values
  for (i in 1 : (N - p)) {
    y_sim[i] = multi_normal_rng(mut_rest[i + 1], Sigma);
    log_lik[i] = multi_normal_lpdf(y[i + 1] | mut_rest[i + 1], Sigma);
  }
}
