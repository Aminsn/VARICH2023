# This script fits an ARCH model to the data.
# Model specification is in stan_files/ARCH.stan
# Author: Amin Shoari Nejad

# ======== loading libraries ========
library(tidyverse)
library(cmdstanr)

# ======== reading and preparing data =========
df <- readRDS("data.rds")
# centering wind speed
df$wind_speed = df$wind_speed - mean(df$wind_speed)
k = 7 # number of time series
df_long <- df |> 
    select(date, dumpsite_top:eastlink) |> 
    pivot_longer(-date, names_to = 'type', values_to = 'value') |> 
    mutate(time_index = rep(1:(n()/k), each = k),
           site_index = rep(1:k, times = n()/(k)))

# observations indices (needed for Stan)
df_long_obs <- df_long |> 
    na.omit() |> 
    select(-date, -type)
# missing indices (needed for Stan)
df_long_mis <- df_long |> 
    filter(is.na(value)) |> 
    select(-date, -value, -type)

# ========  model data ======== 
arch_data <- list(m = 7,
                 N_obs = nrow(df_long_obs),
                 N_mis = nrow(df_long_mis),
                 N = max(df_long_obs$time_index, 
                         df_long_mis$time_index),
                 y_obs = df_long_obs$value,
                 i_obs = df_long_obs$time_index,
                 j_obs = df_long_obs$site_index,
                 i_mis = df_long_mis$time_index,
                 j_mis = df_long_mis$site_index,
                 x_w = df$wind_speed,
                 x_d = df$dumping,
                 x_dr = df$dredging)



# ======== model compilation and fitting ========
mod <- cmdstan_model('stan_files/ARCH.stan')

out <- mod$sample(
  data = arch_data,
  iter_warmup = 200,
  iter_sampling = 800,
  parallel_chains = 4
)

# ======== checking for convergence ======== 
test = out$draws(c("beta1", "beta2","beta3","alpha", "gamma0", "gamma1", "phi"))
max(summary(test)$rhat) < 1.05
# ========  saving the fitted model ======== 
saveRDS(out, 'fitted_objects/arch_fitted.rds')


