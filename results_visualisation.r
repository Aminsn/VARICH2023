library(tidyverse)
library(posterior)

# Reading and preparing data
df <- readRDS("df_total.rds")
df <- df %>% filter(year(Date) < 2019) 
k = 7 # number of time series
turb_use <- df %>% select(Date, buoy1_top:eastlink) %>% 
  pivot_longer(-Date, names_to = 'Type', values_to = 'Value') %>% 
  mutate(i_index = rep(1:(n()/k), each = k),
         j_index = rep(1:k, times = n()/(k)))

turb_mis <- turb_use %>% filter(is.na(Value)) %>% 
  select(-Date, -Value, -Type)

out = readRDS('fitted_objects/varich_fitted.rds') #Loading the fitted model

# Checking Rhat values 
# df_rhat = summary(out$draws(c("beta1", "beta2", "beta3", "alpha", "gamma0", "gamma1", "phi")))
# hist(df_rhat$rhat)
# 
# # Goodness of fit evaluation using WAIC & LOO
# loglik_matrix  <- out$draws(c("log_lik")) # log lik
# 
# loo::waic(loglik_matrix)
# loo::loo(loglik_matrix)

# ========================= A sample plot of the raw data ============================
plt_theme = theme(
  axis.text.x = element_text(size = 15, angle = 0, margin = margin(t = 6)),
  axis.text.y = element_text(size = 15, margin = margin(t = 6)),
  legend.text = element_text(family = "Arial",size = 15),
  legend.title = element_text(size = 15, face = "bold"),
  legend.spacing.x = unit(1,"cm"),
  axis.title.y = element_text(family = "Arial",size = 12, face = "bold"),
  axis.title.x = element_text(family = "Arial",size = 12, face = "bold", 
                              margin = margin(t = 10, r = 0, b = 0, l = 0)),
  title = element_text(face = 'bold'))



df %>% 
  dplyr::select(Date,tolka) %>% 
  filter(year(Date) < 2019) %>% 
  ggplot(aes(Date, tolka)) + geom_line() +
  scale_x_datetime(date_breaks = "2 month", date_labels = '%Y %b') +
  ylab('Turbidity (NTU)') +
  theme_bw() +
  plt_theme


# ========================= Fitting vs Observed plots over time ============================
post_matrix  <- out$draws(c("y_sim")) # Posterior predictives
post_matrix <- as_draws_df(post_matrix) %>% 
  select(contains("y_sim"))

post_missing  <- out$draws(c("y_mis")) # PPs of missing data
post_missing_matrix <- as_draws_df(post_missing) %>% 
  select(contains("y_mis"))

# Summarising samples into quantiles 
summary_tbl <- as.data.frame(t(apply(post_matrix, 2, quantile, probs = c(0.025,0.5, 0.975), na.rm = T))) # total

# Determining which rows are missing data
summary_tbl_modified = summary_tbl %>% 
  mutate(i_index = as.numeric(str_extract(rownames(.), "[[:digit:]]+")),
         j_index = as.numeric(str_sub(rownames(.), -2, -2)))

summary_tbl_modified$i_index = summary_tbl_modified$i_index + 2 # The 2 lags (change it to 1 for without diff model, check the generated quantity block) should be added to convert the model times to the orginial times

# Replacing pps of y_sim with pps of y_mis for missing data
summary_tbl_ready = summary_tbl_modified 


df_posterior = summary_tbl_ready %>% 
  mutate(Type = case_when(j_index == 1 ~ "buoy1_top",
                          j_index == 2 ~ "buoy1_middle",
                          j_index == 3 ~ "buoy1_bottom",
                          j_index == 4 ~ "tolka" ,
                          j_index == 5 ~ "northbank" ,
                          j_index == 6 ~ "poolbeg" ,
                          TRUE ~ "eastlink")) %>%
  rename(low = `2.5%`, mean = `50%`, high = `97.5%`) %>% 
  group_by(Type) %>% 
  mutate(time = i_index) %>% 
  ungroup() %>% 
  mutate(across(everything(.), ~ replace(., . < 0, 0))) %>% 
  select(-c(i_index, j_index))


turb_use_modified= turb_use %>% 
  rename(mean = Value,
         time = i_index)

# Merging observation with fitted values
fit_obs_merged = turb_use_modified %>% 
  select(Type, mean, time) %>% 
  rename(observed = mean) %>% 
  left_join(., df_posterior, by = c("Type","time")) %>% 
  rename(fit = mean) %>% 
  pivot_longer(names_to = "Type2",
               values_to = "turbidity",
               -c(Type, time, low, high))

# Truncating the results to positive values and removing the big missing chunks
df_plot = fit_obs_merged %>%
  mutate(across(c(Type, Type2), ~ str_to_title(.))) %>% 
  mutate(Type = factor(Type, levels = c("Buoy1_top","Buoy1_middle", "Buoy1_bottom", "Northbank", "Poolbeg","Eastlink","Tolka")))


# Generating the fitted vs observed over time plots
(dre_fit_time = df_plot %>% 
  left_join(., turb_use %>% select(Date, i_index) %>% rename(time = i_index)) %>% 
  filter(time > 2) %>%
    mutate(Type = str_replace(Type,"Buoy1", "Dumpsite"),
           Type = factor(Type, levels = c("Dumpsite_top","Dumpsite_middle", "Dumpsite_bottom", "Northbank", "Poolbeg","Eastlink","Tolka"))) %>% 
  ggplot(aes(x = Date, y = turbidity)) +
  geom_ribbon(aes(ymin = low  , ymax = high), alpha = 0.3) +
  geom_line(aes(color = Type2), size = 1, alpha = 0.6) + 
  labs(x = 'Date', y = 'Turbidity (NTU)', color = 'Type') +
  scale_x_datetime(date_breaks = "2 month", date_labels = '%Y %b') +
  facet_grid(Type ~ ., scales="free_y") +
  theme_bw() +
  plt_theme)
