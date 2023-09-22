# Description: This script compares the fitted models using WAIC and LOOIC.
# Author: Amin Shoari Nejad

# ----------- Load the necessary libraries ------------
library(tidyverse)

# ----------------- WAIC & LOO comparisons --------------
# This section compares models using the Watanabe-Akaike Information Criterion (WAIC) 
# and Leave-One-Out Information Criterion (LOOIC) 

# Load fitted model objects from specified paths.
ARCH   = readRDS("fitted_objects/arch_fitted.rds")
VAR    = readRDS("fitted_objects/var_fitted.rds")
VARCH  = readRDS("fitted_objects/varch_fitted.rds")
VARICH = readRDS("fitted_objects/varich_fitted.rds")

# Consolidate the fitted model objects into a list.
lst = list(ARCH = ARCH, VAR = VAR, VARCH = VARCH, VARICH = VARICH)

# Initialize an empty data frame to store WAIC & LOOIC statistics.
df = data.frame()

# Loop over each model in the list to compute WAIC and LOOIC.
for(t in names(lst)){
  
  # Extract the log likelihood matrix from each model's draws.
  loglik_matrix = lst[[t]]$draws(c("log_lik")) 
  
  # Compute WAIC and LOOIC for each model.
  waic  = loo::waic(loglik_matrix)
  looic = loo::loo(loglik_matrix)
  
  # Format the WAIC and LOOIC results and add the model name for identification.
  waic_modif = waic$estimates %>%
    cbind(type = rownames(.)) %>%
    as.data.frame() %>%
    mutate(model = t)
  
  looic_modif = looic$estimates %>%
    cbind(type = rownames(.)) %>%
    as.data.frame() %>%
    mutate(model = t)
  
  # Append the formatted results to the main data frame.
  df = rbind(df, waic_modif, looic_modif)
  
}

# Visualize the WAIC & LOOIC comparisons across models.
df %>%
  # Convert estimate and standard error columns to numeric data types.
  mutate(across(Estimate:SE, ~ as.numeric(as.character(.)))) %>% 
  # Compute the lower and upper bounds for the error bars.
  mutate(low = Estimate - SE, 
         high = Estimate + SE,
         # Specify the order of the models on the x-axis.
         model = factor(model, levels = c('VAR', 'ARCH', 'VARCH', 'VARICH'))) %>% 
  # Filter for only waic and looic statistics.
  filter(type == "waic" | type == "looic") %>%
  # Plot the data.
  ggplot(aes(model, Estimate, group=type, color=type)) + 
  labs(x = "Model", color = 'Type') + 
  geom_line(size = 1) +
  geom_point()+
  geom_errorbar(aes(ymin=Estimate-SE, ymax=Estimate+SE), 
                width=.2, 
                size = 1,  
                position=position_dodge(0.05)) + 
  theme_bw() +
  theme(
    axis.text.x = element_text(size = 15, angle = 0, margin = margin(t = 6)),
    axis.text.y = element_text(size = 15, margin = margin(t = 6)),
    legend.text = element_text(family = "Arial",size = 15),
    legend.title = element_text(size = 15, face = "bold"),
    legend.spacing.x = unit(1,"cm"),
    axis.title.y = element_text(family = "Arial",size = 12, face = "bold"),
    axis.title.x = element_text(family = "Arial",size = 12, face = "bold", 
                                margin = margin(t = 10, r = 0, b = 0, l = 0)),
    title = element_text(face = 'bold'))
