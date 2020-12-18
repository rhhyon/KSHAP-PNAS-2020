library('lmerTest')
library('lme4')
library('plyr')


############################################################
# Moderation analysis using primary PLS component
############################################################
df <- read.csv('../PLS_components_meet_weighted-True_control-personality-False.csv')
df$phys_dist = scale(df$phys_dist)
df2 <- df
colnames(df2)[1:2] <- c("sub2", "sub1")
df_double <- rbind(df, df2)
md <- summary(lmer(soc_proximity_cleaned ~ neural_similarity_c1 + phys_dist + neural_similarity_c1:phys_dist + (1|sub1) + (1|sub2), data = df_double))

se_scaling_factor <- sqrt(2 * (57-1)) / sqrt(57 - 1) # use this to adjust SE

B_neural_sim <- md$coefficients[2]
B_phys_dist <- md$coefficients[3]
B_interaction <- md$coefficients[4]

se_neural_sim <- md$coefficients[2, 2] * se_scaling_factor
se_phys_dist <- md$coefficients[3, 2] * se_scaling_factor
se_interaction <- md$coefficients[4, 2] * se_scaling_factor

df_neural_sim <- md$coefficients[2, 3]
df_phys_dist <- md$coefficients[3, 3]
df_interaction <- md$coefficients[4, 3]

t_neural_sim <- md$coefficients[2, 4]
t_phys_dist <- md$coefficients[3, 4]
t_interaction <- md$coefficients[4, 4]

p_raw_neural_sim <- md$coefficients[2, 5]
p_raw_phys_dist <- md$coefficients[3, 5]
p_raw_interaction <- md$coefficients[4, 5]

p_neural_sim <- 2 * pt(-abs(t_neural_sim), 236)
p_phys_dist <- 2 * pt(-abs(t_phys_dist), 236)
p_interaction <- 2 * pt(-abs(t_interaction), 236)

results = data.frame(B_neural_sim, B_phys_dist, B_interaction,
                  se_neural_sim, se_phys_dist, se_interaction,
                  df_neural_sim, df_phys_dist, df_interaction,
                  t_neural_sim, t_phys_dist, t_interaction,
                  p_neural_sim, p_phys_dist, p_interaction
                  )





############################################################
# Moderation analysis using within/between systems connectivity
############################################################

# setwd('..')
# setwd('/Users/hyon/mnt/kshap/KSHAP-PNAS-2020/scripts')

df <- read.csv('../meet_weighted-True_control-personality-False_brain_networks_melted.csv')
# df <- subset(df, phys_dist != 0) # Optional
df <- subset(df, variable == 'DMN|DATTN') # insert system of interest
df$soc_proximity_cleaned = -df$soc_dist_cleaned
df$phys_dist <- scale(df$phys_dist)
df$value <- scale(df$value)
df2 <- df
colnames(df2)[1:2] <- c("sub2", "sub1")
df_double <- rbind(df, df2)

df_subset$value <- scale(df_subset$value)
md <- summary(lmer(soc_proximity_cleaned ~ value + phys_dist + value:phys_dist + (1|sub1) + (1|sub2), df_subset))

se_scaling_factor <- sqrt(2 * (57-1)) / sqrt(57 - 1) # use this to adjust SE

B_neural_sim <- md$coefficients[2]
B_phys_dist <- md$coefficients[3]
B_interaction <- md$coefficients[4]

se_neural_sim <- md$coefficients[2, 2] * se_scaling_factor
se_phys_dist <- md$coefficients[3, 2] * se_scaling_factor
se_interaction <- md$coefficients[4, 2] * se_scaling_factor

df_neural_sim <- md$coefficients[2, 3]
df_phys_dist <- md$coefficients[3, 3]
df_interaction <- md$coefficients[4, 3]

t_neural_sim <- md$coefficients[2, 4]
t_phys_dist <- md$coefficients[3, 4]
t_interaction <- md$coefficients[4, 4]

p_raw_neural_sim <- md$coefficients[2, 5]
p_raw_phys_dist <- md$coefficients[3, 5]
p_raw_interaction <- md$coefficients[4, 5]

p_neural_sim <- 2 * pt(-abs(t_neural_sim), 236)
p_phys_dist <- 2 * pt(-abs(t_phys_dist), 236)
p_interaction <- 2 * pt(-abs(t_interaction), 236)

results = data.frame(B_neural_sim, B_phys_dist, B_interaction,
                  se_neural_sim, se_phys_dist, se_interaction,
                  df_neural_sim, df_phys_dist, df_interaction,
                  t_neural_sim, t_phys_dist, t_interaction,
                  p_neural_sim, p_phys_dist, p_interaction
                  )
