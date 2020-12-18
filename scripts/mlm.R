library('lmerTest')
library('lme4')
library('ggplot2')
library('plyr')

df <- read.csv('../meet_weighted-True_control-personality-False_brain_networks_melted.csv')
# df <- subset(df, phys_dist != 0) # Optional

# Turn social distance into social network proximity
df$soc_proximity_cleaned <- -df$soc_dist_cleaned

# Create doubled dataframe
names(df)[names(df) == 'Unnamed..0'] <- 'sub1'
names(df)[names(df) == 'Unnamed..1'] <- 'sub2'
df2 <- df
colnames(df2)[1:2] <- c("sub2", "sub1")

df_double <- rbind(df, df2)

se_scaling_factor <- sqrt(2 * (57-1)) / sqrt(57 - 1) # use this to adjust SE

f <- function(data){
  data$value <- scale(data$value)
  data$soc_proximity_cleaned <- data$soc_proximity_cleaned
  md <- summary(lmer(soc_proximity_cleaned ~ scale(value) + (1|sub1) + (1|sub2), data))
  B <- md$coefficients[2]
  se <- md$coefficients[2,2] * se_scaling_factor
  df <- md$coefficients[2,3]
  t <- md$coefficients[2,4]
  p_raw <- md$coefficients[2,5]
  p_penalized <- 2 * pt(-abs(t), 236)
  return(data.frame(B, se, df, t, p_raw, p_penalized))
}
mlm_df <- ddply(df_double, .(variable), f)
mlm_df$p_corrected <- p.adjust(mlm_df$p_penalized, method = 'fdr')
subset(mlm_df, p_corrected < .05)
