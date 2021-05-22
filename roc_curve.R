library(readr)
library(dplyr)
library(ggplot2)

data_dir <- "~/code/collab-cs235/pset3/results"
baseline <- read_csv(file.path(data_dir, "b_roc.csv"))
baseline$model <- "Baseline CNN"
mcnn <- read_csv(file.path(data_dir, "mcnn_roc.csv"))
mcnn$model <-  "Multilayer CNN"
mrcnn <- read_csv(file.path(data_dir, "mrcnn_roc.csv"))
mrcnn$model <- "Multilayer CNN + Dropout"
dcnn <- read_csv(file.path(data_dir, "dcnn_roc.csv"))
dcnn$model <- "Multilayer CNN + Dropout + Deep FCN"

model_levels <- c("Baseline CNN", "Multilayer CNN", "Multilayer CNN + Dropout", 
                  "Multilayer CNN + Dropout + Deep FCN")
df <- rbind(baseline, mcnn, mrcnn, dcnn)

roc <- df %>%
  mutate(model = factor(model, levels = model_levels)) %>%
  ggplot(aes(fpr, tpr)) +
  geom_path(aes(color = model)) +
  geom_abline(slope = 1, linetype = "dashed", color = "red") +
  theme_bw() +
  guides(color = guide_legend(title = "")) +
  theme(legend.position = "top") +
  labs(x = "False Positive Rate", y = "True Positive Rate")
print(roc)
