# Description: Make plots showing accuracy curves for each model

library(readr)
library(stringr)
library(dplyr)
library(ggplot2)
library(tidyr)

data_dir <- "~/code/collab-cs235/pset3/results"
test_accuracy <- read_csv(file.path(data_dir, "test_accuracies.csv"))
files <- list.files(data_dir)
files <- files[files != "test_accuracies.csv" & str_detect(files, "_results.csv")]
dfs <- list()
for (i in 1:length(files)) {
  f <- files[i]
  df <- read_csv(file.path(data_dir, f))
  df$model <- str_split(f, "_results.csv", simplify = T)[, 1]
  dfs[[i]] <- df
}
df <- dfs %>%
  bind_rows() %>%
  group_by(model) %>%
  mutate(epoch = row_number()) %>%
  pivot_longer(!all_of(c("model", "epoch")), names_to = "accuracy_type", values_to = "accuracy")

model_levels <- c("Baseline CNN", "Multilayer CNN", "Multilayer CNN + Dropout", 
                  "Multilayer CNN + Dropout + Deep FCN")
acc_plot <- df %>%
  mutate(model = case_when(
    model == "baseline" ~ "Baseline CNN",
    model == "multilayer_cnn" ~ "Multilayer CNN",
    model == "multilayer_reg_cnn" ~ "Multilayer CNN + Dropout",
    model == "deep_cnn" ~ "Multilayer CNN + Dropout + Deep FCN"
  )) %>%
  mutate(model = factor(model, levels = model_levels)) %>%
  mutate(accuracy_type = ifelse(accuracy_type == "train_accuracy", "Train Accuracy", "Validation Accuracy")) %>%
  ggplot(aes(epoch, accuracy)) +
  geom_path(aes(color = accuracy_type)) +
  theme_bw() +
  facet_wrap(~model) +
  guides(color = guide_legend(title = "")) +
  theme(legend.position = "top") +
  labs(x = "Epoch", y = "Accuracy")
print(acc_plot)

test_accuracy$model <- model_levels
test_plot <- test_accuracy %>%
  ggplot(aes(model, accuracy)) +
  geom_col(aes(fill = model)) +
  guides(fill = FALSE) +
  theme_bw() +
  labs(x = "Model", y = "Accuracy")
print(test_plot)



  
