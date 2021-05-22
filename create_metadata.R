library(readr)
library(dplyr)
library(stringr)

data_dir <- "/Users/tomasbencomo/code/collab-cs235/pset3"
fp <- file.path(data_dir, "mass_case_description_train_set.csv")
df <- read_csv(fp)

image_dir <- "data_fixed_crop_with_mask"
df <- df %>%
  mutate(id = str_split(patient_id, "_", simplify = T)[, 2]) %>%
  mutate(filepath = file.path(image_dir, id)) %>%
  mutate(filepath = file.path(filepath, str_c(str_c(side, view, sep="_"), ".h5"))) %>%
  mutate(label = case_when(
    pathology == "BENIGN" | pathology == "BENIGN_WITHOUT_CALLBACK" ~ 0,
    pathology == "MALIGNANT" ~ 1
  )) %>%
  select(-od_img_path, -od_crop_path, -mask_path)

# Not all patients in metadata are available to us
# Some patients are available but have missing images
available_dirs <- list.files(file.path(data_dir, image_dir), full.names = F)
df <- df %>%
  filter(id %in% available_dirs) %>%
  filter(file.exists(file.path(data_dir, filepath)))

outfp <- file.path(data_dir, "image_metadata.csv")
write_csv(df, outfp)
