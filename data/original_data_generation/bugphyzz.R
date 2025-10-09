library(dplyr)
library(stringr)
library(bugphyzz)

bp <- importBugphyzz()

# Function to check if a taxon name is a binomial name
is_binomial <- function(name) {
  words <- str_split(name, "\\s+")[[1]]
  length(words) == 2 && !grepl("sp\\.", name)
}

# Function to filter, select, and remove duplicates from a data frame
process_data_frame <- function(df, additional_filter = NULL) {
  processed_df <- df %>%
    filter(Rank == "species" & sapply(Taxon_name, is_binomial) & Frequency %in% c("always", "usually"))
  
  if (!is.null(additional_filter) && "Attribute_value" %in% colnames(processed_df)) {
    processed_df <- processed_df %>% filter(additional_filter)
  }
  
  if ("Attribute_value" %in% colnames(processed_df)) {
    processed_df <- processed_df %>%
      select(NCBI_ID, Taxon_name, Attribute_value) %>%
      distinct(Taxon_name, .keep_all = TRUE)
  } else {
    processed_df <- processed_df %>%
      select(NCBI_ID, Taxon_name) %>%
      distinct(Taxon_name, .keep_all = TRUE)
  }
  
  processed_df
}

# List of data frames to include in the aggregation
data_frames <- list(
  "motility",
  "gram stain",
  "aerophilicity", 
  "extreme environment",
  "biofilm formation",
  "animal pathogen",
  "biosafety level",
  "health associated",
  "host-associated",
  "plant pathogenicity",
  "spore formation",
  "hemolysis",
  "shape"
)

# Process each data frame and store the results in a list
processed_data <- lapply(data_frames, function(df_name) {
  additional_filter <- NULL
  
  if (df_name == "hemolysis" && "Attribute_value" %in% colnames(bp[[df_name]])) {
    processed_df <- process_data_frame(bp[[df_name]])
    additional_filter <- bp[[df_name]]$Attribute_value[match(processed_df$Taxon_name, bp[[df_name]]$Taxon_name)] %in% c("alpha", "beta", "gamma")
    processed_df <- processed_df[additional_filter, ]
  } else if (df_name == "shape" && "Attribute_value" %in% colnames(bp[[df_name]])) {
    shape_counts <- table(bp[[df_name]]$Attribute_value)
    shape_percentages <- shape_counts / sum(shape_counts) * 100
    top_shapes <- names(shape_percentages[shape_percentages >= 1])
    processed_df <- process_data_frame(bp[[df_name]])
    additional_filter <- bp[[df_name]]$Attribute_value[match(processed_df$Taxon_name, bp[[df_name]]$Taxon_name)] %in% top_shapes
    processed_df <- processed_df[additional_filter, ]
  } else {
    processed_df <- process_data_frame(bp[[df_name]])
  }
  
  processed_df %>%
    rename_with(~ df_name, .cols = matches("Attribute_value"))
})

# Merge all processed data frames based on Taxon_name
merged_data <- Reduce(function(x, y) full_join(x, y, by = c("NCBI_ID", "Taxon_name")), processed_data)

# Calculate the percentage of annotated metadata for each row
annotation_percentage <- rowSums(!is.na(merged_data[, -c(1, 2)])) / (ncol(merged_data) - 2) * 100

# Add a new column indicating rows with >60% annotated metadata
merged_data$highly_annotated <- annotation_percentage > 60

merged_data_highly_annotated <- merged_data[which(merged_data$highly_annotated),]
merged_data_not_highly_annotated <- merged_data[which(!merged_data$highly_annotated),]
saveRDS(merged_data, file = "merged_data.rds")
write.csv2(merged_data, file = "table_1.csv", row.names = F, quote = F)
write.csv2(merged_data_highly_annotated, file = "table_1_highly.csv", row.names = F, quote = F)
write.csv2(merged_data_not_highly_annotated, file = "table_1_not_highly.csv", row.names = F, quote = F)
