# Ingestion script for Transfermarkt using worldfootballR (and rvest as a fast-path fallback)
# R version 4.4.3
suppressPackageStartupMessages({
  library(dplyr)
  library(rvest)
  library(readr)
})

cat("Initializing Transfermarkt ingestion...\n")

# Output directory ensuring Medallion architecture
dir.create("data/bronze", recursive = TRUE, showWarnings = FALSE)
output_file <- "data/bronze/transfermarkt_raw.csv"

# Function to safely scrape National Team aggregated values using rvest
# because worldfootballR is club-focused and looping over 100+ national 
# teams individually triggers severe rate limits on TM.
scrape_national_team_values <- function() {
  cat("Fetching aggregated National Team squad values from Transfermarkt...\n")
  
  # The global national teams page on Transfermarkt
  url <- "https://www.transfermarkt.com/wettbewerbe/nationalmannschaften/statistik"
  
  tryCatch({
    page <- read_html(url)
    
    # Extract the main table
    tables <- html_table(page, fill = TRUE)
    
    # Typically, the first or second table contains the data
    df <- tables[[1]]
    
    # Clean the column names and data
    # TM tables have some empty/icon columns, so we select by index
    # 2: Country name
    # 3: Squad size
    # 5: Total Market Value
    df_clean <- df %>%
      select(Nation = 2, Squad = 3, Total_Value = 5) %>%
      filter(Nation != "", !is.na(Squad)) %>%
      mutate(
        Total_Value_Num = case_when(
          grepl("bn", Total_Value) ~ as.numeric(gsub("€|bn", "", Total_Value)) * 1000000000,
          grepl("m", Total_Value) ~ as.numeric(gsub("€|m", "", Total_Value)) * 1000000,
          grepl("k", Total_Value) ~ as.numeric(gsub("€|k", "", Total_Value)) * 1000,
          TRUE ~ 0
        )
      )
    
    return(df_clean)
    
  }, error = function(e) {
    cat("Error scraping Transfermarkt: ", e$message, "\n")
    return(NULL)
  })
}

# Execute
df_tm <- scrape_national_team_values()

if (!is.null(df_tm) && nrow(df_tm) > 0) {
  cat("Successfully fetched", nrow(df_tm), "national teams.\n")
  write_csv(df_tm, output_file)
  cat("Saved Bronze data to:", output_file, "\n")
} else {
  cat("Failed to retrieve data. Please check connection or if Transfermarkt blocked the IP.\n")
  quit(status=1)
}
