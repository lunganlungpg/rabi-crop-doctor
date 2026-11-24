# ==============================================================================
# advisory.R
# Bilingual (English & Hindi) Crop Disease Symptom & ICAR/KVK Advisory Lookup
# ==============================================================================

#' List All Supported Indian Rabi Crop Classes and Pathogen Metadata
#'
#' Returns the full reference table for Wheat, Mustard, Chickpea, and Potato
#' health/disease classes bundled with \pkg{rabidoctor}.
#'
#' @param crop Optional crop filter (\code{"wheat"}, \code{"mustard"},
#'   \code{"chickpea"}, or \code{"potato"}).
#' @return A \code{data.frame} with disease IDs, pathogens, ETLs, and peak months.
#' @examples
#' list_rabi_diseases("wheat")
#' @export
list_rabi_diseases <- function(crop = NULL) {
  csv_path <- system.file("extdata", "rabi_classes_advisory.csv", package = "rabidoctor")
  if (!nzchar(csv_path) || !file.exists(csv_path)) {
    csv_path <- file.path("R-package", "inst", "extdata", "rabi_classes_advisory.csv")
  }
  df <- utils::read.csv(csv_path, stringsAsFactors = FALSE, fileEncoding = "UTF-8")
  if (!is.null(crop)) {
    df <- df[tolower(df$crop) == tolower(crop), , drop = FALSE]
  }
  rownames(df) <- NULL
  df
}

#' Retrieve Symptoms and Farmer IPM Actions for a Specific Disease Class
#'
#' Looks up symptoms, agro-climatic triggers, Economic Threshold Level (ETL),
#' and step-by-step field actions in English or Hindi.
#'
#' @param class_id Character class identifier (e.g., \code{"wheat_yellow_rust"}).
#' @param lang Language code: \code{"en"} for English (default) or \code{"hi"} for Hindi.
#' @return Named list containing crop metadata, symptoms, triggers, and action steps.
#' @examples
#' get_rabi_advisory("wheat_yellow_rust", lang = "en")
#' @export
get_rabi_advisory <- function(class_id, lang = c("en", "hi")) {
  lang <- match.arg(lang)
  df <- list_rabi_diseases()
  idx <- which(df$id == class_id)
  if (length(idx) == 0L) {
    stop(sprintf("Unknown class_id '%s'. Run list_rabi_diseases() for valid IDs.", class_id), call. = FALSE)
  }
  row <- df[idx[1L], , drop = FALSE]
  actions_raw <- if (lang == "hi") row$action_hi else row$action_en
  actions <- unlist(strsplit(actions_raw, " | ", fixed = TRUE))

  list(
    id = row$id,
    crop = row$crop,
    is_healthy = as.logical(row$is_healthy),
    name = if (lang == "hi") row$name_hi else row$name_en,
    pathogen = row$pathogen,
    etl_pct = row$etl_pct,
    peak_months = row$peak_months,
    weather_trigger = if (lang == "hi") row$weather_trigger_hi else row$weather_trigger_en,
    symptoms = if (lang == "hi") row$symptoms_hi else row$symptoms_en,
    actions = actions,
    kisan_helpline = "1800-180-1551"
  )
}
