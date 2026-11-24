# ==============================================================================
# incidence.R
# Field Disease Incidence (%), Wheeler's Percent Disease Index (PDI),
# 0-5 Foliar Severity Grading, and Regional Risk Ranking in Base R.
# ==============================================================================

#' Convert Foliar Lesion Percentage to Standard 0-5 Pathology Grade
#'
#' Maps estimated leaf lesion area (\%) onto the 6-tier botanical rating scale
#' aligned cleanly with the Python and Android APK inference engines:
#' \itemize{
#'   \item Grade 0: <= 2.0\% (Healthy / Trace clean blade)
#'   \item Grade 1: > 2.0\% to 10.0\% (Initial infection spots)
#'   \item Grade 2: > 10.0\% to 25.0\% (Low-moderate spread)
#'   \item Grade 3: > 25.0\% to 50.0\% (Moderate damage)
#'   \item Grade 4: > 50.0\% to 75.0\% (Severe foliar blight/rust)
#'   \item Grade 5: > 75.0\% (Very severe / desiccated lamina)
#' }
#'
#' @param severity_pct Numeric vector of leaf lesion area percentages (0 to 100).
#' @return Integer vector of severity grades in \code{0:5}.
#' @examples
#' severity_to_grade(c(0.5, 8.0, 18.5, 42.0, 82.0))
#' @export
severity_to_grade <- function(severity_pct) {
  if (!is.numeric(severity_pct)) {
    stop("'severity_pct' must be a numeric vector.", call. = FALSE)
  }
  vapply(severity_pct, function(pct) {
    if (is.na(pct)) return(NA_integer_)
    if (pct <= 2.0) return(0L)
    if (pct <= 10.0) return(1L)
    if (pct <= 25.0) return(2L)
    if (pct <= 50.0) return(3L)
    if (pct <= 75.0) return(4L)
    5L
  }, integer(1))
}

#' Classify Field Disease Incidence and PDI Against KVK Thresholds
#'
#' Evaluates overall field urgency using general Indian KVK Economic Threshold
#' Level (ETL) guidelines for cereal rusts, Brassica blights, and potato blights.
#'
#' @param incidence_pct Numeric scalar: percentage of diseased plants/leaves.
#' @param pdi_pct Numeric scalar: Wheeler (1969) Percent Disease Index (\%).
#' @return Character string describing risk category and recommended urgency.
#' @examples
#' evaluate_etl_risk(incidence_pct = 28.0, pdi_pct = 16.5)
#' @export
evaluate_etl_risk <- function(incidence_pct, pdi_pct) {
  if (is.na(incidence_pct) || incidence_pct == 0.0) {
    return("Healthy / Safe")
  }
  if (incidence_pct < 10.0 && pdi_pct < 8.0) {
    return("Low Watch (Monitor Borders)")
  }
  if (incidence_pct < 25.0) {
    return("Moderate / Near ETL - Verify with KVK")
  }
  "High Alert / Exceeds ETL - Prompt Action Needed"
}

#' Compute Plot Disease Incidence (%) and Wheeler Percent Disease Index (PDI)
#'
#' Calculates epidemiological survey summaries for W-pattern plot scouting:
#' \deqn{\text{Incidence (\%)} = \frac{N_{\text{infected}}}{N_{\text{total}}} \times 100}
#' \deqn{\text{PDI (\%)} = \frac{\sum \text{Grade}_{i}}{N_{\text{total}} \times \text{max\_scale}} \times 100}
#'
#' @param is_healthy Logical vector (\code{TRUE} for healthy leaf, \code{FALSE} for diseased).
#' @param severity_pct Numeric vector of leaf lesion area percentages (0--100).
#' @param disease Optional character vector of diagnosed disease IDs.
#' @param max_scale Integer maximum rating grade (default \code{5L}).
#' @return Named list containing \code{total_inspected}, \code{infected_count},
#'   \code{disease_incidence_pct}, \code{pdi_pct}, \code{mean_severity_pct},
#'   \code{risk_level}, and \code{disease_counts}.
#' @examples
#' compute_field_incidence(
#'   is_healthy = c(TRUE, FALSE, FALSE, TRUE),
#'   severity_pct = c(0.5, 8.0, 40.0, 1.0),
#'   disease = c("wheat_healthy", "wheat_yellow_rust", "wheat_yellow_rust", "wheat_healthy")
#' )
#' @export
compute_field_incidence <- function(is_healthy, severity_pct, disease = NULL, max_scale = 5L) {
  n_total <- length(is_healthy)
  if (n_total != length(severity_pct)) {
    stop("'is_healthy' and 'severity_pct' must have equal length.", call. = FALSE)
  }
  if (n_total == 0L) {
    return(list(
      total_inspected = 0L,
      infected_count = 0L,
      disease_incidence_pct = 0.0,
      pdi_pct = 0.0,
      mean_severity_pct = 0.0,
      risk_level = "No Data",
      disease_counts = integer(0)
    ))
  }

  is_healthy <- as.logical(is_healthy)
  # Trace infections (<2%) on explicitly diseased leaves still count as Grade 1 in Wheeler PDI
  effective_sev <- ifelse(is_healthy, 0.0, pmax(as.numeric(severity_pct), 2.5))
  grades <- ifelse(is_healthy, 0L, severity_to_grade(effective_sev))

  infected <- sum(!is_healthy, na.rm = TRUE)
  incidence_pct <- (infected / n_total) * 100.0
  pdi_pct <- (sum(grades, na.rm = TRUE) / (n_total * as.numeric(max_scale))) * 100.0
  mean_sev <- mean(ifelse(is_healthy, 0.0, as.numeric(severity_pct)), na.rm = TRUE)

  counts <- if (is.null(disease)) integer(0) else table(as.character(disease))

  list(
    total_inspected = as.integer(n_total),
    infected_count = as.integer(infected),
    disease_incidence_pct = round(incidence_pct, 2),
    pdi_pct = round(pdi_pct, 2),
    mean_severity_pct = round(mean_sev, 2),
    risk_level = evaluate_etl_risk(incidence_pct, pdi_pct),
    disease_counts = counts
  )
}

#' Load Bundled Regional Indian Rabi Surveillance Dataset
#'
#' Reads the agro-climatic Rabi disease incidence table bundled in
#' \code{inst/extdata/rabi_field_incidence_india.csv}.
#'
#' @param crop Optional character crop filter (e.g., \code{"Wheat"}, \code{"Potato"}).
#' @return A \code{data.frame} of state-wise crop disease incidence, weather, and PDI.
#' @examples
#' head(rabi_surveillance_data())
#' rabi_surveillance_data(crop = "Wheat")
#' @export
rabi_surveillance_data <- function(crop = NULL) {
  csv_path <- system.file("extdata", "rabi_field_incidence_india.csv", package = "rabidoctor")
  if (!nzchar(csv_path) || !file.exists(csv_path)) {
    # Source checkout fallback before package installation
    csv_path <- file.path("data", "rabi_field_incidence_india.csv")
  }
  df <- utils::read.csv(csv_path, stringsAsFactors = FALSE)
  if (!is.null(crop)) {
    keep <- tolower(df$crop) == tolower(crop)
    df <- df[keep, , drop = FALSE]
  }
  rownames(df) <- NULL
  df
}

#' Summarize and Rank Regional Indian Rabi Hotspots by Composite Risk Index
#'
#' Computes \code{risk_index = 0.6 * disease_incidence_pct + 0.4 * mean_pdi_pct}
#' and sorts regions from highest to lowest epidemiological priority.
#'
#' @param crop Optional character crop filter passed to \code{\link{rabi_surveillance_data}}.
#' @return Sorted \code{data.frame} with an added \code{risk_index} column.
#' @examples
#' summarize_regional_incidence()
#' @export
summarize_regional_incidence <- function(crop = NULL) {
  df <- rabi_surveillance_data(crop = crop)
  df$risk_index <- round(df$disease_incidence_pct * 0.6 + df$mean_pdi_pct * 0.4, 2)
  df[order(-df$risk_index), , drop = FALSE]
}
