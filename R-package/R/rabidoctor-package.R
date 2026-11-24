#' rabidoctor: Rabi Crop Disease Incidence and KVK Advisory Companion Package
#'
#' Companion R package for the **Rabi Crop Doctor** open-source project
#' (\url{https://github.com/lunganlungpg/rabi-crop-doctor}), designed for
#' agricultural statisticians, plant pathologists, and KVK field scouts across
#' Indian Rabi cropping belts (Wheat, Mustard, Chickpea, and Potato).
#'
#' @section Main Functions:
#' \itemize{
#'   \item \code{\link{severity_to_grade}}: Map lesion area percentage to 0--5 pathology scale.
#'   \item \code{\link{compute_field_incidence}}: Compute Plot Disease Incidence (\%) and Wheeler PDI (\%).
#'   \item \code{\link{evaluate_etl_risk}}: Classify field outbreak status against Economic Threshold Levels.
#'   \item \code{\link{rabi_surveillance_data}}: Load regional Indian Rabi incidence and weather dataset.
#'   \item \code{\link{summarize_regional_incidence}}: Rank agro-climatic hotspots by composite risk index.
#'   \item \code{\link{get_rabi_advisory}}: Retrieve symptom guides and IPM recommendations in English or Hindi.
#' }
#'
#' @docType package
#' @name rabidoctor
"_PACKAGE"
