\name{rabidoctor}
\alias{rabidoctor}
\alias{rabidoctor-package}
\alias{severity_to_grade}
\alias{evaluate_etl_risk}
\alias{compute_field_incidence}
\alias{rabi_surveillance_data}
\alias{summarize_regional_incidence}
\alias{list_rabi_diseases}
\alias{get_rabi_advisory}
\title{Rabi Crop Disease Incidence, Severity Grading, and KVK Advisories}
\description{
Base-R epidemiological and KVK extension companion package for the
Rabi Crop Doctor project (\url{https://github.com/lunganlungpg/rabi-crop-doctor}).
Provides foliar lesion grading (0--5 scale), Plot Disease Incidence (\%),
Wheeler (1969) Percent Disease Index (PDI), state-wise Indian Rabi
agro-climatic surveillance datasets, and bilingual English/Hindi farmer advisories.
}
\usage{
severity_to_grade(severity_pct)
evaluate_etl_risk(incidence_pct, pdi_pct)
compute_field_incidence(is_healthy, severity_pct, disease = NULL, max_scale = 5L)
rabi_surveillance_data(crop = NULL)
summarize_regional_incidence(crop = NULL)
list_rabi_diseases(crop = NULL)
get_rabi_advisory(class_id, lang = c("en", "hi"))
}
\arguments{
\item{severity_pct}{Numeric vector of foliar lesion area percentages (0--100).}
\item{incidence_pct}{Numeric percentage of diseased plants/leaves.}
\item{pdi_pct}{Numeric Wheeler Percent Disease Index (0--100).}
\item{is_healthy}{Logical vector indicating healthy (\code{TRUE}) vs infected (\code{FALSE}) leaves.}
\item{disease}{Optional character vector of diagnosed disease IDs.}
\item{max_scale}{Integer maximum rating grade (default \code{5L}).}
\item{crop}{Optional crop name filter (\code{"wheat"}, \code{"mustard"}, \code{"chickpea"}, \code{"potato"}).}
\item{class_id}{Character class ID such as \code{"wheat_yellow_rust"}.}
\item{lang}{Character language code: \code{"en"} or \code{"hi"}.}
}
\examples{
severity_to_grade(c(0.5, 8.0, 22.0, 48.0))
compute_field_incidence(
  is_healthy = c(TRUE, FALSE, FALSE, TRUE),
  severity_pct = c(0.5, 8.0, 40.0, 1.0),
  disease = c("wheat_healthy", "wheat_yellow_rust", "wheat_yellow_rust", "wheat_healthy")
)
head(summarize_regional_incidence())
}
