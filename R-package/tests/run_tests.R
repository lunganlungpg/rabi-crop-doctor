# Base-R automated test suite for rabidoctor (runs without external CRAN deps)
library(rabidoctor)

# 1. Grade conversion boundary checks
stopifnot(identical(severity_to_grade(c(0, 2.0, 2.5, 10.0, 25.0, 50.0, 75.0, 88.0)),
                    c(0L, 0L, 1L, 1L, 2L, 3L, 4L, 5L)))

# 2. Wheeler PDI & Incidence math parity with Python/APK
# 2/4 infected = 50% incidence; grades = (0, 1, 3, 0) => sum=4 / (4*5=20) = 20.0% PDI
res <- compute_field_incidence(
  is_healthy   = c(TRUE, FALSE, FALSE, TRUE),
  severity_pct = c(0.5, 8.0, 40.0, 1.0),
  disease      = c("wheat_healthy", "wheat_yellow_rust", "wheat_yellow_rust", "wheat_healthy")
)
stopifnot(res$total_inspected == 4L)
stopifnot(res$infected_count == 2L)
stopifnot(abs(res$disease_incidence_pct - 50.0) < 1e-6)
stopifnot(abs(res$pdi_pct - 20.0) < 1e-6)

# 3. Bundled regional surveillance table check
surv <- summarize_regional_incidence()
stopifnot(nrow(surv) >= 14L)
stopifnot("risk_index" %in% names(surv))

# 4. Bilingual advisory lookup check
adv_en <- get_rabi_advisory("potato_late_blight", lang = "en")
adv_hi <- get_rabi_advisory("potato_late_blight", lang = "hi")
stopifnot(adv_en$etl_pct == 5)
stopifnot(length(adv_en$actions) >= 2L)
stopifnot(nzchar(adv_hi$name))

cat("ALL rabidoctor R PACKAGE TESTS PASSED!\n")
