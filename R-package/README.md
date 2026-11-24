# rabidoctor (R Companion Package)

Base-R epidemiological and KVK advisory toolkit for **Rabi Crop Doctor**.

## Installation from GitHub
```r
# Zero external CRAN dependencies required
remotes::install_github("lunganlungpg/rabi-crop-doctor", subdir = "R-package")
```

## Quick Example
```r
library(rabidoctor)

# 1. Compute Plot Disease Incidence (%) and Wheeler PDI (%)
plot_audit <- compute_field_incidence(
  is_healthy   = c(TRUE, TRUE, FALSE, FALSE, TRUE),
  severity_pct = c(0.5, 1.0, 18.0, 36.0, 0.8),
  disease      = c("wheat_healthy", "wheat_healthy", "wheat_yellow_rust", "wheat_yellow_rust", "wheat_healthy")
)
print(plot_audit)

# 2. Explore Regional Indian Rabi Hotspots
hotspots <- summarize_regional_incidence("Wheat")
print(hotspots)

# 3. Look up Bilingual Farmer IPM Advisories
adv <- get_rabi_advisory("wheat_yellow_rust", lang = "hi")
cat(adv$name, "\n", adv$symptoms, "\n")
```
