# Prespecified supportive FLAME analysis (contract v4.1, role: supportive_only)
# Package flameRisk pinned at commit 85a6d1b81dcbdd1f45b488d13567cbc5577649bc
suppressMessages({
  library(flameRisk)
  library(jsonlite)
})

d <- read.csv("/tmp/pkg_run/combined_features/COMBINED_FEATURES_V4.csv", stringsAsFactors = FALSE)
d$obs <- tolower(as.character(d$outcome_observed_48h)) %in% c("true", "1", "yes")
d$y <- suppressWarnings(as.numeric(d$early_aki_48h))
d <- d[d$obs & !is.na(d$y), ]

first <- substr(tolower(as.character(d$sex)), 1, 1)
d$sex_group <- ifelse(first == "f", "Female", ifelse(first == "m", "Male", "Unknown"))
d$asa_group <- as.character(round(suppressWarnings(as.numeric(d$asa))))
d$age10 <- (suppressWarnings(as.numeric(d$age)) - 60) / 10
d$egfr10 <- (pmin(suppressWarnings(as.numeric(d$baseline_egfr)), 120) - 60) / 10
d$anesthesia_hours <- suppressWarnings(as.numeric(d$anesthesia_minutes)) / 60
d$mean_depth5 <- suppressWarnings(as.numeric(d$mean_depth_below_65_mmhg)) / 5
d$surgical_category <- as.character(d$surgical_category)
d$center <- as.character(d$center)

keep <- c("age10", "egfr10", "anesthesia_hours", "mean_depth5", "y")
d <- d[stats::complete.cases(d[, keep]), ]
cat(sprintf("analytic rows: %d   events: %d\n", nrow(d), sum(d$y)))

episodes <- do.call(rbind, lapply(seq_len(nrow(d)), function(i) {
  v <- fromJSON(d$episode_durations_json[i])
  if (length(v) == 0) return(NULL)
  data.frame(id = d$analysis_case[i], episode = seq_along(v), duration = as.numeric(v))
}))
cat(sprintf("episode rows: %d   max episodes: %d\n", nrow(episodes), max(episodes$episode)))

data.reg <- data.frame(
  id = d$analysis_case, y = d$y, center = d$center,
  age10 = d$age10, sex_group = d$sex_group, asa_group = d$asa_group,
  egfr10 = d$egfr10, surgical_category = d$surgical_category,
  anesthesia_hours = d$anesthesia_hours, mean_depth5 = d$mean_depth5,
  stringsAsFactors = FALSE
)

# Patients without any hypotensive episode contribute no accumulation, but the
# package requires every subject to appear in the episode table.
missing_ids <- setdiff(data.reg$id, episodes$id)
if (length(missing_ids) > 0) {
  episodes <- rbind(episodes,
                    data.frame(id = missing_ids, episode = 1L, duration = 0))
  cat(sprintf("added %d zero-duration rows for patients without hypotension\n", length(missing_ids)))
}

form <- y ~ C(center) + age10 + C(sex_group) + C(asa_group) + egfr10 +
  C(surgical_category) + anesthesia_hours + mean_depth5

# Prespecified clinical contrast: one 20-minute episode vs four 5-minute episodes.
# Both scenarios total 20 hypotensive minutes; mean depth is held at 5 mmHg.
scenario_matrix <- matrix(c(20, 5, 0, 5, 0, 5, 0, 5), nrow = 2)
colnames(scenario_matrix) <- paste0("ep", 1:4)

results <- list()
for (k_value in c(4, 5, 6)) {
  cat(sprintf("\n=== fitting FLAME with k = %d ===\n", k_value))
  fit <- try(flameLFT(form, binomial(link = "logit"), data.reg, episodes,
                      var.id = "id", k = k_value), silent = TRUE)
  if (inherits(fit, "try-error")) {
    cat("FIT FAILED: ", as.character(fit), "\n")
    results[[paste0("k", k_value)]] <- list(status = "failed", message = as.character(fit))
    next
  }
  cat("converged; edf summary:\n")
  print(summary(fit$model)$s.table)

  # CompareRisk expects one covariate row per scenario. Use the pooled cohort's
  # typical profile (medians for continuous, modes for categorical).
  typical <- function(x) {
    if (is.numeric(x)) return(median(x))
    ux <- unique(x); ux[which.max(tabulate(match(x, ux)))]
  }
  # Factor columns must carry the fitted model's full level sets so that
  # predict() can build contrasts from a single profile.
  mkfactor <- function(value, column) {
    factor(value, levels = sort(unique(as.character(data.reg[[column]]))))
  }
  profile <- data.frame(
    center = mkfactor(typical(data.reg$center), "center"),
    age10 = typical(data.reg$age10),
    sex_group = mkfactor(typical(data.reg$sex_group), "sex_group"),
    asa_group = mkfactor(typical(data.reg$asa_group), "asa_group"),
    egfr10 = typical(data.reg$egfr10),
    surgical_category = mkfactor(typical(data.reg$surgical_category), "surgical_category"),
    anesthesia_hours = typical(data.reg$anesthesia_hours),
    mean_depth5 = 1.0
  )
  newdata.reg <- rbind(profile, profile)
  rownames(newdata.reg) <- NULL
  cat("covariate profile used for the contrast:\n"); print(profile)
  cmp <- try(CompareRisk(fit$model, scenario_matrix, newdata.reg,
                         contrast = "1-2", n.sims = 2000), silent = TRUE)
  if (inherits(cmp, "try-error")) {
    cat("CONTRAST FAILED: ", as.character(cmp), "\n")
    results[[paste0("k", k_value)]] <- list(status = "fit_ok_contrast_failed")
    next
  }
  cat("--- CompareRisk (scenario 1 = one 20-min episode; scenario 2 = four 5-min episodes) ---\n")
  print(cmp)
  results[[paste0("k", k_value)]] <- list(status = "ok", table = cmp)
}

saveRDS(results, "/tmp/work/FLAME_RESULTS.rds")
cat("\nsaved /tmp/work/FLAME_RESULTS.rds\n")
