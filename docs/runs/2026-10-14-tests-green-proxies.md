# Test and lint run (green): after the proxies, tuning and bootstrap changes

- Run: 2026-10-14 04:42 UTC
- Commands: ```python -m pytest -v``` and ```python -m ruff check .``` (ruff 0.17.0)
- Python 3.11.9, pytest 9.1.1, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1

```text
============================= test session starts =============================
collecting ... collected 48 items

tests/test_data.py::test_cleaning_drops_cancellations_fees_anonymous_and_invalid_rows PASSED [  2%]
tests/test_data.py::test_interactions_have_one_row_per_customer_item_basket PASSED [  4%]
tests/test_data.py::test_interactions_carry_the_unit_price_of_the_line PASSED [  6%]
tests/test_data.py::test_item_names_pick_the_most_common_description PASSED [  8%]
tests/test_metrics.py::test_precision_at_k_on_toy_case PASSED            [ 10%]
tests/test_metrics.py::test_precision_uses_only_top_k PASSED             [ 12%]
tests/test_metrics.py::test_precision_divides_by_k_even_when_list_is_short PASSED [ 14%]
tests/test_metrics.py::test_recall_at_k_on_toy_case PASSED               [ 16%]
tests/test_metrics.py::test_recall_is_one_when_all_relevant_retrieved PASSED [ 18%]
tests/test_metrics.py::test_recall_with_no_relevant_items_is_zero PASSED [ 20%]
tests/test_metrics.py::test_ndcg_perfect_ranking_is_one PASSED           [ 22%]
tests/test_metrics.py::test_ndcg_on_toy_case PASSED                      [ 25%]
tests/test_metrics.py::test_ndcg_two_relevant_ranks_one_and_three PASSED [ 27%]
tests/test_metrics.py::test_ndcg_zero_when_nothing_relevant_retrieved PASSED [ 29%]
tests/test_metrics.py::test_ndcg_ideal_dcg_is_capped_at_k PASSED         [ 31%]
tests/test_metrics.py::test_hit_rate_is_one_when_any_recommendation_is_relevant PASSED [ 33%]
tests/test_metrics.py::test_hit_rate_ignores_hits_beyond_k_and_empty_truth PASSED [ 35%]
tests/test_metrics.py::test_revenue_at_k_sums_unit_prices_of_recommended_items_actually_bought PASSED [ 37%]
tests/test_metrics.py::test_revenue_at_k_respects_k_and_missing_prices PASSED [ 39%]
tests/test_metrics.py::test_catalogue_coverage_is_distinct_recommended_items_over_catalogue_size PASSED [ 41%]
tests/test_metrics.py::test_catalogue_coverage_needs_a_catalogue PASSED  [ 43%]
tests/test_metrics.py::test_relative_uplift PASSED                       [ 45%]
tests/test_models.py::test_recommendations_never_include_items_already_bought_in_training PASSED [ 47%]
tests/test_models.py::test_recommendations_are_at_most_k_and_ranked_lists PASSED [ 50%]
tests/test_models.py::test_results_are_deterministic PASSED              [ 52%]
tests/test_models.py::test_different_seeds_change_the_als_factors PASSED [ 54%]
tests/test_models.py::test_unknown_user_falls_back_to_popularity PASSED  [ 56%]
tests/test_models.py::test_popularity_ranks_by_distinct_buyers PASSED    [ 58%]
tests/test_models.py::test_collaborative_models_beat_popularity_on_planted_taste_groups PASSED [ 60%]
tests/test_models.py::test_explanations_point_to_items_the_user_bought PASSED [ 62%]
tests/test_models.py::test_als_fold_in_reproduces_trained_user_factor PASSED [ 64%]
tests/test_proxies.py::test_proxies_on_a_hand_computed_toy_case PASSED   [ 66%]
tests/test_proxies.py::test_prices_for_revenue_never_come_from_the_test_period PASSED [ 68%]
tests/test_proxies.py::test_revenue_is_omitted_without_a_price_column PASSED [ 70%]
tests/test_proxies.py::test_evaluate_reports_all_proxy_metrics_for_every_model PASSED [ 72%]
tests/test_proxies.py::test_detailed_evaluation_exposes_per_customer_values PASSED [ 75%]
tests/test_proxies.py::test_mean_sd_uses_sample_standard_deviation PASSED [ 77%]
tests/test_proxies.py::test_bootstrap_uplift_ci_is_deterministic_and_ordered PASSED [ 79%]
tests/test_proxies.py::test_bootstrap_uplift_ci_collapses_for_a_constant_ratio PASSED [ 81%]
tests/test_split.py::test_temporal_split_has_no_leakage PASSED           [ 83%]
tests/test_split.py::test_split_is_strictly_before_cutoff PASSED         [ 85%]
tests/test_split.py::test_split_covers_every_interaction_once PASSED     [ 87%]
tests/test_split.py::test_test_fraction_is_approximately_respected PASSED [ 89%]
tests/test_split.py::test_split_rejects_bad_fraction PASSED              [ 91%]
tests/test_split.py::test_split_does_not_depend_on_row_order PASSED      [ 93%]
tests/test_tuning.py::test_validation_window_sits_inside_the_training_period PASSED [ 95%]
tests/test_tuning.py::test_tune_als_returns_a_grid_member_and_a_full_table PASSED [ 97%]
tests/test_tuning.py::test_tuning_is_deterministic PASSED                [100%]

============================= 48 passed in 0.78s ==============================
```

Lint:

```text
All checks passed!
```

