# Test run (green): after the implementation

- Run: 2026-10-14 04:02 UTC
- Command: ```python -m pytest -v```
- Python 3.11.9, pytest 9.1.1, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1

```text
============================= test session starts =============================
collecting ... collected 29 items

tests/test_data.py::test_cleaning_drops_cancellations_fees_anonymous_and_invalid_rows PASSED [  3%]
tests/test_data.py::test_interactions_have_one_row_per_customer_item_basket PASSED [  6%]
tests/test_data.py::test_item_names_pick_the_most_common_description PASSED [ 10%]
tests/test_metrics.py::test_precision_at_k_on_toy_case PASSED            [ 13%]
tests/test_metrics.py::test_precision_uses_only_top_k PASSED             [ 17%]
tests/test_metrics.py::test_precision_divides_by_k_even_when_list_is_short PASSED [ 20%]
tests/test_metrics.py::test_recall_at_k_on_toy_case PASSED               [ 24%]
tests/test_metrics.py::test_recall_is_one_when_all_relevant_retrieved PASSED [ 27%]
tests/test_metrics.py::test_recall_with_no_relevant_items_is_zero PASSED [ 31%]
tests/test_metrics.py::test_ndcg_perfect_ranking_is_one PASSED           [ 34%]
tests/test_metrics.py::test_ndcg_on_toy_case PASSED                      [ 37%]
tests/test_metrics.py::test_ndcg_two_relevant_ranks_one_and_three PASSED [ 41%]
tests/test_metrics.py::test_ndcg_zero_when_nothing_relevant_retrieved PASSED [ 44%]
tests/test_metrics.py::test_ndcg_ideal_dcg_is_capped_at_k PASSED         [ 48%]
tests/test_models.py::test_recommendations_never_include_items_already_bought_in_training PASSED [ 51%]
tests/test_models.py::test_recommendations_are_at_most_k_and_ranked_lists PASSED [ 55%]
tests/test_models.py::test_results_are_deterministic PASSED              [ 58%]
tests/test_models.py::test_different_seeds_change_the_als_factors PASSED [ 62%]
tests/test_models.py::test_unknown_user_falls_back_to_popularity PASSED  [ 65%]
tests/test_models.py::test_popularity_ranks_by_distinct_buyers PASSED    [ 68%]
tests/test_models.py::test_collaborative_models_beat_popularity_on_planted_taste_groups PASSED [ 72%]
tests/test_models.py::test_explanations_point_to_items_the_user_bought PASSED [ 75%]
tests/test_models.py::test_als_fold_in_reproduces_trained_user_factor PASSED [ 79%]
tests/test_split.py::test_temporal_split_has_no_leakage PASSED           [ 82%]
tests/test_split.py::test_split_is_strictly_before_cutoff PASSED         [ 86%]
tests/test_split.py::test_split_covers_every_interaction_once PASSED     [ 89%]
tests/test_split.py::test_test_fraction_is_approximately_respected PASSED [ 93%]
tests/test_split.py::test_split_rejects_bad_fraction PASSED              [ 96%]
tests/test_split.py::test_split_does_not_depend_on_row_order PASSED      [100%]

============================= 29 passed in 0.53s ==============================
```

