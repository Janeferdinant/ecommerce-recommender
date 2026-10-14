# Test run (green): after the implementation

- Run: 2026-10-14 03:56 UTC
- Command: ```python -m pytest -v```
- Python 3.11.9, pytest 9.1.1, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1

```text
============================= test session starts =============================
collecting ... collected 26 items

tests/test_metrics.py::test_precision_at_k_on_toy_case PASSED            [  3%]
tests/test_metrics.py::test_precision_uses_only_top_k PASSED             [  7%]
tests/test_metrics.py::test_precision_divides_by_k_even_when_list_is_short PASSED [ 11%]
tests/test_metrics.py::test_recall_at_k_on_toy_case PASSED               [ 15%]
tests/test_metrics.py::test_recall_is_one_when_all_relevant_retrieved PASSED [ 19%]
tests/test_metrics.py::test_recall_with_no_relevant_items_is_zero PASSED [ 23%]
tests/test_metrics.py::test_ndcg_perfect_ranking_is_one PASSED           [ 26%]
tests/test_metrics.py::test_ndcg_on_toy_case PASSED                      [ 30%]
tests/test_metrics.py::test_ndcg_two_relevant_ranks_one_and_three PASSED [ 34%]
tests/test_metrics.py::test_ndcg_zero_when_nothing_relevant_retrieved PASSED [ 38%]
tests/test_metrics.py::test_ndcg_ideal_dcg_is_capped_at_k PASSED         [ 42%]
tests/test_models.py::test_recommendations_never_include_items_already_bought_in_training PASSED [ 46%]
tests/test_models.py::test_recommendations_are_at_most_k_and_ranked_lists PASSED [ 50%]
tests/test_models.py::test_results_are_deterministic PASSED              [ 53%]
tests/test_models.py::test_different_seeds_change_the_als_factors PASSED [ 57%]
tests/test_models.py::test_unknown_user_falls_back_to_popularity PASSED  [ 61%]
tests/test_models.py::test_popularity_ranks_by_distinct_buyers PASSED    [ 65%]
tests/test_models.py::test_collaborative_models_beat_popularity_on_planted_taste_groups PASSED [ 69%]
tests/test_models.py::test_explanations_point_to_items_the_user_bought PASSED [ 73%]
tests/test_models.py::test_als_fold_in_reproduces_trained_user_factor PASSED [ 76%]
tests/test_split.py::test_temporal_split_has_no_leakage PASSED           [ 80%]
tests/test_split.py::test_split_is_strictly_before_cutoff PASSED         [ 84%]
tests/test_split.py::test_split_covers_every_interaction_once PASSED     [ 88%]
tests/test_split.py::test_test_fraction_is_approximately_respected PASSED [ 92%]
tests/test_split.py::test_split_rejects_bad_fraction PASSED              [ 96%]
tests/test_split.py::test_split_does_not_depend_on_row_order PASSED      [100%]

============================= 26 passed in 0.59s ==============================
```

