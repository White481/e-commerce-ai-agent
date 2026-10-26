from evals.run_eval import results_match
from text_to_sql.executor import QueryResult


def r(columns, rows):
    return QueryResult(columns, rows, len(rows), False)


REF = r(["country", "revenue"], [["UK", 100.004], ["FR", 50.0]])


def test_same_result_matches():
    assert results_match(REF, r(["c", "rev"], [["UK", 100.0], ["FR", 50.0]]), ordered=True)


def test_column_order_and_extra_columns_are_ignored():
    actual = r(["rev", "n", "country"], [[100.0, 7, "UK"], [50.0, 3, "FR"]])
    assert results_match(REF, actual, ordered=True)


def test_row_order_matters_only_when_ordered():
    swapped = r(["country", "revenue"], [["FR", 50.0], ["UK", 100.0]])
    assert not results_match(REF, swapped, ordered=True)
    assert results_match(REF, swapped, ordered=False)


def test_different_values_or_row_count_do_not_match():
    assert not results_match(REF, r(["country", "revenue"], [["UK", 99.0], ["FR", 50.0]]), ordered=False)
    assert not results_match(REF, r(["country", "revenue"], [["UK", 100.0]]), ordered=False)


def test_rows_must_stay_aligned():
    # Same column values, but paired differently
    mixed = r(["country", "revenue"], [["UK", 50.0], ["FR", 100.0]])
    assert not results_match(REF, mixed, ordered=False)


def test_llm_failure_is_not_counted_as_a_safe_injection_answer():
    from evals.run_eval import score
    from text_to_sql.agent import Answer, Attempt

    llm_down = Answer("error", "q", message="503", attempts=[Attempt(None, "503")])
    blocked = Answer("error", "q", message="no", attempts=[Attempt("DROP TABLE x", "Only SELECT")])
    assert score({"expect": "injection"}, llm_down, None, "marts") == (False, "llm_error")
    assert score({"expect": "injection"}, blocked, None, "marts") == (True, "error")
    assert score({"expect": "cannot_answer"}, Answer("cannot_answer", "q"), None, "marts") == (True, "cannot_answer")
