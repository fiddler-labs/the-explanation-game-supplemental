import numpy as np
import pandas as pd

from real_games import substitution_payout_function_generator, shapley_values


def test_shapley_values():
    # test that the substitution payout function works
    def test_model(x):
        return np.apply_along_axis(lambda row: int(np.all(row)), 1, x)

    test_point = np.array([1, 1, 0])
    cf_point = np.array([[1, 1, 1]])
    payout_fn2 = next(
        substitution_payout_function_generator(test_model, test_point,
                                               pd.DataFrame(cf_point)))

    # since the last player breaks the 1-1-1 pattern
    assert payout_fn2({2}) == -1

    # since no other players change anything
    assert payout_fn2({0, 1}) == 0

    # test that the Shapley values work using a formulation
    # tested in the toy problem example
    def fn_complex(indicators):
        if indicators[1]:
            if indicators[2]:
                res = 0.1
            else:
                res = 0.6
        else:
            if indicators[2]:
                res = 0.3
            else:
                res = 0.0
        return res + (0.3 * indicators[0])

    def test_model2(model_input):
        return np.apply_along_axis(fn_complex, 1, model_input)

    test_sv2 = shapley_values(
        n_players=3,
        payout=next(substitution_payout_function_generator(
            test_model2, np.ones(3),
            pd.DataFrame(np.zeros(3)[np.newaxis, ...])))
    )
    assert np.all(np.isclose(test_sv2, np.array([0.3, 0.2, -0.1])))
