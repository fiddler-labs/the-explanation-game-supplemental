import json
import textwrap
from typing import Any, Callable, Dict, Set

from multivariate_bernoulli import MultivariateBernoulli


class AttributionGame:
    def __init__(self,
                 model: Callable[[Dict[Any, bool]], float],
                 model_input: Dict[Any, bool],
                 distribution: MultivariateBernoulli):
        self.model = model
        self.model_input = model_input
        self.model_input_logical_inverse = {
            key: not value for key, value in self.model_input.items()}
        self.fx_on_input = self.model(**self.model_input)

        # take data distribution, the joint-marginal of that,
        # and the uniform distribution
        self.distribution = distribution
        self.joint_marginal_distribution = distribution.joint_marginal()
        self.uniform_distribution = distribution.uniform()

        # precompute the expectations across all distributions
        self.expected_fx = self.distribution.expected(model)
        self.expected_fx_joint_marginal = \
            self.joint_marginal_distribution.expected(model)
        self.expected_fx_uniform = self.uniform_distribution.expected(model)
        self.expected_fx_logical_inverse = self.model(
            **self.model_input_logical_inverse)

    def conditional_expectation_payout(self, coalition: Set[int]):
        # edge case: conditional on everything
        if len(coalition) == self.distribution.M:
            expected_fx_conditional = self.fx_on_input
        else:
            # take the conditional expectation over the input for the coalition
            expected_fx_conditional = self.distribution.expected(
                self.model, conditional=self.input_for_coalition(coalition))
        return expected_fx_conditional - self.expected_fx

    def _expectation_payout(self, coalition: Set[int],
                            distribution='marginal'):
        """Take the payout over a certain distribution"""

        # determine the distribution to use
        assert distribution in ('marginal', 'joint_marginal', 'uniform')
        if distribution == 'marginal':
            distribution = self.distribution
            expected_fx = self.expected_fx
        elif distribution == 'joint_marginal':
            distribution = self.joint_marginal_distribution
            expected_fx = self.expected_fx_joint_marginal
        else:
            distribution = self.uniform_distribution
            expected_fx = self.expected_fx_uniform

        # edge case: E[f(replace(x, X', U))] = f(x)
        if len(coalition) == distribution.M:
            expected_fx_conditional = self.fx_on_input
        # main case: compute E[f(replace(x, X', S))]
        else:
            # take the non-conditional expectation over the function
            # where input values are replaced withe the input for the coalition
            def model_with_replacement(**y):
                """f(replace(x, y, S))"""
                return self.model(
                    **{**y, **self.input_for_coalition(coalition)})
            expected_fx_conditional = distribution.expected(
                model_with_replacement)
        return expected_fx_conditional - expected_fx

    def marginal_expectation_payout(self, coalition: Set[int]):
        return self._expectation_payout(coalition, 'marginal')

    def joint_marginal_expectation_payout(self, coalition: Set[int]):
        return self._expectation_payout(coalition, 'joint_marginal')

    def uniform_expectation_payout(self, coalition: Set[int]):
        return self._expectation_payout(coalition, 'uniform')

    def logical_inverse_expectation_payout(self, coalition: Set[int]):
        combination = {**self.model_input_logical_inverse,
                       **self.input_for_coalition(coalition)}
        f_combination = self.model(**combination)
        return f_combination - self.expected_fx_logical_inverse

    def input_for_coalition(self, players: Set[int]) -> Dict[Any, bool]:
        """Gets the model input corresponding to a set of players."""
        inputs = {}
        for i in players:
            variable_name = self.distribution.variable_names[i]
            input_value = self.model_input[variable_name]
            inputs[variable_name] = input_value
        return inputs

    def _text_explanation(self, phi, expected_fx, display_width=80) -> str:
        """[Deprecated] Create a pretty-printed text explanation based upon a
        set of Shapley values."""
        assert display_width >= 20, \
            'Formatting configured for display width >=20 columns.'
        # figure out how wide to format things so they line up
        var_name_width = max(max(map(lambda phi_i: len(f'{phi_i:.3f}'), phi)),
                             max(map(len, self.distribution.variable_names)))

        # create the phi_1 + phi_2 + ... text
        attribution_equation_text = ' + '.join(
            f'{" " * (var_name_width - len(variable))}'
            f'phi_{i:02d}->"{variable:}"'
            for i, variable in enumerate(self.distribution.variable_names))
        # create the 1.23 + 3.45 + -5.67 ... text
        attribution_equation_with_numbers = ' + '.join(
            f'{phi_i:>{15 + var_name_width - len(f"{phi_i:.3f}")}.3f}'
            for phi_i in phi)

        res = '\n'.join([
            f'f(x) = {self.fx_on_input:.3f}',
            textwrap.indent(textwrap.fill(
                f'= {"E[f(X)]":>{var_name_width}}'
                f' + {attribution_equation_text}', display_width), '     '),
            textwrap.indent(textwrap.fill(
                f'= {expected_fx:>{var_name_width}.3f}'
                f' + {attribution_equation_with_numbers}', display_width),
                '     '),
            '    ' + '.'*display_width,
            f'And x = {json.dumps(self.model_input, indent=2)}'
        ])
        return res
