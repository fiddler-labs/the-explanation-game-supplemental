import itertools
import logging
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


class MultivariateBernoulli:
    """Represents a multivariate Bernoulli random variable of M dimensions
    parameterized by an M-dimensional tensor of expected values.

    If a list of variable names is not provided, the dimensions will be named
    using integers 0, 1, 2, ..., M-1
    """
    def __init__(self, mu: np.ndarray,
                 variable_names:
                 Optional[Union[Sequence[str], Sequence[int]]] = None):
        assert np.all(np.array(mu.shape) == 2), \
            'Outcome probability tensor `mu` must have shape (2, 2, ..., 2)'
        assert np.isclose(mu.sum(), 1, rtol=0.001), \
            'Probabilities must sum to 1'
        self.mu = mu
        self.M = mu.ndim
        if variable_names is None:
            self.variable_names = np.arange(self.M)
        else:
            assert 'p(X)' not in variable_names, \
                '"p(X)" is a reserved name. ' \
                'Please use a different variable name.'
            if isinstance(variable_names[0], int):
                self.variable_names = np.array(variable_names)
            else:
                self.variable_names = np.array(variable_names, dtype=object)
            assert len(self.variable_names) == self.M, \
                f'Number of names must be equal to dimensionality of mu ' \
                f'({self.M}), not {len(self.variable_names)}'

    def _raise_on_bad_input(self, input_dict: Dict[Any, bool]):
        # verify input is sane
        for variable_name, value in input_dict.items():
            assert variable_name in self.variable_names, \
                f'Variable name {variable_name} not found in this ' \
                f'MultivariateBernoulli'
            assert isinstance(value, bool), \
                f'{variable_name} must be of type `bool`'

    def conditional(self, on: Dict[Any, bool]):
        """Return a MultivariateBernoulli representing this variable
        onditional on the variable_name -> variable value information in the
        `on` parameter."""
        self._raise_on_bad_input(on)
        if len(on) == self.M:
            raise ValueError('Cannot condition on every variable!')

        # construct the index that selects the slice of the tensor
        # corresponding to the conditional
        index = []
        new_variable_names = []
        for variable in self.variable_names:
            if variable in on:
                # if we condition on this variable, the index
                # should select the conditioned value of the variable
                index.append(int(on[variable]))
            else:
                # otherwise we will not slice this variable and
                # it will stay on as one of the variables in the
                # conditional MultivariateBernoulli returned
                index.append(slice(None))
                new_variable_names.append(variable)

        # slice the tensor
        new_mu = self.mu[tuple(index)]

        # rescale to sum to probability of 1
        new_mu = new_mu / new_mu.sum()

        return MultivariateBernoulli(new_mu, new_variable_names)

    def prob(self, outcome: Dict[Any, bool]):
        self._raise_on_bad_input(outcome)

        outcome_phrase = [f'{variable_name}={status}'
                          for variable_name, status in outcome.items()]
        LOG.debug(f'Computing P({", ".join(outcome_phrase)})')

        # sort each dimension into either marginalized (summed over)
        # or selected in the outcome (and thus we track the value selected
        # as an index)
        marginalized_dimensions = []
        outcome_index = []
        for i, variable_name in enumerate(self.variable_names):
            if variable_name in outcome:
                outcome_index.append(int(outcome[variable_name]))
            else:
                marginalized_dimensions.append(i)
        marginalized_dimensions = tuple(marginalized_dimensions)
        outcome_index = tuple(outcome_index)

        # compute probability
        marginal_tensor = self.mu.sum(axis=marginalized_dimensions)
        result = marginal_tensor[outcome_index]
        LOG.debug(f'Result of marginal_tensor[outcome_index] for outcome_'
                  f'index = {outcome_index} is {result:.3f}')
        return result

    def joint_marginal(self):
        """Creates the joint-marginal version of this variable
        such that all dimensions are distributed independently,
        but the marginals are all the same as this variable's marginals."""
        jm_mu = None
        for i in range(self.mu.ndim):
            # step 1: compute the marginal
            sum_axes = tuple(j for j in range(self.mu.ndim) if j != i)
            ith_marginal = self.mu.sum(axis=sum_axes)

            # step 2: expand the new joint-marginal expectation tensor with
            # the marginal
            jm_mu = (jm_mu[..., np.newaxis] * ith_marginal[np.newaxis, ...]
                     if jm_mu is not None
                     else ith_marginal)
        return MultivariateBernoulli(jm_mu, list(self.variable_names))

    def sampled_approximation(self, n_samples=10_000, seed=0):
        """Bootstrap samples n_samples points from this distribution and
        returns a new MultivariateBernoulli representing the empirical
        distribution of this sample."""
        rng = np.random.RandomState(seed=seed)
        possible_values = 2 ** self.M
        sampling_distribution = self.mu.flat
        samples = rng.choice(
            possible_values, n_samples, replace=True, p=sampling_distribution)
        probs = np.bincount(samples) / n_samples
        new_mu = probs.reshape(self.mu.shape)
        return MultivariateBernoulli(new_mu, list(self.variable_names))

    def uniform(self):
        """Return a uniform distribution over the space of this multivariate
        bernoulli"""
        new_mu = np.full(self.mu.shape, 2 ** -self.M)
        return MultivariateBernoulli(new_mu, list(self.variable_names))

    def iter_inputs(self):
        return itertools.product([0, 1], repeat=self.M)

    def iter_probs(self):
        return self.mu.flat

    def to_table(self):
        table = pd.DataFrame(
            self.iter_inputs(),
            columns=self.variable_names
        )
        table['p(x)'] = self.iter_probs()
        return table

    def expected(self, f: Callable,
                 conditional: Optional[Dict[Any, bool]] = None):
        """Compute E[f(x) | <conditional>]"""
        if conditional is None:
            conditional = dict()
        x = self.conditional(conditional)
        result = 0.
        for input_tuple, probability in zip(x.iter_inputs(), x.iter_probs()):
            full_input = {
                **conditional, **dict(zip(x.variable_names, input_tuple))}
            result += probability * f(**full_input)
        return result

    def corr(self):
        """TODO: implement more elegantly / faster to compute"""
        res = pd.DataFrame(np.nan, index=self.variable_names,
                           columns=self.variable_names)
        for i in range(self.M):
            for j in range(i + 1):
                e_x = self.prob({self.variable_names[i]: True})
                e_y = self.prob({self.variable_names[j]: True})
                e_xy = self.prob({self.variable_names[i]: True,
                                  self.variable_names[j]: True})
                cov = e_xy - e_x * e_y

                std_x = np.sqrt(e_x * (1 - e_x))
                std_y = np.sqrt(e_y * (1 - e_y))

                corr = cov / (std_x * std_y)
                res.iloc[i, j] = corr
                res.iloc[j, i] = corr
        return res
