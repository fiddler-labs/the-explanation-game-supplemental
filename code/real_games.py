import heapq
import itertools
import time
from typing import Callable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm


def _get_row_df(row: Union[np.ndarray, pd.DataFrame]):
    """Verifies/promotes to single-row 2d input format as a DataFrame"""
    if row.ndim == 1:
        try:
            row = row[np.newaxis, ...]
        except Exception:
            raise ValueError('Row must be an array or DataFrame')
    assert row.ndim == 2
    assert row.shape[0] == 1
    return pd.DataFrame(row)


def substitution_payout_function_generator(
        model,
        model_input: Union[np.ndarray, pd.DataFrame],
        counterfactuals: Union[np.ndarray, pd.DataFrame],
        max_chunksize=1_000):
    """A paralellized/buffered implementation that pre-evaluates all possible
     payout function inputs and returns a glorified lookup table."""

    # adjust input format as needed
    model_input = _get_row_df(model_input)

    # get all counterfactual predictions
    counterfactual_pred = model(counterfactuals)

    n_players = model_input.shape[1]
    n_payouts = 2 ** n_players

    # construct mapping a coalition tuple -> integer for table lookup
    coalition_table = dict()
    for i in range(n_payouts):
        binary_list = tuple(map(int, bin(i)[2:]))
        binary_list = np.pad(binary_list,
                             pad_width=(n_players - len(binary_list), 0),
                             mode='constant')
        coalition_table[tuple(np.where(binary_list)[0])] = i

    def payout_function_from_payout_table(payout_table):
        return lambda s: payout_table[coalition_table[tuple(s)]]

    # create a generator for composite inputs as lists
    mi_list = model_input.iloc[0].tolist()
    cf_list_of_lists = counterfactuals.values.tolist()
    composite_inputs = itertools.chain.from_iterable(
        itertools.product(*zip(cf_list, mi_list))
        for cf_list in tqdm.tqdm(cf_list_of_lists, smoothing=0)
    )

    # iterate composite inputs in tuples of "chunksize" elements at a time
    max_rows_per_chunk = n_payouts * max_chunksize
    composite_input_chunk_iterator = iter(
        lambda: list(itertools.islice(composite_inputs, max_rows_per_chunk)),
        list())
    cf_pred_chunk_iterator = np.array_split(counterfactual_pred,
                                            range(max_chunksize,
                                                  len(counterfactuals),
                                                  max_chunksize))

    # buffered by max_chunksize, run model inference
    # and generate payout functions
    for input_chunk, cf_pred_chunk in zip(composite_input_chunk_iterator,
                                          cf_pred_chunk_iterator):
        # cast input as typed DataFrame
        input_df = pd.DataFrame(input_chunk, columns=model_input.columns)
        for column_name in input_df:
            input_df[column_name] = input_df[column_name].astype(
                model_input[column_name].dtype)

        # run predictions
        chunk_pred = model(input_df)

        # split chunks back to individual counterfactuals and
        # compute payout tables
        pred_tables = np.split(chunk_pred, len(cf_pred_chunk))
        payout_tables = (tuple(pred_table - cf_pred_chunk[i])
                         for i, pred_table in enumerate(pred_tables))

        for pt in payout_tables:
            yield payout_function_from_payout_table(pt)


def slow_shapley_values(n_players: int,
                        payout: Callable[[Sequence[int]], float]):
    """Computes the exact shapley values for a payout function v(S)

    NOTE: This is a slower algorithm that grows in factorial time rather than
    exponential.
    """
    values = np.zeros(n_players)
    for perm in itertools.permutations(range(n_players)):
        coalition = set()
        past_payout = payout(coalition)
        for player in perm:
            coalition.add(player)
            payout_after_adding = payout(coalition)
            values[player] += (payout_after_adding - past_payout)
            past_payout = payout_after_adding

    values = values / np.math.factorial(n_players)
    return values


def shapley_values(n_players: int,
                   payout: Callable[[Sequence[int]], float]):
    """Computes the exact shapley values for a payout function v(S)"""

    # precompute shapley weights
    shapley_weights = list()
    for pre_player_coalition_size in range(n_players):
        shapley_weights.append(
             np.math.factorial(pre_player_coalition_size)
             * np.math.factorial(n_players - pre_player_coalition_size - 1)
             / np.math.factorial(n_players)
        )

    # compute each player's shapley value one at a time
    values = list()
    for player_id in range(n_players):
        value = 0

        # start with the emtpy set prior
        value += shapley_weights[0] * (payout((player_id,)) - payout(tuple()))

        # then run sizes 1 to n_players
        other_players = tuple(i for i in range(n_players) if i != player_id)
        for pre_player_coalition_size in range(1, n_players):
            weight = shapley_weights[pre_player_coalition_size]
            for pre_player_coalition in itertools.combinations(
                    other_players, pre_player_coalition_size):
                value += weight * (
                        payout(
                            heapq.merge(pre_player_coalition, (player_id,)))
                        - payout(pre_player_coalition)
                )
        values.append(value)
    return values


def multi_shapley_values(model, explained_point, counterfactuals):
    phi_0 = pd.DataFrame(model(counterfactuals), index=counterfactuals.index,
                         columns=['phi_0'])
    phi = pd.DataFrame(0., index=counterfactuals.index,
                       columns=counterfactuals.columns)
    print('pre-computing payout functions...')
    time.sleep(0.5)
    payout_functions = tuple(
        substitution_payout_function_generator(
            model, explained_point, counterfactuals))
    time.sleep(0.5)
    print('computing shapley values...')
    time.sleep(0.5)
    for i, payout_fn in enumerate(tqdm.tqdm(payout_functions)):
        sv = shapley_values(n_players=explained_point.shape[1],
                            payout=payout_fn)
        phi.iloc[i] = sv
    return phi_0, phi


def plot_phi(phi: pd.DataFrame, whis: float = 1.5,
             outlier_sample: Optional[int] = None):
    """Plots a set of boxplots with jittered fliers to give a sense of
    outlier density."""
    # manually identify points that will be "fliers" in the boxplot
    df = phi.melt(value_name='attribution', var_name='feature')
    p25, p75 = phi.quantile(.25), phi.quantile(.75)
    iqr = p75 - p25
    low_whis = p25 - whis * iqr
    high_whis = p75 + whis * iqr
    low_whis_melted = df['feature'].apply(low_whis.to_dict().get)
    high_whis_melted = df['feature'].apply(high_whis.to_dict().get)
    is_flier = df['attribution'].lt(low_whis_melted) | df['attribution'].gt(
        high_whis_melted)
    flier_df = df[is_flier]

    # subsample the fliers
    if outlier_sample is not None:
        def sample_fliers(flier_series, random_state=0):
            if flier_series.shape[0] <= outlier_sample:
                return flier_series
            return flier_series.sample(outlier_sample,
                                       random_state=random_state)

        flier_sample = flier_df.groupby('feature').apply(sample_fliers)
    else:
        flier_sample = flier_df

    # fix where there are no fliers for a level
    flier_sample = pd.concat(
        [
            pd.DataFrame([
                {'feature': feat, 'attribution': phi[feat].median()}
                for feat in phi.columns]),
            flier_sample
        ],
        axis=0,
        sort=False
    )

    # #  ALTERNATE: plot a violin plot
    # ax = sns.violinplot(x='attribution', y='feature', data=df,
    #                     palette=itertools.cycle(sns.color_palette()[1:]),
    #                     scale='width', linewidth=2.5, cut=0)

    # plot the boxplot without fliers
    ax = sns.boxplot(x='attribution', y='feature', data=df, whis=whis,
                     palette=itertools.cycle(sns.color_palette()[1:]),
                     showfliers=False, linewidth=2.5)
    # plot the fliers with jitter
    ax = sns.stripplot(x='attribution', y='feature', data=flier_sample,
                       color='black', ax=ax, size=1.5, alpha=0.25)
    # # OPTIONAL: plot a gray violin plot
    # ax = sns.violinplot(x='attribution', y='feature', data=df, color='black',
    #                     linewidth=0, ax=ax, scale='width')
    # plt.setp(ax.collections, alpha=.5)  # set the opacity

    return ax


def plot_attribution_scatters(phi_0, phi,
                              n_col=3, subplot_size=4):
    n_row = int(np.ceil(phi.shape[1] / n_col))
    figsize = (n_col * subplot_size, n_row * subplot_size)
    # adjust scatterplot args depending on number of points
    scatter_size, scatter_alpha = ((1.0, 0.3) if phi.shape[0] > 1_000
                                   else (2.0, 0.5))
    with sns.axes_style('white'):
        fig, axes = plt.subplots(
            n_row, n_col, figsize=figsize, sharex=True, sharey=True,
            gridspec_kw={'hspace': 0, 'wspace': 0})

        # create the scatterplots
        sort_order = phi_0['phi_0'].argsort()
        sorted_phi = phi.iloc[sort_order]
        sorted_phi_0 = phi_0.iloc[sort_order]['phi_0']
        for (i, feat), color in zip(enumerate(sorted_phi),
                                    itertools.cycle(sns.color_palette()[1:])):
            ax = axes[i // 3, i % 3]
            # add a gridline through zero
            ax.axhline(0.0, color='black', lw=1, alpha=0.5, zorder=.9)
            # plot the mean
            mean_line = ax.axhline(sorted_phi[feat].mean(),
                                   color=sns.color_palette()[0],
                                   lw=2, zorder=.9)
            ax.scatter(sorted_phi_0, sorted_phi[feat], s=scatter_size,
                       alpha=scatter_alpha, marker='.', color=color)
            ax.set_title(feat, pad=-18)

        # make a big "ghost" axis to add a single set of x/y axis labels
        ghost_axis = fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        ghost_axis.tick_params(labelcolor='none', top=False, bottom=False,
                               left=False, right=False)
        # set the global labels
        ghost_axis.set_ylabel('Attribution', labelpad=16)
        ghost_axis.set_xlabel('Model Output', labelpad=16)

        plt.legend([mean_line], ['Mean Attribution'], loc='lower right')

    return fig


def plot_attributions_over_prediction(
        phi_0, phi, counterfactual_name: Optional[str] = None,
        window_size=100, error_alpha=0.1):
    # sort by phi_0
    phi_0 = phi_0['phi_0']
    sort_order = phi_0.argsort()
    phi_0 = phi_0.iloc[sort_order]
    phi = phi.iloc[sort_order]

    # compute rolling mean/std as center and error interval
    roller = phi.rolling(window_size, min_periods=window_size, center=True)
    center = roller.mean()
    half_error_width = roller.std()
    error_lower = center - half_error_width
    error_upper = center + half_error_width

    # plot mean and error range
    for feat, color in zip(center, itertools.cycle(sns.color_palette()[1:])):
        plt.plot(phi_0, center[feat], color=color, label=feat)
        plt.fill_between(phi_0, error_lower[feat], error_upper[feat],
                         color=color, alpha=error_alpha)
    plt.ylabel(f'Average Attribution (smoothing windowsize={window_size})')
    plt.xlabel('Counterfactual Prediction')
    plt.legend()


def result_plots(phi_0, phi, counterfactual_name: Optional[str] = None,
                 fig_path=None, figname=None):
    """Plots attributions via a few different visualizations.
    If counterfactual_name is not given, then no titles are added to plots."""
    ax = plot_phi(phi)
    if counterfactual_name is not None:
        title = (f'Distribution of Attributions'
                 f'\nwith {counterfactual_name} Counterfactual')
        ax.set_title(title)
    if fig_path is not None and figname is not None:
        plt.savefig(fig_path / f'{figname}_attribution_distribution.png',
                    bbox_inches='tight')
    plt.show()

    plot_attributions_over_prediction(phi_0, phi)
    if counterfactual_name is not None:
        title = (f'Average Attributions vs. Model Outputs'
                 f'\nfor {counterfactual_name} Counterfactual')
        plt.title(title)
    if fig_path is not None and figname is not None:
        plt.savefig(fig_path / f'{figname}_mean_attribution_vs_score.png',
                    bbox_inches='tight')
    plt.show()

    fig = plot_attribution_scatters(phi_0, phi)
    if counterfactual_name is not None:
        title = (f'Attributions vs. Model Outputs'
                 f'\nfor {counterfactual_name} Counterfactual')
        fig.suptitle(title)
    if fig_path is not None and figname is not None:
        plt.savefig(fig_path / f'{figname}_attribution_vs_score_scatter.png',
                    bbox_inches='tight')
    plt.show()
