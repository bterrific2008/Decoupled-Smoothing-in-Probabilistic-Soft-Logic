import argparse
import os
from operator import add
from mpl_toolkits.axes_grid1 import Grid

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MultipleLocator
from sklearn.metrics import roc_auc_score, average_precision_score


def read_predictions(method, data_nm, random_seed, pct_lbl, learn_eval='eval'):
    # set the working directory and import helper functions
    # get the current working directory and then redirect into the functions under code
    cwd = os.getcwd()

    # import the results from the results folder
    results_cwd = os.path.join(os.path.abspath(cwd), 'results', 'decoupled-smoothing', learn_eval)

    # read the predictions data
    predictions = {}
    file_path = os.path.join(results_cwd, method, data_nm, '{:04d}'.format(random_seed),
                             'inferred-predicates{:02d}'.format(int(pct_lbl * 100)), 'GENDER.txt')
    with open(file_path, 'r') as f:
        for line in f:
            node, label, prob = line.strip().split('\t')
            node = int(node)
            label = int(label)
            prob = float(prob)
            if node in predictions:
                predictions[node][label] = prob
            else:
                predictions[node] = {label: prob}

    return predictions


def read_truth(data_nm, random_seed, pct_lbl, learn_eval='eval'):
    # set the working directory and import helper functions
    # get the current working directory and then redirect into the functions under code
    cwd = os.getcwd()

    # import the results from the results folder
    data_cwd = os.path.join(os.path.abspath(cwd), 'data', learn_eval, data_nm,
                            '{:02d}pct'.format(int(pct_lbl * 100)),
                            '{:04d}rand'.format(random_seed),
                            'gender_truth.txt')

    # read the truth data
    truth = {}
    with open(data_cwd, 'r') as f:
        for line in f:
            node, gender, true = line.strip().split('\t')
            if float(true) > 0:
                node = int(node)
                gender = int(gender)
                truth[node] = gender

    return truth


def read_baseline_results(filepath):
    if filepath:
        raise NotImplementedError(f'read_baseline_results {filepath}')
    return None, None


def find_tptn(predictions, truth):
    y_true = []
    y_score = []
    tp = []
    tp_score = []
    tn = []
    tn_score = []
    for node in predictions.keys():
        if node in truth:
            # find truth data per node, and associated score
            y_true.append([int(1 == truth[node]), int(2 == truth[node])])
            y_score.append([predictions[node][1], predictions[node][2]])

            # find true positives
            tp.append(int(2 == truth[node]))
            tp_score.append(predictions[node][2])

            # find true negatives
            tn.append(int(1 == truth[node]))
            tn_score.append(predictions[node][1])

    return y_true, y_score, tp, tp_score, tn, tn_score


# find roc and prc scores
def calculate_metrics(truth, score):
    roc_score = roc_auc_score(truth, score, average="weighted")
    prc_score = average_precision_score(truth, score, average="weighted")
    return roc_score, prc_score


def create_graph(pct_list, results, metric, base_ds_mean=None, base_ds_std=None):
    fig, ax = plt.subplots()
    npct_list = np.array(pct_list) * 100

    # title, labels
    ax.set_title('PSL-DS {} Scores'.format(metric), fontsize=20)
    ax.set_xlabel('Percent of Nodes Initially Labeled').set_fontsize(15)
    ax.set_ylabel(metric).set_fontsize(15)

    # fix ticks on x axis
    plt.xlim((0, 100.1))
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    # fix ticks on y axis
    plt.ylim((0.45, 0.8))
    # FixedLocator([0, 25, 50, 75, 100])
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))

    # only leave bottom-left spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if base_ds_mean and base_ds_std:
        ax.errorbar(npct_list, base_ds_mean, yerr=base_ds_std, label='DS ORG',
                    fmt='--o', elinewidth=3, capthick=2, color='blueviolet')

    # draw the error bars
    for name, mean, std, color, line_format in results:
        ax.errorbar(npct_list, mean, yerr=std, label=name, fmt=line_format, elinewidth=3,
                    capthick=2, color=color)

    # add the legend
    figlegend = plt.figure()
    ax_leg = figlegend.add_subplot(111)
    legend = ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=6, frameon=False)
    ax_leg.axis('off')
    # plt.legend(loc="upper left", prop=dict(size=8))
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # plt.show()
    plt.savefig('PSL-DS {} Figure'.format(metric), figsize=(10, 10), dpi=100)
    figlegend.savefig('PSL-DS {} Legend'.format(metric),
                      bbox_inches=legend.get_window_extent().transformed(
                          figlegend.dpi_scale_trans.inverted()))
    fig.clf()


def write_to_csv(evaluation_results, metrics, pct_list, random_seeds):
    """

    :param evaluation_results:
    :param metrics:
    :param pct_list:
    :param random_seeds:
    :return:
    """

    for metric in metrics:
        with open('{}.csv'.format(metric), 'w+') as f:
            f.write('{},{}\n'.format(
                'model',
                ','.join([str(pct) for pct in pct_list])
            ))
            averaged_model_scores = {}
            for random in evaluation_results[metric].keys():
                f.write('{}\n'.format(random))
                for model, calc_pct in evaluation_results[metric][random].items():
                    f.write('{},{}\n'.format(
                        model,
                        ','.join([str(pct) for pct in calc_pct])))
                    if model not in averaged_model_scores:
                        averaged_model_scores[model] = calc_pct
                    else:
                        averaged_model_scores[model] = list(map(add, averaged_model_scores[model],
                                                                calc_pct))
            f.write('average\n')
            for model in averaged_model_scores.keys():
                averaged_model_scores[model] = [i / len(random_seeds)
                                                for i in averaged_model_scores[model]]
                f.write('{},{}\n'.format(
                    model,
                    ','.join([str(pct) for pct in averaged_model_scores[model]])
                ))


def create_all_graphs(evaluation_results, metrics, models, pct_list, random_seeds, base_ds_mean,
                      base_ds_std):
    color_map = {'cli_one_hop': 'royalblue',
                 'cli_oh_prior': 'royalblue',
                 'cli_one_hop_gpp': 'royalblue',
                 'cli_two_hop': 'orange',
                 'cli_decoupled_smoothing': 'green',
                 'cli_decoupled_smoothing_closefriend_t200': 'red',
                 'cli_decoupled_smoothing_closefriend': 'gray'}
    label_name = {'cli_one_hop': '1-hop',
                  'cli_oh_prior': '1-Hop',
                  'cli_one_hop_gpp': '1-Hop',
                  'cli_two_hop': '2-Hop',
                  'cli_decoupled_smoothing': 'PSL-DS',
                  'cli_decoupled_smoothing_closefriend_t200': 'DS Pref Con (200)',
                  'cli_decoupled_smoothing_closefriend': 'DS Pref Con (normalized)'}
    format_guide = {'cli_one_hop': '--o',
                    'cli_oh_prior': '--o',
                    'cli_one_hop_gpp': '--o',
                    'cli_two_hop': '--o',
                    'cli_decoupled_smoothing': '-o',
                    'cli_decoupled_smoothing_closefriend_t200': '-o',
                    'cli_decoupled_smoothing_closefriend': '-o'}

    metric_name = {'tp_roc': 'AUROC',
                   'tp_prc': 'AUPRC',
                   'tn_roc': 'TN AUROC',
                   'tn_prc': 'TN AUPRC',
                   'cat': 'Categorical Accuracy'}

    for metric in metrics:
        results = []
        for model in models:
            temp_array = np.array(
                [evaluation_results[metric][seed][model] for seed in random_seeds])
            mean_values = list(temp_array.mean(axis=0))
            std_values = list(temp_array.std(axis=0))

            results.append(
                (label_name[model], mean_values, std_values, color_map[model], format_guide[model]))

        create_graph(pct_list, results, metric_name[metric], base_ds_mean, base_ds_std)


def evaluate(models, metrics, pct_list, random_seeds):
    """

    :param models:
    :param metrics:
    :param pct_list:
    :param random_seeds:
    :return:
    """

    evaluation_results = {}
    for metric in metrics:
        evaluation_results[metric] = {}
    for seed in random_seeds:
        for metric in metrics:
            evaluation_results[metric][seed] = {}
        for model in models:
            tp_roc = []
            tp_prc = []
            tn_roc = []
            tn_prc = []
            cat = []
            for pct in pct_list:
                predictions = read_predictions(model, 'Amherst41', seed, pct)
                truth = read_truth('Amherst41', seed, pct)
                y_true, y_score, tp, tp_score, tn, tn_score = find_tptn(predictions, truth)

                roc_score, prc_score = calculate_metrics(tp, tp_score)
                tp_roc.append(roc_score)
                tp_prc.append(prc_score)

                roc_score, prc_score = calculate_metrics(tn, tn_score)
                tn_roc.append(roc_score)
                tn_prc.append(prc_score)

                cat.append(np.mean(
                    np.equal(np.array(y_true).argmax(axis=-1), np.array(y_score).argmax(axis=-1))))

            evaluation_values = {'tp_roc': tp_roc,
                                 'tp_prc': tp_prc,
                                 'tn_roc': tn_roc,
                                 'tn_prc': tn_prc,
                                 'cat': cat}

            for metric in metrics:
                evaluation_results[metric][seed][model] = evaluation_values[metric]

    return evaluation_results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate PSL performance')
    parser.add_argument('-b', '--baseline', help='csv file containing baseline results')
    parser.add_argument('-g', '--graph', action='store_true', help='output a graph or not')
    parser.add_argument('-m', '--models', nargs="*", type=str,
                        default=['cli_two_hop', 'cli_one_hop', 'cli_decoupled_smoothing',
                                 'cli_decoupled_smoothing_closefriend',
                                 'cli_decoupled_smoothing_closefriend_t200'],
                        help='specify which models to evaluate')
    parser.add_argument('-e', '--evaluation_metrics', nargs="*", type=str,
                        default=['tp_roc', 'cat'],
                        help='specify which metric you want to evaluate with (default is all)')
    parser.add_argument('-p', '--percentages', nargs='*', type=float,
                        default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help='specify which percentages to evaluate')
    parser.add_argument('-r', '--random_seeds', nargs='*', type=int,
                        default=[1, 12345, 837, 2841, 4293, 6305, 6746, 9056, 9241, 9547],
                        help='specify which random seeds to evaluate')

    # TODO assert that baseline has same # as percentages

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    models = args.models
    metrics = args.evaluation_metrics
    percentages = args.percentages
    random_seeds = args.random_seeds
    graph = args.graph

    base_ds_mean, base_ds_std = read_baseline_results(args.baseline)
    evaluation_results = evaluate(models, metrics, percentages, random_seeds)
    write_to_csv(evaluation_results, metrics, percentages, random_seeds)
    if graph:
        create_all_graphs(evaluation_results, metrics, models, percentages, random_seeds,
                          base_ds_mean, base_ds_std)


if __name__ == "__main__":
    main()