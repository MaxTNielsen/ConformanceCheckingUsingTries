import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import csv
import re
from statistics import mean, stdev


LOGS = ["BPI_2012", "BPI_2017", "BPI_2020", "M10", "M1", "M2",
        "M3", "M4", "M5", "M6", "M7", "M8", "M9"]

LOG_TYPES_COMPL = ["completeness20", "completeness50"]

LOG_TYPES_CONF = ["confidence20", "confidence50"]

LOG_TYPES_NORMAL = ["sim", "sample"]

LOG_TYPES = LOG_TYPES_COMPL + LOG_TYPES_NORMAL  # // LOG_TYPES_CONF

log_names = ['BPI_2012', 'BPI_2017', 'BPI_2020', 'M1',
             'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']

normal_output = ['BPI_2012_sample', 'BPI_2017_sample', 'M1_sim', 'M2_sim',
                 'M3_sim', 'M4_sim', 'M5_sim', 'M6_sim', 'M7_sim', 'M8_sim', 'M9_sim', 'M10_sim']

normal_labels = ['BPI2012', 'BPI2017', 'M1', 'M2',
                 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']

confidence_output = [
    'BPI_2012_sim_confidence20_sample', 'BPI_2012_sim_confidence50_sample', 'BPI_2017_sim_confidence20_sample', 'BPI_2017_sim_confidence50_sample',
    'M1_sim_confidence20', 'M1_sim_confidence50', 'M2_sim_confidence20', 'M2_sim_confidence50', 'M3_sim_confidence20',
    'M3_sim_confidence50', 'M4_sim_confidence20', 'M4_sim_confidence50', 'M5_sim_confidence20', 'M5_sim_confidence50',
    'M6_sim_confidence20', 'M6_sim_confidence50', 'M7_sim_confidence20', 'M7_sim_confidence50', 'M8_sim_confidence20',
    'M8_sim_confidence50', 'M9_sim_confidence20', 'M9_sim_confidence50', 'M10_sim_confidence20', 'M10_sim_confidence50'
]

completeness_output = [
    'BPI_2012_completeness20_sample', 'BPI_2012_completeness50_sample',
    'BPI_2017_completeness20_sample', 'BPI_2017_completeness50_sample',
    'M1_sim_completeness20', 'M1_sim_completeness50', 'M2_sim_completeness20', 'M2_sim_completeness50',
    'M3_sim_completeness20', 'M3_sim_completeness50', 'M4_sim_completeness20', 'M4_sim_completeness50',
    'M5_sim_completeness20', 'M5_sim_completeness50', 'M6_sim_completeness20', 'M6_sim_completeness50',
    'M7_sim_completeness20', 'M7_sim_completeness50', 'M8_sim_completeness20', 'M8_sim_completeness50',
    'M9_sim_completeness20', 'M9_sim_completeness50', 'M10_sim_completeness20', 'M10_sim_completeness50'
]

completeness_labels = [
    'BPI2012_20', 'BPI2012_50',
    'BPI2017_20', 'BPI2017_50',
    'M1_20', 'M1_50', 'M2_20', 'M2_50',
    'M3_20', 'M3_50', 'M4_20', 'M4_50',
    'M5_20', 'M5_50', 'M6_20', 'M6_50',
    'M7_20', 'M7_50', 'M8_20', 'M8_50',
    'M9_20', 'M9_50', 'M10_20', 'M10_50'
]

prob_labels = [
    'M1_avg_simulated.csv', 'M1_prob_only.csv',
    'M1_simulated.csv', 'M1_weighted_avg_simulated.csv',
    'M2_avg_simulated.csv', 'M2_prob_only.csv',
    'M2_simulated.csv', 'M2_weighted_avg_simulated.csv'
]


def get_dataset_metrics(prefix_path: str, dims: dict, avg_costs: dict, regex_f: object) -> dict:
    dataset_dicts = {}
    dataset_keys = []
    for filename in os.listdir(prefix_path):
        f = os.path.join(prefix_path, filename)
        if os.path.isfile(f):
            with open(f, mode='r') as file:
                csvFile = csv.reader(file)
                if regex_f:
                    filename_ = regex_f(filename, LOGS, LOG_TYPES)
                else:
                    filename_ = filename
                dataset_keys.append(filename_)
                dataset_dicts[filename_] = copy.deepcopy(avg_costs)
                for i, lines in enumerate(csvFile):
                    if i == 0:
                        continue
                    for key, val in dims.items():
                        dataset_dicts[filename_][key].append(
                            float(lines[val]))
            file.close()

    return dataset_dicts, dataset_keys


def extract_filename(filename: str, logs, log_types) -> str:
    regx = r"("+"|".join(logs)+")*("+"|".join(log_types)+")*"
    regex_match = re.findall(regx, filename)
    regex_match = [val for t in regex_match for val in t if val]
    regex_match = sorted(set(regex_match), key=lambda x: regex_match.index(x))
    return "_".join(regex_match)


def plot_bar_chart_comparison_(labels: list, dict1: dict, dict2: dict, dict3: dict, stat: str, bar_labels: list, ylabel: str, title: str, isLog: bool = False, isLegend: bool = True):
    x = np.arange(0, len(labels)*3, 3)
    width = 0.8

    fig, ax = plt.subplots(figsize=(16, 9))
    rects1 = ax.barh(
        x - width, list(dict1[stat].values()), width, label=bar_labels[0])
    rects2 = ax.barh(x, list(dict2[stat].values()), width, label=bar_labels[1])
    rects3 = ax.barh(
        x + width, list(dict3[stat].values()), width, label=bar_labels[2])

    ax.set(yticks=x + width - 0.8, yticklabels=labels,
           ylim=[-1.9, len(labels)*3-width])
    ax.set_xlabel(ylabel)

    if isLegend:
        ax.legend()

    if isLog:
        plt.xscale("log")
    plt.tight_layout()
    #plt.show()


def plot_bar_chart_comparison__(labels: list, dict1: dict, stat: str, ylabel: str, title: str, isLog: bool = False, isLegend: bool = True):
    x = np.arange(0, len(labels), 1)
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.barh(x, list(dict1[stat].values()), width)

    ax.set(yticks=x, yticklabels=labels, ylim=[-1.4, len(labels)-width])
    ax.set_xlabel(ylabel)

    if isLegend:
        ax.legend()

    if isLog:
        plt.xscale("log")
    plt.tight_layout()
    #plt.show()


#mean, std, time
def get_statistics(dataset_dict: dict, dataset_id: list) -> dict:
    return {'sum': {id: sum(dataset_dict[id]['conf']) for id in dataset_id},
            'mean': {id: mean(dataset_dict[id]['conf']) for id in dataset_id}, 'std': {id: stdev(dataset_dict[id]['conf']) for id in dataset_id},
            'time': {id: mean(np.asarray(dataset_dict[id]['time']) / np.asarray(dataset_dict[id]['trace_length'])) for id in dataset_id},
            'time_std': {id: stdev(np.asarray(dataset_dict[id]['time']) / np.asarray(dataset_dict[id]['trace_length'])) for id in dataset_id}}


def print_stat(s_key: str, dict1: dict, dict2: dict, name1: str, name2: str) -> None:
    print("{}: mean {} all logs {} - {}: mean {} all logs {}".format(name1, s_key, round(mean(list(
        dict1[s_key].values())), 3), name2, s_key, round(mean(list(dict2[s_key].values())), 3)))


def print_procentual(s_key: str, dim: str, dict1: dict, dict2: dict) -> None:
    mean1 = round(mean(list(dict1[s_key].values())), 3)
    mean2 = round(mean(list(dict2[s_key].values())), 3)
    p = round((mean1-mean2)/mean2*100, 3)
    dec_inc = "Decrease" if p < 0 else "Increase"
    return "{} {} in {}: {}".format(dec_inc, '%', dim, p)


# INPUT_DIR = os.path.join('..', 'output')

# dims_14 = {'conf': 1, 'trace_length': 2, 'time': 3}
# avg_costs_14 = {'conf': [], 'trace_length': [], 'time': []}

# dims_25 = {'conf': 1, 'compl': 2, 'confi': 3,
#            'total': 4, 'trace_length': 5, 'time': 6}
# avg_costs_25 = {'conf': [], 'compl': [], 'confi': [],
#                 'total': [], 'trace_length': [], 'time': []}

# all_prefix_prob_dict, dict_keys = get_dataset_metrics(
#     INPUT_DIR+"/tripleocc_runs/conf_prob_exp", dims=dims_25, avg_costs=avg_costs_25, regex_f=None)

# all_prefix_prob_stats = get_statistics(all_prefix_prob_dict, prob_labels)

# for key,stat in all_prefix_prob_stats['mean'].items():
#     print(key,stat)

# plot_bar_chart_comparison__(prob_labels, all_prefix_prob_stats,
#                             'mean', "alignment cost", "avg alignment cost pr log")

# INPUT_DIR = os.path.join('..', 'output')

# dims_14 = {'conf': 1, 'trace_length': 2, 'time': 3}
# avg_costs_14 = {'conf': [], 'trace_length': [], 'time': []}

# dims_25 = {'conf': 1, 'compl': 2, 'confi': 3, 'total': 4, 'trace_length': 5, 'time': 6}
# avg_costs_25 = {'conf': [], 'compl': [], 'confi': [], 'total': [], 'trace_length': [], 'time': []}

# no_compl_no_conf_dict, dict_keys = get_dataset_metrics(
#    INPUT_DIR+"/tripleocc_runs/no_compl_no_conf", dims=dims_14, avg_costs=avg_costs_14, regex_f=extract_filename)
# # no_compl_avg_dict, _ = get_dataset_metrics(
# #     INPUT_DIR+"/tripleocc_runs/no_compl_avg", dims=dims_25, avg_costs=avg_costs_25, regex_f=extract_filename)
# # no_compl_min_dict, _ = get_dataset_metrics(
# #     INPUT_DIR+"/tripleocc_runs/no_compl_min", dims=dims_25, avg_costs=avg_costs_25, regex_f=extract_filename)
# no_conf_ws_root_dict, dict_keys_ = get_dataset_metrics(
#     INPUT_DIR+"/tripleocc_runs/no_conf_ws_root", dims=dims_25, avg_costs=avg_costs_25, regex_f=extract_filename)
# no_conf_ws_all_states_dict, _  = get_dataset_metrics(
#     INPUT_DIR+"/tripleocc_runs/no_conf_ws_all_compressed", dims=dims_25, avg_costs=avg_costs_25, regex_f=extract_filename)

# no_compl_no_conf_compl_stats=get_statistics(no_compl_no_conf_dict, completeness_output)
# no_conf_ws_all_stats=get_statistics(no_conf_ws_all_states_dict, completeness_output)
# no_conf_ws_root_stats=get_statistics(no_conf_ws_root_dict, completeness_output)

# plt.rcParams.update({'font.size': 22})
# plot_bar_chart_comparison_(completeness_labels, no_compl_no_conf_compl_stats,
#                             no_conf_ws_root_stats, no_conf_ws_all_stats,'mean', ['ws_none', 'ws_root', 'ws_all'], "alignment cost", "avg alignment cost pr log")

# print("ws root stats:")
# print_stat('mean', no_conf_ws_root_stats, no_compl_no_conf_compl_stats, "ws root", "IWS")
# print_stat('std', no_conf_ws_root_stats, no_compl_no_conf_compl_stats,  "ws root", "IWS")
# print(print_procentual('mean','conformance',no_conf_ws_root_stats, no_compl_no_conf_compl_stats))
# print(40*"--")
# print("ws all stats:")
# print_stat('mean', no_conf_ws_all_stats, no_compl_no_conf_compl_stats, "ws all", "IWS")
# print_stat('std', no_conf_ws_all_stats, no_compl_no_conf_compl_stats,  "ws all", "IWS")
# print(print_procentual('mean','conformance',no_conf_ws_all_stats, no_compl_no_conf_compl_stats))