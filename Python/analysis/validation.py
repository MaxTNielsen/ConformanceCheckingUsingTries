# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import csv
import re
from stats import dims_25, avg_costs_25
from statistics import mean, stdev, correlation


LOGS = ["BPI_2012", "BPI_2017", "BPI_2020", "M10", "M1", "M2",
        "M3", "M4", "M5", "M6", "M7", "M8", "M9"]

LOG_TYPES_COMPL = ["completeness20", "completeness50"]

LOG_TYPES_CONF = ["confidence20", "confidence50"]

LOG_TYPES_NORMAL = ["sim", "sample"]

LOG_TYPES = LOG_TYPES_COMPL + LOG_TYPES_CONF + LOG_TYPES_NORMAL


def get_dataset_metrics(prefix_path: str, dims: dict, avg_costs: dict, regex_f: object) -> dict:
    dataset_dicts = {}
    dataset_keys = []
    for filename in os.listdir(prefix_path):
        f = os.path.join(prefix_path, filename)
        if os.path.isfile(f):
            with open(f, mode='r') as file:
                csvFile = csv.reader(file)
                filename_ = regex_f(filename, LOGS, LOG_TYPES)
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


def plot_line_plot_comparison(ditct1: dict, dict2: dict, logname: str, label1: str, label2: str, stat: str, ylabel: str):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(ditct1[logname]
             [stat][:50], label=label1, marker='.')
    plt.plot(dict2[logname][stat]
             [:50], label=label2, marker='.')
    plt.xlabel("case id")
    plt.ylabel(ylabel)
    fig.tight_layout()
    plt.legend(loc='upper right')
    plt.xticks(range(50))
    plt.show()


def plot_bar_chart_comparison(labels: list, dict1: dict, dict2: dict, stat: str, bar_labels: list, ylabel: str, title: str):
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(labels, list(dict1[stat].values()), width, label=bar_labels[0])
    rects2 = ax.bar(x + 0.15 + width/2,
                    list(dict2[stat].values()), width, label=bar_labels[1])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=90)
    plt.show()

    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.show()


#sum, mean, std
def get_statistics(dataset_dict: dict, dataset_id: list) -> dict:
    return {'sum': {id: sum(dataset_dict[id]['conf']) for id in dataset_id}, 'mean': {id: mean(dataset_dict[id]['conf']) for id in dataset_id},
            'std': {id: stdev(dataset_dict[id]['conf']) for id in dataset_id}, 'time': {id: mean(dataset_dict[id]['time']) for id in dataset_id},
            'time_std': {id: stdev(dataset_dict[id]['time']) for id in dataset_id}}


def print_stat(s_key:str, dict1:dict, dict2:dict) -> None:
    print("mean {}: {} - mean {}: {}".format(s_key, round(mean(list(
        dict1[s_key].values())),3), s_key, round(mean(list(dict2[s_key].values())),3)))


no_compl_no_conf_dict, _ = get_dataset_metrics(
    "results/tripleocc_runs/no_compl_no_conf", dims=dims_25, avg_costs=avg_costs_25, regex_f=extract_filename)
no_compl_avg_dict, _ = get_dataset_metrics(
    "results/tripleocc_runs/no_compl_avg", dims=dims_25, avg_costs=avg_costs_25, regex_f=extract_filename)
no_compl_min_dict, _ = get_dataset_metrics(
    "results/tripleocc_runs/no_compl_min", dims=dims_25, avg_costs=avg_costs_25, regex_f=extract_filename)
no_conf_ws_all_states_dict, _ = get_dataset_metrics(
    "results/tripleocc_runs/no_conf_ws_all_states", dims=dims_25, avg_costs=avg_costs_25, regex_f=extract_filename)
no_conf_ws_root_dict, dataset_keys = get_dataset_metrics(
    "results/tripleocc_runs/no_conf_ws_root", dims=dims_25, avg_costs=avg_costs_25, regex_f=extract_filename)

"""dataset keys"""
# print(dataset_keys)

log_names = ['BPI_2012', 'BPI_2017', 'BPI_2020', 'M1',
             'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']

normal_output = ['BPI_2012_sim_sample', 'BPI_2017_sim_sample', 'M1_sim', 'M2_sim',
                 'M3_sim', 'M4_sim', 'M5_sim', 'M6_sim', 'M7_sim', 'M8_sim', 'M9_sim', 'M10_sim']

confidence_output = [
    'BPI_2012_sim_confidence20_sample', 'BPI_2012_sim_confidence50_sample', 'BPI_2017_sim_confidence20_sample', 'BPI_2017_sim_confidence50_sample',
    'M1_sim_confidence20', 'M1_sim_confidence50', 'M2_sim_confidence20', 'M2_sim_confidence50', 'M3_sim_confidence20',
    'M3_sim_confidence50', 'M4_sim_confidence20', 'M4_sim_confidence50', 'M5_sim_confidence20', 'M5_sim_confidence50',
    'M6_sim_confidence20', 'M6_sim_confidence50', 'M7_sim_confidence20', 'M7_sim_confidence50', 'M8_sim_confidence20',
    'M8_sim_confidence50', 'M9_sim_confidence20', 'M9_sim_confidence50', 'M10_sim_confidence20', 'M10_sim_confidence50'
]

completeness_output = [
    'BPI_2012_sim_completeness20_sample', 'BPI_2012_sim_completeness50_sample', 'BPI_2017_sim_completeness20_sample', 'BPI_2017_sim_completeness50_sample',
    'M1_sim_completeness20', 'M1_sim_completeness50', 'M2_sim_completeness20', 'M2_sim_completeness50', 'M3_sim_completeness20',
    'M3_sim_completeness50', 'M4_sim_completeness20', 'M4_sim_completeness50', 'M5_sim_completeness20', 'M5_sim_completeness50',
    'M6_sim_completeness20', 'M6_sim_completeness50', 'M7_sim_completeness20', 'M7_sim_completeness50', 'M8_sim_completeness20',
    'M8_sim_completeness50', 'M9_sim_completeness20', 'M9_sim_completeness50', 'M10_sim_completeness20', 'M10_sim_completeness50'
]
# %%
###################################################################  confidence comparison  ##################################################################################################

"""lets make a graph to discern the difference in conformance cost case by case between standard (no conf, no compl) and confidence (confidence(avg/min), no compl"""

plot_line_plot_comparison(no_compl_no_conf_dict, no_compl_avg_dict,
                          'M1_sim_confidence50', "M1:no compl:no confi", "M1:no compl:confi avg", 'conf', "alignment cost")

# %%

"""grouped bar with standard (no conf, no compl) and confidence (confidence(avg/min), no compl)"""

no_compl_no_conf_conf_stats = get_statistics(no_compl_no_conf_dict, confidence_output)
no_compl_avg_conf_stats = get_statistics(no_compl_avg_dict, confidence_output)
no_compl_min_conf_stats = get_statistics(no_compl_min_dict, confidence_output)

plot_bar_chart_comparison(confidence_output, no_compl_no_conf_conf_stats,
                          no_compl_avg_conf_stats, 'mean', ['standard', 'avg'], "alignment cost", "avg alignment cost pr log")

plot_bar_chart_comparison(confidence_output, no_compl_no_conf_conf_stats,
                          no_compl_min_conf_stats, 'mean', ['standard', 'min'], "alignment cost", "avg alignment cost pr log")

# %%

print("no_compl_avg_conf_stats:")
print_stat('mean', no_compl_avg_conf_stats, no_compl_no_conf_conf_stats)
print_stat('std', no_compl_avg_conf_stats, no_compl_no_conf_conf_stats)

print("")

print("no_compl_min_conf_stats:")
print_stat('mean', no_compl_min_conf_stats, no_compl_no_conf_conf_stats)
print_stat('std', no_compl_min_conf_stats, no_compl_no_conf_conf_stats)

# %%

print("no_compl_avg_conf_stats:")
print_stat('time', no_compl_avg_conf_stats, no_compl_no_conf_conf_stats)

print("")

print("no_compl_min_conf_stats:")
print_stat('time', no_compl_min_conf_stats, no_compl_no_conf_conf_stats)


# %%
###################################################################  completeness comparison  ##################################################################################################


# """lets make a graph to discern the difference in conformance cost case by case between standard (no conf, no compl) and completeness (completeness(all/root), no confi"""

# plot_line_plot_comparison(no_conf_ws_root_dict, no_conf_ws_all_states_dict,
#                             'M1_sim_confidence50', "M1:compl:root", "M1:compl:all", 'conf', "alignment cost")

# plot_line_plot_comparison(no_compl_no_conf_dict, no_conf_ws_root_dict,
#                             'M1_sim_confidence50', "M1:no compl:no confi", "M1:compl:root", 'conf', "alignment cost")

# plot_line_plot_comparison(no_compl_no_conf_dict, no_conf_ws_all_states_dict,
#                             'M1_sim_confidence50', "M1:no compl:no confi", "M1:compl:all", 'conf', "alignment cost")

# %%

"""grouped bar with standard (no conf, no compl) and completeness (completeness(all/root), no confi"""

no_compl_no_conf_compl_stats=get_statistics(no_compl_no_conf_dict, completeness_output)
no_conf_ws_root_stats=get_statistics(no_conf_ws_root_dict, completeness_output)
no_conf_ws_all_stats=get_statistics(no_conf_ws_all_states_dict, completeness_output)

# %%

plot_bar_chart_comparison(completeness_output, no_compl_no_conf_compl_stats,
                            no_conf_ws_root_stats, 'mean', ['standard', 'ws_root'], "alignment cost", "avg alignment cost pr log")

plot_bar_chart_comparison(completeness_output, no_compl_no_conf_compl_stats,
                            no_conf_ws_all_stats, 'mean', ['standard', 'ws_all'], "alignment cost", "avg alignment cost pr log")

# %%
print("no_conf_ws_root_stats:")
print_stat('mean', no_conf_ws_root_stats, no_compl_no_conf_compl_stats)
print_stat('std', no_conf_ws_root_stats, no_compl_no_conf_compl_stats)

print("")

print("no_conf_ws_all_stats:")
print_stat('mean', no_conf_ws_all_stats, no_compl_no_conf_compl_stats)
print_stat('std', no_conf_ws_all_stats, no_compl_no_conf_compl_stats)

print("")

print("no_conf_ws_root_stats:")
print_stat('time', no_conf_ws_root_stats, no_compl_no_conf_compl_stats)
print_stat('time_std', no_conf_ws_root_stats, no_compl_no_conf_compl_stats)

print("")

print("no_conf_ws_all_stats:")
print_stat('time', no_conf_ws_all_stats, no_compl_no_conf_compl_stats)
print_stat('time_std', no_conf_ws_all_stats, no_compl_no_conf_compl_stats)

# %%

plot_bar_chart_comparison(completeness_output, no_compl_no_conf_compl_stats,
                          no_conf_ws_root_stats, 'time', ['standard', 'ws_root'], "time seconds", "avg seconds")

plot_bar_chart_comparison(completeness_output, no_compl_no_conf_compl_stats,
                          no_conf_ws_all_stats, 'time', ['standard', 'ws_all'], "time seconds", "avg seconds")

# %%
# compl_corr = {id: correlation(no_conf_ws_all_states_dict[id]['conf'], no_conf_ws_all_states_dict[id]['compl']) for id in completeness_output if mean(no_conf_ws_all_states_dict[id]['compl'])}

# plt.plot(list(compl_corr.keys()), list(compl_corr.values()))
# plt.title("ws_all corr plot")
# plt.xlabel("log")
# plt.ylabel("corr - cost and completeness")
# plt.legend(loc='upper right')
# plt.xticks(rotation=90)
# plt.show()

# %%

"""visualising the difference in cost between standard and completeness utilising algorithms on normal logs"""

no_compl_no_conf_compl_stats = get_statistics(
    no_compl_no_conf_dict, normal_output)
no_conf_ws_root_stats = get_statistics(no_conf_ws_root_dict, normal_output)
no_conf_ws_all_stats = get_statistics(
    no_conf_ws_all_states_dict, normal_output)


plot_bar_chart_comparison(normal_output, no_compl_no_conf_compl_stats,
                          no_conf_ws_root_stats, 'mean', ['standard', 'ws_root'], "alignment cost", "avg alignment cost pr log")

plot_bar_chart_comparison(normal_output, no_compl_no_conf_compl_stats,
                          no_conf_ws_all_stats, 'mean', ['standard', 'ws_all'], "alignment cost", "avg alignment cost pr log")

# %%
print("no_conf_ws_root_stats:")
print_stat('mean', no_conf_ws_root_stats, no_compl_no_conf_compl_stats)
print_stat('std', no_conf_ws_root_stats, no_compl_no_conf_compl_stats)

print("")

print("no_conf_ws_all_stats:")
print_stat('mean', no_conf_ws_all_stats, no_compl_no_conf_compl_stats)
print_stat('std', no_conf_ws_all_stats, no_compl_no_conf_compl_stats)

print("")

print("no_conf_ws_root_stats:")
print_stat('time', no_conf_ws_root_stats, no_compl_no_conf_compl_stats)
print_stat('time_std', no_conf_ws_root_stats, no_compl_no_conf_compl_stats)

print("")

print("no_conf_ws_all_stats:")
print_stat('time', no_conf_ws_all_stats, no_compl_no_conf_compl_stats)
print_stat('time_std', no_conf_ws_all_stats, no_compl_no_conf_compl_stats)

# %%

plot_bar_chart_comparison(normal_output, no_compl_no_conf_compl_stats,
                          no_conf_ws_root_stats, 'time', ['standard', 'ws_root'], "time seconds", "avg seconds pr log")

plot_bar_chart_comparison(normal_output, no_compl_no_conf_compl_stats,
                          no_conf_ws_all_stats, 'time', ['standard', 'ws_all'], "time seconds", "avg seconds pr log")

plot_bar_chart_comparison(normal_output, no_conf_ws_root_stats,
                          no_conf_ws_all_stats, 'time', ['ws_root', 'ws_all'], "time seconds", "avg seconds pr log")

# %%
