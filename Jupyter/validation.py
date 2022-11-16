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


#mean, std, time
def get_statistics(dataset_dict: dict, dataset_id: list) -> dict:
    return {'sum': {id: sum(dataset_dict[id]['conf']) for id in dataset_id}, 'mean': {id: mean(dataset_dict[id]['conf']) for id in dataset_id},
            'std': {id: stdev(dataset_dict[id]['conf']) for id in dataset_id}, 'time': {id: mean(dataset_dict[id]['time']) for id in dataset_id},
            'time_std': {id: stdev(dataset_dict[id]['time']) for id in dataset_id}}


def print_stat(s_key:str, dict1:dict, dict2:dict, name1:str, name2:str) -> None:
    print("{}: mean {} all logs {} - {}: mean {} all logs {}".format(name1, s_key, round(mean(list(
        dict1[s_key].values())),3), name2, s_key, round(mean(list(dict2[s_key].values())),3)))

def print_procentual(s_key:str, dim:str, dict1:dict, dict2:dict) -> None:
    mean1 = round(mean(list(dict1[s_key].values())),3)
    mean2 = round(mean(list(dict2[s_key].values())),3)
    p = round((mean1-mean2)/mean2*100,3)
    dec_inc = "Decrease" if p < 0 else "Increase"
    print("{} {} in {}: {}".format(dec_inc, '%', dim,p))

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