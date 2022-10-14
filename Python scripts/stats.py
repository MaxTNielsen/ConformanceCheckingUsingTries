import csv
from statistics import mean
import copy


dims_14 = {'conf': 1, 'time': 2}
avg_costs_14 = {'conf': [], 'time': []}

dims_25 = {'conf': 1, 'compl': 2, 'confi': 3, 'total': 4, 'time': 5}
avg_costs_25 = {'conf': [], 'compl': [], 'confi': [], 'total': [], 'time': []}


def summarize_results(prefix_path, dims, avg_costs, *files):

    for file_name in files:
        with open(prefix_path+file_name, mode='r') as file:
            csvFile = csv.reader(file)
            avg_costs_ = copy.copy(avg_costs)
            for i, lines in enumerate(csvFile):
                if i == 0:
                    continue
                for key, val in dims.items():
                    avg_costs_[key].append(float(lines[val]))
        file.close()

        for key, val in avg_costs_.items():
            avg_costs_[key] = round(mean(val), 2)

        if (len(avg_costs) == 2):
            print_results_14(avg_costs=avg_costs_, file_name=file_name)
        else:
            print_results_25(avg_costs=avg_costs_, file_name=file_name)

    print("")


def print_results_14(avg_costs, file_name):
    print("Avg conf {} - Avg time {} - for {} ".format(
        avg_costs['conf'], avg_costs['time'], file_name))


def print_results_25(avg_costs, file_name):
    print("Avg conf {} - Avg compl {} - Avg confi {} - Avg total {} - Avg time {} - for {} ".format(
        avg_costs['conf'],
        avg_costs['compl'],
        avg_costs['confi'],
        avg_costs['total'],
        avg_costs['time'],
        file_name
    ))


def compare_to_confidence(paths):
    avg_costs = dict()
    for i, (prefix_path, file_name) in enumerate(paths):
        with open(prefix_path+file_name, mode='r') as file:
            csvFile = csv.reader(file)
            avg_costs[i] = []
            for j, lines in enumerate(csvFile):
                if j == 0:
                    continue
                avg_costs[i].append(float(lines[dims_25['total']]))

            avg_costs[i] = mean(avg_costs[i])
        file.close()
    return avg_costs

################################################# without confidence #################################################


PREFIX_PATH_NO_CONF = "output files/conf, compl/"

print("")
print("#"*30+" without confidence "+"#"*30)
print("")

warm_start_runs_old_no_conf = [
    "M1_simulated_M1.xes_14_.csv",
    "M1_simulated_M1_warm_2.xes_14_.csv",
    "M1_simulated_M1_warm_5.xes_14_.csv",
    "M1_simulated_M1_simulated_long.xes_14_.csv",
    "M1_simulated_M1_simulated_short.xes_14_.csv"
]


summarize_results(PREFIX_PATH_NO_CONF, dims_14, avg_costs_14,
                  *warm_start_runs_old_no_conf)


warm_start_runs_new_no_conf = [
    "M1_simulated_M1.xes_25_.csv",
    "M1_simulated_M1_warm_2.xes_25_.csv",
    "M1_simulated_M1_warm_5.xes_25_.csv",
    "M1_simulated_M1_simulated_long.xes_25_.csv",
    "M1_simulated_M1_simulated_short.xes_25_.csv"
]

summarize_results(PREFIX_PATH_NO_CONF, dims_25, avg_costs_25,
                  *warm_start_runs_new_no_conf)

################################################# with confidence #################################################

print("#"*30+" with confidence "+"#"*30)
print("")

PREFIX_PATH_CONF = "output files/conf, compl, confi/"

warm_start_runs_old_conf = [
    "M1_simulated_M1.xes_14_.csv",
    "M1_simulated_M1_warm_2.xes_14_.csv",
    "M1_simulated_M1_warm_5.xes_14_.csv",
    "M1_simulated_M1_simulated_long.xes_14_.csv",
    "M1_simulated_M1_simulated_short.xes_14_.csv"
]

summarize_results(PREFIX_PATH_NO_CONF, dims_14, avg_costs_14,
                  *warm_start_runs_old_no_conf)

warm_start_runs_new_conf = [
    "M1_simulated_M1.xes_25_.csv",
    "M1_simulated_M1_warm_2.xes_25_.csv",
    "M1_simulated_M1_warm_5.xes_25_.csv",
    "M1_simulated_M1_simulated_long.xes_25_.csv",
    "M1_simulated_M1_simulated_short.xes_25_.csv"
]

summarize_results(PREFIX_PATH_NO_CONF, dims_25, avg_costs_25,
                  *warm_start_runs_new_no_conf)

# ################################################# compare to confidence #################################################

print("#"*30+" comparison confidence "+"#"*30)
print("")

print("")
print("-"*30+" long "+"-"*30)
print("")

prefix_paths = [PREFIX_PATH_CONF, PREFIX_PATH_NO_CONF]
paths = list(
    zip(prefix_paths, [warm_start_runs_new_conf[3], warm_start_runs_new_no_conf[3]]))

results = compare_to_confidence(paths)

if results[0] == 0 or results[1] == 0:
    print("increase in avg cost with {}% for {} ".format(0, "long trace log"))
else:
    print("increase in avg cost with {}% for {} ".format(
        round(((results[0]/results[1])-1)*100, 2), "long trace log"))

print("")
print("-"*30+" short "+"-"*30)
print("")

prefix_paths = [PREFIX_PATH_CONF, PREFIX_PATH_NO_CONF]
paths = list(
    zip(prefix_paths, [warm_start_runs_new_conf[4], warm_start_runs_new_no_conf[4]]))

results = compare_to_confidence(paths)

if results[0] == 0 or results[1] == 0:
    print("increase in avg cost with {}% for {} ".format(0, "short trace log"))
else:
    print("increase in avg cost with {}% for {} ".format(
        round(((results[0]/results[1])-1)*100, 2), "short trace log"))

print("")
