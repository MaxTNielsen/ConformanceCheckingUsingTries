import csv
from statistics import mean
import copy
import os


dims_14 = {'conf': 1, 'time': 2}
avg_costs_14 = {'conf': [], 'time': []}

dims_25 = {'conf': 1, 'compl': 2, 'confi': 3, 'total': 4, 'time': 5}
avg_costs_25 = {'conf': [], 'compl': [], 'confi': [], 'total': [], 'time': []}


def summarize_results(prefix_path, dims, avg_costs, printFunction):

    for filename in os.listdir(prefix_path):
        f = os.path.join(prefix_path, filename)
        if os.path.isfile(f):
            with open(f, mode='r') as file:
                csvFile = csv.reader(file)
                avg_costs_copy = copy.deepcopy(avg_costs)
                for i, lines in enumerate(csvFile):
                    if i == 0:
                        continue
                    for key, val in dims.items():
                        avg_costs_copy[key].append(float(lines[val]))
            file.close()

        for key, val in avg_costs_copy.items():
            avg_costs_copy[key] = round(mean(val), 2)

        printFunction(avg_costs=avg_costs_copy, file_name=filename)

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


if __name__ == "__main__":
    ################################################# without confidence #################################################
    PREFIX_PATH_OLD_OCC = "output files/old_OCC_runs/"
    PREFIX_PATH_NO_CONF = "output files/conf, compl/"

    print("")
    print("#"*30+" without confidence "+"#"*30)
    print("")

    summarize_results(PREFIX_PATH_OLD_OCC, dims_14,
                      avg_costs_14, print_results_14)

    summarize_results(PREFIX_PATH_NO_CONF, dims_25,
                      avg_costs_25, print_results_25)

    ################################################# with confidence #################################################

    PREFIX_PATH_CONF = "output files/conf, compl, confi/"

    print("#"*30+" with confidence "+"#"*30)
    print("")

    summarize_results(PREFIX_PATH_CONF, dims_25,
                      avg_costs_25, print_results_25)

    # ################################################# with weighted confidence #################################################

    PREFIX_PATH_WEIGHTED_CONF = "output files/conf, compl, weighted confi/"

    print("#"*30+" with weighted confidence "+"#"*30)
    print("")

    summarize_results(PREFIX_PATH_WEIGHTED_CONF, dims_25,
                      avg_costs_25, print_results_25)

    # ################################################# with weighted scaled confidence #################################################

    PREFIX_PATH_WEIGHTED_SCALED_CONF = "output files/conf, compl, weighted scaled confi/"

    print("#"*30+" with weighted scaled confidence "+"#"*30)
    print("")

    summarize_results(PREFIX_PATH_WEIGHTED_SCALED_CONF,
                      dims_25, avg_costs_25, print_results_25)

    # ################################################# only pref prob #################################################

    PREFIX_PATH_ONLY_PREF_PROB = "output files/conf, compl, only pref prob/"

    print("#"*30+" only pref prob confidence "+"#"*30)
    print("")

    summarize_results(PREFIX_PATH_ONLY_PREF_PROB, dims_25,
                      avg_costs_25, print_results_25)


    # ################################################# standard #################################################

    PREFIX_PATH_ONLY_PREF_PROB = "output files/standard/"

    print("#"*30+" only pref prob confidence "+"#"*30)
    print("")

    summarize_results(PREFIX_PATH_ONLY_PREF_PROB, dims_25,
                      avg_costs_25, print_results_25)

    # ################################################# compare to confidence #################################################

    # prefix_paths = [PREFIX_PATH_CONF, PREFIX_PATH_NO_CONF]

    # print("#"*30+" comparison confidence "+"#"*30)
    # print("")

    # print("")
    # print("-"*30+" long "+"-"*30)
    # print("")

    # paths = list(
    #     zip(prefix_paths, [warm_start_runs_new[3], warm_start_runs_new[3]]))

    # results = compare_to_confidence(paths)

    # if results[0] == 0 or results[1] == 0:
    #     print("increase in avg cost with {}% for {} ".format(0, "long trace log"))
    # else:
    #     print("increase in avg cost with {}% for {} ".format(
    #         round(((results[0]/results[1])-1)*100, 2), "long trace log"))

    # print("")
    # print("-"*30+" short "+"-"*30)
    # print("")

    # prefix_paths = [PREFIX_PATH_CONF, PREFIX_PATH_NO_CONF]
    # paths = list(
    #     zip(prefix_paths, [warm_start_runs_new[4], warm_start_runs_new[4]]))

    # results = compare_to_confidence(paths)

    # if results[0] == 0 or results[1] == 0:
    #     print("increase in avg cost with {}% for {} ".format(0, "short trace log"))
    # else:
    #     print("increase in avg cost with {}% for {} ".format(
    #         round(((results[0]/results[1])-1)*100, 2), "short trace log"))
    # print("")
    # Avg conf 5.59 - Avg compl 0.5 - Avg confi 0.0 - Avg total 5.63 - Avg time 1.13 - for M1_simulated_M1.xes_25_.csv
    # Avg conf 5.08 - Avg compl 2.29 - Avg confi 0.0 - Avg total 6.32 - Avg time 0.89 - for M1_simulated_M1_warm_2.xes_25_.csv
    # Avg conf 5.41 - Avg compl 2.56 - Avg confi 0.0 - Avg total 6.15 - Avg time 0.67 - for M1_simulated_M1_warm_5.xes_25_.csv
    # Avg conf 9.23 - Avg compl 0.39 - Avg confi 0.0 - Avg total 9.26 - Avg time 2.74 - for M1_simulated_M1_simulated_long.xes_25_.csv
    # Avg conf 5.73 - Avg compl 0.89 - Avg confi 0.0 - Avg total 5.76 - Avg time 0.62 - for M1_simulated_M1_simulated_short.xes_25_.csv