import csv
from statistics import mean
from sys import prefix


def summarize_results(prefix_path, dim, dim_type, *files):
    for file_name in files:
        with open(prefix_path+file_name, mode='r') as file:
            csvFile = csv.reader(file)
            l = list()
            for lines in csvFile:
                l.append(lines[dim_type])
            l = [float(i) for i in l[1:-1]]
            print("Avg {} {} for {} ".format(
                dim[dim_type], round(mean(l), 2), file_name))
    print("")


def compare_to_confidence(prefix_paths, files):
    results = []

    for i, file_name in enumerate(files):
        with open(prefix_paths[i]+file_name, mode='r') as file:
            csvFile = csv.reader(file)
            l = list()
            for lines in csvFile:
                l.append(lines[4])
            l = [float(i) for i in l[1:-1]]
            results.append(mean(l))

    return results

################################################# without confidence #################################################


PREFIX_PATH_NO_CONF = "conf, compl/"

# csv_files = ["old_OCC_false.csv"]

# new_csv_files = ["old_OCC_warm-start_no_align_false.csv",
# "old_OCC_warm-start_all_states.csv",
# "conf_cost_no_warm-start_false.csv",
# "conf_cost_warm-start_all_states_if_in_ws_false.csv",
# "conf_cost_warm-start_no_align_if_in_ws_false.csv",
# "conf_cost_warm-start_all_states_false.csv",
# "bounded_cost_no_warm-start_standard_false.csv",
# "bounded_cost_warm-start_no_align_standard_false.csv",
# "bounded_cost_warm-start_no_align_false.csv",
# "bounded_cost_no_warm-start_false.csv",
# "bounded_cost_warm-start_all_states_if_in_ws_standard_false.csv",
# "bounded_cost_warm-start_all_states_standard_false.csv",
# "test.csv"]

warm_start_runs_old_no_conf = [
    "M1_simulated_M1.xes_14_.csv",
    "M1_simulated_M1_warm_2.xes_14_.csv",
    "M1_simulated_M1_warm_5.xes_14_.csv",
    "M1_simulated_M1_sim_long.xes_14_.csv",
    "M1_simulated_M1_sim_short.xes_14_.csv"
]

dim = {1: "cost", 2: "time"}
dim_type = 1

summarize_results(PREFIX_PATH_NO_CONF, dim, dim_type,
                  *warm_start_runs_old_no_conf)

warm_start_runs_new_no_conf = [
    "M1_simulated_M1.xes_25_.csv",
    "M1_simulated_M1_warm_2.xes_25_.csv",
    "M1_simulated_M1_warm_5.xes_25_.csv",
    "M1_simulated_M1_sim_long.xes_25_.csv",
    "M1_simulated_M1_sim_short.xes_25_.csv"
]

dim = {4: "cost", 5: "time"}
dim_type = 4

summarize_results(PREFIX_PATH_NO_CONF, dim, dim_type,
                  *warm_start_runs_new_no_conf)

################################################# with confidence #################################################

PREFIX_PATH_CONF = "conf, compl, confi/"

warm_start_runs_old_conf = [
    "M1_simulated_M1.xes_14_.csv",
    "M1_simulated_M1_warm_2.xes_14_.csv",
    "M1_simulated_M1_warm_5.xes_14_.csv",
    "M1_simulated_M1_sim_long.xes_14_.csv",
    "M1_simulated_M1_sim_short.xes_14_.csv"
]

dim = {1: "cost", 2: "time"}
dim_type = 1

summarize_results(PREFIX_PATH_CONF, dim, dim_type, *warm_start_runs_old_conf)

warm_start_runs_new_conf = [
    "M1_simulated_M1.xes_25_.csv",
    "M1_simulated_M1_warm_2.xes_25_.csv",
    "M1_simulated_M1_warm_5.xes_25_.csv",
    "M1_simulated_M1_sim_long.xes_25_.csv",
    "M1_simulated_M1_sim_short.xes_25_.csv"
]

dim = {4: "cost", 5: "time"}
dim_type = 4

summarize_results(PREFIX_PATH_CONF, dim, dim_type, *warm_start_runs_new_conf)

################################################# compare to confidence #################################################

prefix_paths = [PREFIX_PATH_CONF, PREFIX_PATH_NO_CONF]

results = compare_to_confidence(
    prefix_paths, [warm_start_runs_new_conf[3], warm_start_runs_new_no_conf[3]])
print("increase by a factor of {} for {} ".format(
    round(results[0]/results[1], 2), "long trace log"))

results = compare_to_confidence(
    prefix_paths, [warm_start_runs_new_conf[4], warm_start_runs_new_no_conf[4]])
print("increase by {} for {} ".format(
    round(results[0]/results[1], 2), "short trace log"))