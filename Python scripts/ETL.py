import csv
from statistics import mean

# csv_files = ["old_OCC_false.csv"]

# dim = {1:"cost", 2:"time"}
# dim_type = 2

# for file_name in csv_files:
#     with open("conf, compl, confi/"+file_name, mode='r') as file:
#         csvFile = csv.reader(file)
#         l = list()
#         for lines in csvFile:
#             l.append(lines[dim_type]) # 1 = cost, 2 = execution time
#         l = [float(i) for i in l[1:-1]]
#         print("Avg {} {} for {} ".format(dim[dim_type],round(mean(l), 2), file_name))

# print("")

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

# dim = {4:"cost", 5:"time"}
# dim_type = 5

# for file_name in new_csv_files:
#      with open("conf, compl, confi/"+file_name, mode='r') as file:
#         csvFile = csv.reader(file)
#         l = list()
#         for lines in csvFile:
#             l.append(lines[dim_type]) # 4 = cost, 5 = execution time
#         l = [float(i) for i in l[1:-1]]
#         print("Avg {} {} for {} ".format(dim[dim_type],round(mean(l), 2), file_name))

# print("")

warm_start_runs_old = [
    "M1_simulated_M1.xes_14_.csv",
    "M1_simulated_M1_warm_2.xes_14_.csv",
    "M1_simulated_M1_warm_5.xes_14_.csv"
]

dim = {1:"cost", 2:"time"}
dim_type = 2

for file_name in warm_start_runs_old:
    with open("conf, compl, confi/"+file_name, mode='r') as file:
        csvFile = csv.reader(file)
        l = list()
        for lines in csvFile:
            l.append(lines[dim_type]) # 1 = cost, 2 = execution time
        l = [float(i) for i in l[1:-1]]
        print("Avg {} {} for {} ".format(dim[dim_type],round(mean(l), 2), file_name))

print("")

warm_start_runs_new = [
    "M1_simulated_M1.xes_25_.csv",
    "M1_simulated_M1_warm_2.xes_25_.csv",
    "M1_simulated_M1_warm_5.xes_25_.csv"
]

dim = {4:"cost", 5:"time"}
dim_type = 5

for file_name in warm_start_runs_new:
    with open("conf, compl, confi/"+file_name, mode='r') as file:
        csvFile = csv.reader(file)
        l = list()
        for lines in csvFile:
            l.append(lines[dim_type]) # 4 = cost, 5 = execution time
        l = [float(i) for i in l[1:-1]]
        print("Avg {} {} for {} ".format(dim[dim_type],round(mean(l), 2), file_name))


"""
Run with bounded cost:

Avg cost 5.29 for M1_simulated_M1.xes_14_.csv
Avg cost 6.75 for M1_simulated_M1_warm_2.xes_14_.csv
Avg cost 7.03 for M1_simulated_M1_warm_5.xes_14_.csv

Avg cost 5.29 for M1_simulated_M1.xes_25_.csv
Avg cost 6.8 for M1_simulated_M1_warm_2.xes_25_.csv
Avg cost 7.52 for M1_simulated_M1_warm_5.xes_25_.csv½

--------------------------------------------------------

Run with old minimisation:

Avg cost 5.29 for M1_simulated_M1.xes_14_.csv
Avg cost 6.75 for M1_simulated_M1_warm_2.xes_14_.csv
Avg cost 7.03 for M1_simulated_M1_warm_5.xes_14_.csv

Avg cost 5.29 for M1_simulated_M1.xes_25_.csv
Avg cost 6.8 for M1_simulated_M1_warm_2.xes_25_.csv
Avg cost 7.52 for M1_simulated_M1_warm_5.xes_25_.csv

--------------------------------------------------------

Run with old minimisation and warm-start all states

completeness only

Avg cost 4.03 for M1_simulated_M1.xes_25_.csv
Avg cost 4.81 for M1_simulated_M1_warm_2.xes_25_.csv
Avg cost 5.04 for M1_simulated_M1_warm_5.xes_25_.csv

completeness + alignment length

Avg cost 4.41 for M1_simulated_M1.xes_25_.csv
Avg cost 5.38 for M1_simulated_M1_warm_2.xes_25_.csv
Avg cost 5.7 for M1_simulated_M1_warm_5.xes_25_.csv

"""