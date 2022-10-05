import csv
from statistics import mean

# csv_files = ["old_OCC_false.csv"]

# for file_name in csv_files:
#     with open("conf, compl, confi/"+file_name, mode='r') as file:
#         csvFile = csv.reader(file)
#         l = list()
#         for lines in csvFile:
#             l.append(lines[1])
#         l = [int(i) for i in l[1:-1]]
#         print("Avg {} for {} ".format(round(mean(l),2), file_name))

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

# for file_name in new_csv_files:
#      with open("conf, compl, confi/"+file_name, mode='r') as file:
#         csvFile = csv.reader(file)
#         l = list()
#         for lines in csvFile:
#             l.append(lines[5])
#         l = [float(i) for i in l[1:-1]]
#         print("Avg {} for {} ".format(round(mean(l),2), file_name))

warm_start_runs_old = [
    "M1_simulated_M1.xes_14_.csv",
    "M1_simulated_M1_warm_2.xes_14_.csv",
    "M1_simulated_M1_warm_5.xes_14_.csv"
]

for file_name in warm_start_runs_old:
    with open("conf, compl, confi/"+file_name, mode='r') as file:
        csvFile = csv.reader(file)
        l = list()
        for lines in csvFile:
            l.append(lines[1]) # 1 = cost, 2 = execution time
        l = [float(i) for i in l[1:-1]]
        print("Avg {} for {} ".format(round(mean(l), 2), file_name))

warm_start_runs_new = [
    "M1_simulated_M1.xes_25_.csv",
    "M1_simulated_M1_warm_2.xes_25_.csv",
    "M1_simulated_M1_warm_5.xes_25_.csv"
]

print("")

for file_name in warm_start_runs_new:
    with open("conf, compl, confi/"+file_name, mode='r') as file:
        csvFile = csv.reader(file)
        l = list()
        for lines in csvFile:
            l.append(lines[4]) # 4 = cost, 5 = execution time
        l = [float(i) for i in l[1:-1]]
        print("Avg {} for {} ".format(round(mean(l), 2), file_name))


"""
Run with bounded cost:

Avg 5.29 for M1_simulated_M1.xes_14_.csv
Avg 6.75 for M1_simulated_M1_warm_2.xes_14_.csv
Avg 7.03 for M1_simulated_M1_warm_5.xes_14_.csv

Avg 5.29 for M1_simulated_M1.xes_25_.csv
Avg 6.8 for M1_simulated_M1_warm_2.xes_25_.csv
Avg 7.52 for M1_simulated_M1_warm_5.xes_25_.csv

------------------------------------------------

Run with old minimisation:

Avg 5.29 for M1_simulated_M1.xes_14_.csv
Avg 6.75 for M1_simulated_M1_warm_2.xes_14_.csv
Avg 7.03 for M1_simulated_M1_warm_5.xes_14_.csv

Avg 5.29 for M1_simulated_M1.xes_25_.csv
Avg 6.8 for M1_simulated_M1_warm_2.xes_25_.csv
Avg 7.52 for M1_simulated_M1_warm_5.xes_25_.csv

------------------------------------------------

Run with old minimisation and warm-start all states

completeness only

Avg 4.03 for M1_simulated_M1.xes_25_.csv
Avg 4.81 for M1_simulated_M1_warm_2.xes_25_.csv
Avg 5.04 for M1_simulated_M1_warm_5.xes_25_.csv

completeness + alignment length

Avg 4.41 for M1_simulated_M1.xes_25_.csv
Avg 5.38 for M1_simulated_M1_warm_2.xes_25_.csv
Avg 5.7 for M1_simulated_M1_warm_5.xes_25_.csv

"""