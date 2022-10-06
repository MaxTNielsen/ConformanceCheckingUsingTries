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

#print("")

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

#print("")

warm_start_runs_old = [
    "M1_simulated_M1.xes_14_.csv",
    "M1_simulated_M1_warm_2.xes_14_.csv",
    "M1_simulated_M1_warm_5.xes_14_.csv"
]

dim = {1:"cost", 2:"time"}
dim_type = 1

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
dim_type = 4

for file_name in warm_start_runs_new:
    with open("conf, compl, confi/"+file_name, mode='r') as file:
        csvFile = csv.reader(file)
        l = list()
        for lines in csvFile:
            l.append(lines[dim_type]) # 4 = cost, 5 = execution time
        l = [float(i) for i in l[1:-1]]
        print("Avg {} {} for {} ".format(dim[dim_type],round(mean(l), 2), file_name))


"""
COST:

Old OCC
Avg cost 5.29 for M1_simulated_M1.xes_14_.csv 
Avg cost 6.75 for M1_simulated_M1_warm_2.xes_14_.csv
Avg cost 7.03 for M1_simulated_M1_warm_5.xes_14_.csv

New OCC
Avg cost 5.29 for M1_simulated_M1.xes_25_.csv
Avg cost 6.75 for M1_simulated_M1_warm_2.xes_25_.csv
Avg cost 7.03 for M1_simulated_M1_warm_5.xes_25_.csv

TIME:

Old OCC
Avg time 0.83 for M1_simulated_M1.xes_14_.csv 
Avg time 0.32 for M1_simulated_M1_warm_2.xes_14_.csv
Avg time 0.31 for M1_simulated_M1_warm_5.xes_14_.csv

New OCC
Avg time 1.1 for M1_simulated_M1.xes_25_.csv
Avg time 0.95 for M1_simulated_M1_warm_2.xes_25_.csv
Avg time 0.31 for M1_simulated_M1_warm_5.xes_25_.csv

"""