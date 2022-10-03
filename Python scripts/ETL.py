import csv
from statistics import mean

csv_files = ["old_OCC_false.csv"]

for file_name in csv_files:
    with open("conf, compl, confi/"+file_name, mode='r') as file:
        csvFile = csv.reader(file)
        l = list()
        for lines in csvFile:
            l.append(lines[1])
        l = [int(i) for i in l[1:-1]]
        print("Avg {} for {} ".format(round(mean(l),2), file_name))

new_csv_files = ["old_OCC_warm-start_no_align_false.csv",
"old_OCC_warm-start_all_states.csv",
"conf_cost_no_warm-start_false.csv",
"conf_cost_warm-start_all_states_if_in_ws_false.csv",
"conf_cost_warm-start_no_align_if_in_ws_false.csv",
"conf_cost_warm-start_all_states_false.csv",
"bounded_cost_no_warm-start_standard_false.csv",
"bounded_cost_warm-start_no_align_standard_false.csv",
"bounded_cost_warm-start_no_align_false.csv",
"bounded_cost_no_warm-start_false.csv",
"bounded_cost_warm-start_all_states_if_in_ws_standard_false.csv",
"bounded_cost_warm-start_all_states_standard_false.csv",
"test.csv"]

for file_name in new_csv_files:
     with open("conf, compl, confi/"+file_name, mode='r') as file:
        csvFile = csv.reader(file)
        l = list()
        for lines in csvFile:
            l.append(lines[4])
        l = [float(i) for i in l[1:-1]]
        print("Avg {} for {} ".format(round(mean(l),2), file_name))