import pm4py
import os
import numpy as np
log = pm4py.read_xes(os.path.join('..','input','experiment-data','BPI_2012','BPI_2012_1k_sample.xes'))
all_case_durations = pm4py.get_all_case_durations(log)
print((np.mean(all_case_durations) / 60) / 60)