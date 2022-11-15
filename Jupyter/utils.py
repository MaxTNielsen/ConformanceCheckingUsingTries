import sys
import pandas as pd
import numpy as np
import stats
import os
import re
from statistics import mean

LOGS = ["BPI_2017","BPI_2012","M1","M2","M4","M8","M9"]
LOG_TYPES = ['completeness20', 'completeness50', 'sim']

def get_log_type(filename:str) -> str:
    logs_pattern = '|'.join([str(log_type) for log_type in ['completeness20', 'completeness50']])
    res = re.search(logs_pattern, filename)
    return res.group() if res is not None else 'sim'

def get_occ_dicts(dir_path: str, isC_3PO:bool=False) -> dict:
    """ params:
                dir_path : directory path to occ results
        returns:
                nested dict with dataframes containing the average for a given log over all runs
        keys:
                first level : M1, .. , BPI_2017 (all logs)
                second level : sim, completeness20, completeness 50
    """
    log_dfs = {log_n:{log_t:pd.DataFrame.__class__ for log_t in LOG_TYPES} for log_n in LOGS}
    temp_df = {log_n:{log_t:{} for log_t in LOG_TYPES} for log_n in LOGS}
    
    for dir_ in os.listdir(dir_path):
        run_dir = os.path.join(dir_path, dir_)
        for dir__ in os.listdir(run_dir):
            log_dir = os.path.join(run_dir, dir__)
            for f_ in os.listdir(log_dir):
                f_path = os.path.join(log_dir, f_)
                f_key = get_log_type(f_)
                t = pd.read_csv(f_path)
                if 'conf' not in temp_df[dir__][f_key] or 'exe' not in temp_df[dir__][f_key]:
                    temp_df[dir__][f_key]['conf'] = []
                    temp_df[dir__][f_key]['exe'] = []              
                temp_df[dir__][f_key]['conf'].append(t[t.columns[1]].values.tolist())
                temp_df[dir__][f_key]['exe'].append(t[t.columns[-1]].values.tolist())
                if isC_3PO:
                    if 'compl' not in temp_df[dir__][f_key]:
                        temp_df[dir__][f_key]['compl'] = []
                    temp_df[dir__][f_key]['compl'].append(t[t.columns[2]].values.tolist())
                    del t[t.columns[2]]
                    del t[t.columns[2]]
                    t.rename(columns = {' total cost':'total cost'}, inplace = True)
                del t[t.columns[-1]]
                del t[t.columns[1]]
                log_dfs[dir__][f_key] = t
    
    for log in LOGS:
        for log_type in LOG_TYPES:
            exe_times = (mean([int(ele) for ele in tp]) for tp in list(zip(*temp_df[log][log_type]['exe'])))
            conf_cost = (mean([int(ele) for ele in tp]) for tp in list(zip(*temp_df[log][log_type]['conf'])))
            if isC_3PO:
                compl_cost = (mean([int(ele) for ele in tp]) for tp in list(zip(*temp_df[log][log_type]['compl'])))
                log_dfs[log][log_type]['Completeness cost'] = np.array(compl_cost)
            log_dfs[log][log_type]['Conformance cost'] = np.array(conf_cost)
            log_dfs[log][log_type]['ExecutionTime'] = np.array(exe_times)
            
    return log_dfs


def get_hmmconf_dict(dir_path: str):
    """ params:
                dir_path : directory path to hmmconf results
        returns:
                nested dict with dataframes containing the results for a given log
        keys:
                first level : M1, .. , BPI_2017 (all logs)
                second level : sim, completeness20, completeness 50
    """
    log_dfs = {log_n:{log_t:pd.DataFrame.__class__ for log_t in LOG_TYPES} for log_n in LOGS}
    temp_df = {log_n:{log_t:list() for log_t in LOG_TYPES} for log_n in LOGS}
    
    for dir_ in os.listdir(dir_path):
        log_dir = os.path.join(dir_path, dir_)
        for f_ in os.listdir(log_dir):
            f_path = os.path.join(log_dir, f_)
            f_key = get_log_type(f_)
            t = pd.read_csv(f_path)
            temp_df[dir_][f_key].append(t)
    

    for log_name in LOGS:
        for log_type in LOG_TYPES:
            df_concat = pd.concat((temp_df[log_name][log_type]))
            by_row_index = df_concat.groupby(df_concat.index)
            log_dfs[log_name][log_type] = by_row_index.mean()

    return log_dfs


"""check wether conformance and completeness should be averaged"""

# def verify_conf_compl(occ_dict:dict, isC_3PO:bool=False) -> None: 
#     exe_times = []
#     conf_cost = []
#     compl_cost = []
#     for run_dir in ['run_1','run_2','run_3','run_4','run_5']:
#         exe_times.append(occ_dict[run_dir]['M1']['sim'][occ_dict[run_dir]['M1']['sim'].columns[-1]].values)
#         conf_cost.append(occ_dict[run_dir]['M1']['sim'][occ_dict[run_dir]['M1']['sim'].columns[1]].values)
#         if isC_3PO:
#             compl_cost.append(occ_dict[run_dir]['M1']['sim'][occ_dict[run_dir]['M1']['sim'].columns[2]].values)

#     exe_times = (mean([int(ele) for ele in tp]) for tp in list(zip(*exe_times)))
#     conf_cost = (mean([int(ele) for ele in tp]) for tp in list(zip(*conf_cost)))
#     if isC_3PO:
#         compl_cost = (mean([int(ele) for ele in tp]) for tp in list(zip(*compl_cost)))

#     msg = "mean conf cost {} for log {}"
#     print(msg.format(mean(conf_cost), 'M1_sim_5_runs'))
#     print(msg.format(mean(occ_dict['run_1']['M1']['sim'][occ_dict['run_1']['M1']['sim'].columns[1]].values), 'M1_sim'))

#     if isC_3PO:
#         msg = "mean compl cost {} for log {}"
#         print(msg.format(mean(compl_cost), 'M1_sim_5_runs'))
#         print(msg.format(mean(occ_dict['run_1']['M1']['sim'][occ_dict['run_1']['M1']['sim'].columns[2]].values), 'M1_sim'))