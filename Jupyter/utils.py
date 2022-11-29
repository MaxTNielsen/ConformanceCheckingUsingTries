import pandas as pd
import numpy as np
import os
import re
from statistics import mean
import xml.etree.ElementTree as ET

LOG_TYPES = ['completeness20', 'completeness50', 'sim']

def get_log_type(filename:str) -> str:
    logs_pattern = '|'.join([str(log_type) for log_type in ['completeness20', 'completeness50']])
    res = re.search(logs_pattern, filename)
    return res.group() if res is not None else 'sim'

def get_occ_dfs(dir_path:str, logs:str, isC_3PO:bool=False) -> pd.DataFrame.__class__:
    """ params:
                dir_path : directory path to occ results
        returns:
                concatenated df with results from all logs
    """
    log_dfs = {log_n:{log_t:pd.DataFrame.__class__ for log_t in LOG_TYPES} for log_n in logs}
    temp_df = {log_n:{log_t:{} for log_t in LOG_TYPES} for log_n in logs}
    
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
                    if 'compl' not in temp_df[dir__][f_key] or 'confi' not in temp_df[dir__][f_key]:
                        temp_df[dir__][f_key]['compl'] = []
                        temp_df[dir__][f_key]['confi'] = []
                    temp_df[dir__][f_key]['compl'].append(t[t.columns[2]].values.tolist())
                    temp_df[dir__][f_key]['confi'].append(t[t.columns[3]].values.tolist())
                    del t[t.columns[2]] ## delete completeness
                    del t[t.columns[2]] ## delete confidence
                    del t[t.columns[2]] ## delete total cost - not needed
                del t[t.columns[-1]] ## delete execution time
                del t[t.columns[1]] ## delete conformance
                log_dfs[dir__][f_key] = t
    
    for log in logs:
        for log_type in LOG_TYPES:
            exe_times = (mean([int(ele) for ele in tp]) for tp in list(zip(*temp_df[log][log_type]['exe'])))
            conf_cost = (mean([int(ele) for ele in tp]) for tp in list(zip(*temp_df[log][log_type]['conf'])))
            log_dfs[log][log_type]['Conformance cost'] = np.array(conf_cost)
            if isC_3PO:
                compl_cost = (mean([int(ele) for ele in tp]) for tp in list(zip(*temp_df[log][log_type]['compl'])))
                confi_cost = (mean([int(ele) for ele in tp]) for tp in list(zip(*temp_df[log][log_type]['confi'])))
                log_dfs[log][log_type]['Completeness cost'] = np.array(compl_cost)
                log_dfs[log][log_type]['Confidence cost'] = np.array(confi_cost)
            log_dfs[log][log_type]['ExecutionTime'] = np.array(exe_times)
            
    df_list = []
    for log_n, log_types in log_dfs.items():
        for log_t, results in log_types.items():
            results['model'] = len(results) * [log_n]
            results['log_type'] = len(results) * [log_t]
            df_list.append(results)

    log_dfs = pd.concat(df_list, ignore_index=True)

    log_dfs['Conformance cost'] = log_dfs['Conformance cost'].astype(float)
    log_dfs['ExecutionTime'] = log_dfs['ExecutionTime'].astype(float)

    if isC_3PO:
        log_dfs['Completeness cost'] = log_dfs['Completeness cost'].astype(float)
        log_dfs['Confidence cost'] = log_dfs['Confidence cost'].astype(float)

    return log_dfs


def get_hmmconf_df(dir_path: str, logs:str) -> pd.DataFrame.__class__:
    """ params:
                dir_path : directory path to occ results
        returns:
                concatenated df with results from all logs
    """
    log_dfs = {log_n:{log_t:pd.DataFrame.__class__ for log_t in LOG_TYPES} for log_n in logs}
    temp_df = {log_n:{log_t:list() for log_t in LOG_TYPES} for log_n in logs}
    
    for dir_ in os.listdir(dir_path):
        log_dir = os.path.join(dir_path, dir_)
        for f_ in os.listdir(log_dir):
            f_path = os.path.join(log_dir, f_)
            f_key = get_log_type(f_)
            t = pd.read_csv(f_path)
            temp_df[dir_][f_key].append(t)
    

    for log_name in logs:
        for log_type in LOG_TYPES:
            df_concat = pd.concat((temp_df[log_name][log_type]))
            by_row_index = df_concat.groupby(df_concat.index)
            log_dfs[log_name][log_type] = by_row_index.mean()
            log_dfs[log_name][log_type].rename(columns={
                'execution time': 'ExecutionTime',
                'caseid': 'TraceId'
            }, inplace=True)


    df_list = []
    for log_n, log_types in log_dfs.items():
        for log_t, results in log_types.items():
            results['model'] = len(results) * [log_n]
            results['log_type'] = len(results) * [log_t]
            df_list.append(results)

    log_dfs = pd.concat(df_list, ignore_index=True)

    return log_dfs


def get_trace_map(filename):
    """ 
        build a map to index traces in numerical order in log for BPI_XXXX logs 
        necessary due to events processed out of order in BP algorithm
    """
    trace_to_idx = {}
    tree = ET.parse(filename)
    root = tree.getroot()
    for i,trace in enumerate(root.findall('{http://www.xes-standard.org/}trace')):
      trace_ = trace.findall("./*[@key='concept:name']")
      trace_id = trace_[0].attrib['value'].split('_')[-1]
      trace_to_idx[trace_id] = i
    return trace_to_idx


def preprocess_bp_results(data_frame:pd.DataFrame.__class__):
    """ retrieve values from fields in .csv file """
    data_frame['caseId'] = data_frame['caseId'].apply(lambda x: x.split(' ')[-1]).apply(lambda x: x.split('_')[-1])
    data_frame['activityId'] = data_frame['activityId'].apply(lambda x: x.split(' ')[-1])
    data_frame['conformance'] = data_frame['conformance'].apply(lambda x: x.split(' ')[-1])
    data_frame['completeness'] = data_frame['completeness'].apply(lambda x: x.split(' ')[-1])
    data_frame['confidence'] = data_frame['confidence'].apply(lambda x: x.split(' ')[-1])
    data_frame['processing-time'] = data_frame['processing-time'].apply(lambda x: x.split(' ')[-1])

    """ rescale processing time from nanoseconds to miliseconds """
    data_frame['processing-time'] = data_frame['processing-time'].astype(float).apply(lambda x: x/(10.0**6.0))
    #data_frame['processing-time'] = data_frame['processing-time'].apply(lambda x: x/1.0*10.0**6.0)

    data_frame.rename(columns={
        'processing-time':  'ExecutionTime',
        'activityId': 'activityId',
        'caseId': 'TraceId',
        }, inplace=True)

def get_bp_df(dir_path: str, logs:str) -> pd.DataFrame.__class__:
    """ params:
                dir_path : directory path to occ results
        returns:
                concatenated df with results from all logs
    """
    log_dfs = {log_n:{log_t:pd.DataFrame.__class__ for log_t in LOG_TYPES} for log_n in logs}
    temp_df = {log_n:{log_t:{} for log_t in LOG_TYPES} for log_n in logs}
    bp_log_paths = dict()
    traces_to_idx = dict()

    data_set_dir = os.path.join('..','input','experiment-data')
    bp_log_paths['BPI_2012'] = os.path.join(data_set_dir, 'BPI_2012','BPI_2012_1k_sample.xes')
    bp_log_paths['BPI_2017'] = os.path.join(data_set_dir, 'BPI_2017','BPI_2017_1k_sample.xes')


    for dir_ in os.listdir(dir_path):
        run_dir = os.path.join(dir_path, dir_)
        for dir__ in os.listdir(run_dir):
            log_dir = os.path.join(run_dir, dir__)
            if dir__ in ['BPI_2012', 'BPI_2017']:
                traces_to_idx[dir__] = get_trace_map(bp_log_paths[dir__])
            for f_ in os.listdir(log_dir):
                f_path = os.path.join(log_dir, f_)
                f_key = get_log_type(f_)
                t = pd.read_csv(f_path, ';')
                preprocess_bp_results(t)
                if 'conf' not in temp_df[dir__][f_key] or 'exe' not in temp_df[dir__][f_key] or 'compl' not in temp_df[dir__][f_key] or 'confi' not in temp_df[dir__][f_key]:
                    temp_df[dir__][f_key]['conf'] = []
                    temp_df[dir__][f_key]['compl'] = []
                    temp_df[dir__][f_key]['confi'] = []
                    temp_df[dir__][f_key]['exe'] = []
                temp_df[dir__][f_key]['conf'].append(t[t.columns[2]].values.tolist())
                temp_df[dir__][f_key]['compl'].append(t[t.columns[3]].values.tolist())
                temp_df[dir__][f_key]['confi'].append(t[t.columns[4]].values.tolist())
                temp_df[dir__][f_key]['exe'].append(t[t.columns[-1]].values.tolist())
                del t[t.columns[2]]
                del t[t.columns[2]]
                del t[t.columns[2]]
                del t[t.columns[2]]
                log_dfs[dir__][f_key] = t
    
    for log in logs:
        for log_type in LOG_TYPES:
            conf_cost = (mean([float(ele) for ele in tp]) for tp in list(zip(*temp_df[log][log_type]['conf'])))
            compl_cost = (mean([float(ele) for ele in tp]) for tp in list(zip(*temp_df[log][log_type]['compl'])))
            confi_cost = (mean([float(ele) for ele in tp]) for tp in list(zip(*temp_df[log][log_type]['confi'])))
            exe_times = (mean([float(ele) for ele in tp]) for tp in list(zip(*temp_df[log][log_type]['exe'])))
            log_dfs[log][log_type]['conformance'] = np.array(conf_cost)
            log_dfs[log][log_type]['completeness'] = np.array(compl_cost)
            log_dfs[log][log_type]['confidence'] = np.array(confi_cost)
            log_dfs[log][log_type]['ExecutionTime'] = np.array(exe_times)
            if log in ['BPI_2012', 'BPI_2017']:
                 log_dfs[log][log_type]['TraceId'] = log_dfs[log][log_type]['TraceId'].apply(lambda x: traces_to_idx[log][x])

    df_list = []
    for log_n, log_types in log_dfs.items():
        for log_t, results in log_types.items():
            results['model'] = len(results) * [log_n]
            results['log_type'] = len(results) * [log_t]
            df_list.append(results)

    log_dfs = pd.concat(df_list, ignore_index=True)

    log_dfs['conformance'] = log_dfs['conformance'].astype(float)
    log_dfs['completeness'] = log_dfs['completeness'].astype(float)
    log_dfs['confidence'] = log_dfs['confidence'].astype(float)
    log_dfs['ExecutionTime'] = log_dfs['ExecutionTime'].astype(float)
    log_dfs['TraceId'] = log_dfs['TraceId'].astype(int)

    return log_dfs

# data_set_dir = os.path.join('..','input','experiment-data')
# get_trace_map(os.path.join(data_set_dir, 'BPI_2012','BPI_2012_1k_sample.xes'))