import numpy as np
import pandas as pd
import time, os, argparse
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict
from pandas.api.types import CategoricalDtype
import re

import hmmconf
from hmmconf import metric, pm_extra
from pm4py import read_pnml as pnml_importer
from pm4py.visualization.transition_system import util


np.set_printoptions(precision=2)


logger = hmmconf.utils.make_logger(__file__)


# DATA_DIR = os.path.join('input','trie-stream')
# NET_FNAME = os.path.join('Petri-net_models','M-nets','M1.pnml')
# LOG_FNAME = os.path.join('M1.csv')

# NET_FP = os.path.join(DATA_DIR, NET_FNAME)
# LOG_FP = os.path.join('output','hmmconf','hmmconf_logs','M-logs','M1',LOG_FNAME)

MODEL_FP = os.path.join('input\\trie-stream\\Petri-net_models')
DATA_FP = os.path.join('output\\hmmconf\\hmmconf_logs')
RESULT_DIR = os.path.join('results', 'hmmconf')

#STORE_DIR = os.path.join('..', 'output','hmmconf','hmmconf_res')

#STORE_FP = os.path.join(STORE_DIR, 'M1.h5')
#STORE_HMMCONF_FP = os.path.join('M1.h5')

ACTIVITY = 'activity'
ACTIVITY_ID = 'activity_id'
CASEID = 'caseid'

# EM params
N_JOBS = 'n_jobs'
N_ITER = 'n_iters'
TOL = 'tol'
RANDOM_SEED_PARAM = 'random_seed'
N_FOLDS = 'n_folds'
IS_TEST = 'is_test'
CONF_TOL = 'conformance_tol'
PRIOR_MULTIPLIER = 'prior_multiplier'
EM_PARAMS = 'em_params'
MAX_N_CASE = 'max_n_case'


# experiment configurations
EXPERIMENT_CONFIGS = {
    N_JOBS: 1,#mp.cpu_count() - 1,
    N_ITER: 1,
    TOL: 1e-2,
    RANDOM_SEED_PARAM: 123,
    N_FOLDS: 5,
    IS_TEST: False,
    CONF_TOL: 0,
    PRIOR_MULTIPLIER: 1.,
    EM_PARAMS: 'to',
    MAX_N_CASE: 10000
}


def experiment_configs2df(configs):
    items = sorted(list(configs.items()), key=lambda t: t[0])
    columns, values = zip(*items)
    return pd.DataFrame([values], columns=columns)


def map_net_activity(net, actmap):
    for t in net.transitions:
        if t.label:
            if t.label in actmap:
                t.label = actmap[t.label]
            else:
                actmap[t.label] = len(actmap)
                t.label = actmap[t.label]


def estimate_conform_params(event_df, state2int, obs2int, 
                            net, init_marking, final_marking,
                            is_inv, add_prior=True, multiplier=1.):
    # group cases
    grouped_by_caseid = event_df.groupby('caseid')
    cases = list()

    for caseid, case_df in grouped_by_caseid:
        case = case_df['activity']
        cases.append((caseid, case))

    results = hmmconf.get_counts_from_log(
        cases, state2int, obs2int,
        net, init_marking, final_marking, is_inv
    )
    trans_count, emit_count, conforming_cid = results

    # get pseudo counts
    if add_prior:
        is_inv_rg = lambda t: t.name is None
        rg, inv_states = hmmconf.build_reachability_graph(net, init_marking, is_inv)
        init = pm_extra.get_init_marking(rg)
        trans_pseudo_count = hmmconf.get_pseudo_counts_transcube(rg, init, 
                                                                 is_inv_rg, 
                                                                 state2int, obs2int, multiplier)
        emit_pseudo_count = hmmconf.get_pseudo_counts_emitmat(rg, init, 
                                                              is_inv_rg, 
                                                              state2int, obs2int, multiplier)

        transcube = hmmconf.estimate_transcube(trans_count, trans_pseudo_count)
        emitmat = hmmconf.estimate_emitmat(emit_count, emit_pseudo_count)
    else:
        transcube = hmmconf.estimate_transcube(trans_count)
        emitmat = hmmconf.estimate_emitmat(emit_count)
        
    return transcube, emitmat, conforming_cid


def event_df_to_hmm_format(df):
    lengths = df.groupby('caseid').count()['activity_id'].values
    X = df[['activity_id']].values
    return X, lengths


class ConformanceObserver:
    def __init__(self):
        self.emitconf = defaultdict(list)
        self.stateconf = defaultdict(list)

    def update(self, status):
        self.emitconf[status.caseid].append(status.last_emitconf)
        self.stateconf[status.caseid].append(status.last_stateconf)


def run(model_fp: str, data_fp: str, conf_out: str, model_name, log_name):

        results_dir = os.path.join(RESULT_DIR, model_name.replace('.pnml', ''))
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)

        start_all = time.time()
        configs_df = experiment_configs2df(EXPERIMENT_CONFIGS)
        info_msg = 'Experiment configuration: \n{}'.format(configs_df)
        logger.info(info_msg)

        net_orig, init_marking_orig, final_marking_orig = pnml_importer(model_fp)
        net, init_marking, final_marking = pnml_importer(model_fp)
        #store = pd.HDFStore(STORE_FP, mode='r')
        #case_prefix_df = store['case_prefix_df']
        log_df = pd.read_csv(data_fp)
        activity_list = sorted(log_df['activity'].unique())
        activity_cat_type = CategoricalDtype(categories=activity_list)

        logger.info('Mapping activity to integer labels')
        obs2int = dict(enumerate(activity_cat_type.categories))
        obs2int = {v:k for k, v in obs2int.items()}
        int2obs = {key:val for key, val in obs2int.items()}
        obs2int_df = pd.DataFrame(list(obs2int.items()), columns=['activity', 'activity_int'])
        info_msg = 'Activity 2 int dataframe: \n{}'.format(obs2int_df)
        logger.info(info_msg)
        map_net_activity(net, obs2int)

        # store_hmmconf = pd.HDFStore(STORE_HMMCONF_FP, mode='w')
        # store_hmmconf['config_df'] = configs_df

        test_split = len(log_df) - int(len(log_df) * 0.2)
        caseid_first = log_df.iloc[test_split]['caseid']
        first_ = log_df[log_df['caseid'] == caseid_first].iloc[0]['id'].astype(int) - 1 #caseid starts from 1
        train_event_df = log_df.iloc[:first_]
        test_event_df = log_df.iloc[first_:-1]

        train_event_df['activity_id'] = train_event_df['activity'].astype(activity_cat_type).cat.codes
        test_event_df['activity_id'] = test_event_df['activity'].astype(activity_cat_type).cat.codes

        logger.info('Process net...')
        is_inv = lambda t: t.label is None
        rg, inv_states = hmmconf.build_reachability_graph(net, init_marking, is_inv)
        sorted_states = sorted(list(rg.states), key=lambda s: (s.data['disc'], s.name))
        node_map = {key:val for val, key in enumerate(map(lambda state: state.name, sorted_states))}
        int2state = {val:key for key, val in node_map.items()}
        state2int = {val:key for key, val in int2state.items()}

        logger.info('Setting up HMM...')
        is_inv_rg = lambda t: t.name is None
        init = hmmconf.get_init_marking(rg)
        startprob = hmmconf.compute_startprob(rg, state2int, is_inv_rg)
        conf_obsmap = {i:i for i in obs2int.values()}
        confmat = hmmconf.compute_confmat(rg, init, is_inv_rg, state2int, conf_obsmap)
        params = estimate_conform_params(
            train_event_df, state2int, obs2int, net_orig, init_marking_orig, final_marking_orig, is_inv
        )
        transcube, emitmat, conforming_caseid = params
        hmmconf_params = {
            'params': EXPERIMENT_CONFIGS[EM_PARAMS],
            'conf_tol': EXPERIMENT_CONFIGS[CONF_TOL],
            'n_iter': EXPERIMENT_CONFIGS[N_ITER],
            'tol': EXPERIMENT_CONFIGS[TOL],
            'verbose': True,
            'n_procs': EXPERIMENT_CONFIGS[N_JOBS],
            'random_seed': EXPERIMENT_CONFIGS[RANDOM_SEED_PARAM]
        }
        hmm = hmmconf.HMMConf(startprob, transcube, emitmat, confmat, int2state,
                            int2obs, **hmmconf_params)

        int2state_list = list(int2state.items())
        stateid_list, state_list = zip(*int2state_list)
        columns = ['state_id', 'state']
        state_id_df = pd.DataFrame({
            'state_id': stateid_list,
            'state': state_list
        })
        info_msg = 'State id df: \n{}'.format(state_id_df)
        logger.info(info_msg)

        logger.info('Make conformance tracker...')
        # add metrics as observers
        injected_dist_rows = list()
        def injected_distance_callback(caseid, event, metric):
            case_prefix = event
            if injected_dist_rows:
                last_row = injected_dist_rows[-1]
                if last_row[0] == caseid:
                    case_prefix = ', '.join([str(last_row[1]), str(event)])
            injected_dist_rows.append((caseid, case_prefix, metric[caseid]))

        injected_distance = metric.InjectedDistanceMetric.create(net, init_marking, is_inv, 
                                                                    injected_distance_callback)
        completeness_rows = list()
        def completeness_callback(caseid, event, metric):
            case_prefix = event
            if injected_dist_rows:
                last_row = injected_dist_rows[-1]
                if last_row[0] == caseid:
                    case_prefix = ', '.join([str(last_row[1]), str(event)])
            completeness_rows.append((caseid, case_prefix, metric[caseid]))

        completeness = metric.CompletenessMetric.create(net, init_marking, is_inv,
                                                        completeness_callback)
        conf_observer = ConformanceObserver()
        observers = [conf_observer, injected_distance, completeness]

        tracker = hmmconf.ConformanceTracker(hmm, max_n_case=EXPERIMENT_CONFIGS[MAX_N_CASE],
                                                observers=observers)

        # testing with less caseids
        # caseids = caseids[:100]
        train_caseids = train_event_df['caseid'].unique()
        n_train_caseids = train_caseids.shape[0]
        filter_by_caseids = train_event_df['caseid'].isin(train_caseids)
        filter_by_conform_caseids = train_event_df['caseid'].isin(conforming_caseid)
        filtered_train_event_df = train_event_df.loc[~filter_by_conform_caseids, :]
        n_train_caseids_filtered = filtered_train_event_df['caseid'].unique().shape[0]

        logger.info('Fitting with {}/{} non-conforming cases'.format(n_train_caseids_filtered, n_train_caseids))

        train_X, train_lengths = event_df_to_hmm_format(filtered_train_event_df)
        start_fit = time.time()
        tracker.hmm.fit(train_X, train_lengths)
        end_fit = time.time()
        took_fit = end_fit - start_fit
        info_msg = 'Training using {} cases took: {:.3f}s'
        info_msg = info_msg.format(n_train_caseids_filtered, took_fit)
        logger.info(info_msg)

        # save the 4 key params
        # logstartprob_fp = '{}_{}_fold-{}_logstartprob.npy'
        # logtranscube_fp = '{}_{}_fold-{}_logtranscube.npy'
        # logtranscube_d_fp = '{}_{}_fold-{}_logtranscube_d.npy'
        # logemitmat_fp = '{}_{}_fold-{}_logemitmat.npy'
        # logemitmat_d_fp = '{}_{}_fold-{}_logemitmat_d.npy'
        # confmat_fp = '{}_{}_fold-{}_confmat.npy'

        # logstartprob_fp_i = logstartprob_fp.format(LOG_FNAME, NET_FNAME, fold_id)
        # logtranscube_fp_i = logtranscube_fp.format(LOG_FNAME, NET_FNAME, fold_id)
        # logtranscube_d_fp_i = logtranscube_d_fp.format(LOG_FNAME, NET_FNAME, fold_id)
        # logemitmat_fp_i = logemitmat_fp.format(LOG_FNAME, NET_FNAME, fold_id)
        # logemitmat_d_fp_i = logemitmat_d_fp.format(LOG_FNAME, NET_FNAME, fold_id)
        # confmat_fp_i = confmat_fp.format(LOG_FNAME, NET_FNAME, fold_id)

        # save the 4 key params
        logstartprob_fp = '{}_{}_logstartprob.npy'
        logtranscube_fp = '{}_{}_logtranscube.npy'
        logtranscube_d_fp = '{}_{}_logtranscube_d.npy'
        logemitmat_fp = '{}_{}_logemitmat.npy'
        logemitmat_d_fp = '{}_{}_logemitmat_d.npy'
        confmat_fp = '{}_{}_confmat.npy'

        # logstartprob_fp_i = logstartprob_fp.format(LOG_FNAME, NET_FNAME)
        # logtranscube_fp_i = logtranscube_fp.format(LOG_FNAME, NET_FNAME)
        # logtranscube_d_fp_i = logtranscube_d_fp.format(LOG_FNAME, NET_FNAME)
        # logemitmat_fp_i = logemitmat_fp.format(LOG_FNAME, NET_FNAME)
        # logemitmat_d_fp_i = logemitmat_d_fp.format(LOG_FNAME, NET_FNAME)
        # confmat_fp_i = confmat_fp.format(LOG_FNAME, NET_FNAME)

        # logger.info('Saving learnt parameters')
        # with open(logstartprob_fp_i, 'wb') as f:
        #     np.save(f, tracker.hmm.logstartprob)
        # with open(logtranscube_fp_i, 'wb') as f:
        #     np.save(f, tracker.hmm.logtranscube)
        # with open(logtranscube_d_fp_i, 'wb') as f:
        #     np.save(f, tracker.hmm.logtranscube_d)
        # with open(logemitmat_fp_i, 'wb') as f:
        #     np.save(f, tracker.hmm.logemitmat)
        # with open(logemitmat_d_fp_i, 'wb') as f:
        #     np.save(f, tracker.hmm.logemitmat_d)
        # with open(confmat_fp_i, 'wb') as f:
        #     np.save(f, tracker.hmm.confmat)

        logger.info('Computing the state probability of both train_df and test_df')
        train_hmmconf_feature = list()
        test_hmmconf_feature = list()
        columns = ['caseid','activity', 'case_prefix'] 
        #columns += list(state_id_df['state'].values)
        columns += ['emitconf', 'stateconf', 'finalconf', 'injected_distance', 'completeness']

        case_prefixes = {}
        for row in train_event_df[['caseid', 'activity', 'activity_id']].itertuples(index=False):
            caseid = row.caseid
            if caseid not in case_prefixes:
                case_prefixes[caseid] = []
            event = row.activity_id
            act = row.activity
            case_prefixes[caseid].append(act)

            logfwd, finalconf, exception = tracker.replay_event(caseid, event)
            emitconf = conf_observer.emitconf[caseid][-1]
            stateconf = conf_observer.stateconf[caseid][-1]
            injected_dict = injected_dist_rows[-1][2]
            completeness = completeness_rows[-1][2]

            hmmconf_feature = [caseid, act, ''.join(case_prefixes[caseid])] #+ list(logfwd) 
            hmmconf_feature += [emitconf, stateconf, finalconf, injected_dict, completeness]
            train_hmmconf_feature.append(hmmconf_feature)
        
        case_prefixes.clear()
        for row in test_event_df[['caseid', 'activity', 'activity_id']].itertuples(index=False):
            caseid = row.caseid
            if caseid not in case_prefixes:
                case_prefixes[caseid] = []
            event = row.activity_id
            act = row.activity
            case_prefixes[caseid].append(act)

            logfwd, finalconf, exception = tracker.replay_event(caseid, event)
            emitconf = conf_observer.emitconf[caseid][-1]
            stateconf = conf_observer.stateconf[caseid][-1]
            injected_dict = injected_dist_rows[-1][2]
            completeness = completeness_rows[-1][2]

            hmmconf_feature = [caseid, act, ''.join(case_prefixes[caseid])] #+ list(logfwd)  
            hmmconf_feature += [emitconf, stateconf, finalconf, injected_dict, completeness]
            test_hmmconf_feature.append(hmmconf_feature)

        train_hmmconf_feature_df = pd.DataFrame.from_records(train_hmmconf_feature, columns=columns)
        test_hmmconf_feature_df = pd.DataFrame.from_records(test_hmmconf_feature, columns=columns)
        print('Train hmmconf feature df: \n{}'.format(train_hmmconf_feature_df.head()))
        print('Test hmmconf feature df: \n{}'.format(test_hmmconf_feature_df.head()))
        logger.info('Train hmmconf feature df: \n{}'.format(train_hmmconf_feature_df.head()))
        logger.info('Test hmmconf feature df: \n{}'.format(test_hmmconf_feature_df.head()))

        # out_fp = os.path.join(results_dir, log_name.replace('.csv', ''))

        train_hmmconf_feature_df.to_csv('train.csv')
        test_hmmconf_feature_df.to_csv(conf_out, index=None)

        err_msg = 'hmmconf feature df n_rows: {} != {}: event_df n_rows'
        err_msg_train = err_msg.format(train_hmmconf_feature_df.shape[0], train_event_df.shape[0])
        err_msg_test = err_msg.format(test_hmmconf_feature_df.shape[0], test_event_df.shape[0])
        assert train_hmmconf_feature_df.shape[0] == train_event_df.shape[0], err_msg_train
        assert test_hmmconf_feature_df.shape[0] == test_event_df.shape[0], err_msg_test

        # save to store
        # train_hmmconf_feature_df_name = 'train_hmmconf_feature_fold_{}_df'.format(fold_id)
        # test_hmmconf_feature_df_name = 'test_hmmconf_feature_fold_{}_df'.format(fold_id)
        # store_hmmconf[train_hmmconf_feature_df_name] = train_hmmconf_feature_df
        # store_hmmconf[test_hmmconf_feature_df_name] = test_hmmconf_feature_df

            # save to store
        #     if fold_id == 0:
        #         store_hmmconf['stateid_df'] = state_id_df

        # store.close()
        # store_hmmconf.close()
        end_all = time.time()
        took_all = (end_all - start_all) / 60
        logger.info('Took: {:.3f} mins'.format(took_all))


def extract_log_name(model_name: str) -> str:
    logs_ = '|'.join([str(log) for log in LOGS])
    return re.search(logs_, model_name)


if __name__ == '__main__':
    BPI_MODELS = os.path.join(MODEL_FP, 'BPI-nets')
    M_MODELS = os.path.join(MODEL_FP, 'M-nets')
    BPI_LOGS = os.path.join(DATA_FP, 'BPI-logs')
    M_LOGS = os.path.join(DATA_FP, 'M-logs')

    model_type = BPI_MODELS
    log_type = BPI_LOGS

    LOGS = ["BPI_2012", "BPI_2017", "M10", "M1", "M2",
           "M3", "M4", "M5", "M6", "M7", "M8", "M9"]


    for filename in os.listdir(model_type):
        result_dir = os.path.join(RESULT_DIR, filename.replace('.pnml', ''))
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        f_ = extract_log_name(filename).group()
        data_dir = os.path.join(log_type, f_)
        for log in os.listdir(data_dir):
            msg = 'Model: {}, ' \
                'Log: {}, '
            msg = msg.format(filename, log)
            print(msg)
            out_fp = os.path.join(result_dir, log)
            m_fp = os.path.join(model_type, filename)
            l_fp = os.path.join(data_dir, log)
            run(m_fp, l_fp, out_fp, filename, log)
