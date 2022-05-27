#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.config import CSI300_BENCH


if __name__ == "__main__":

    # use default data
    provider_uri = "~/autodl-nas/qlib_data/cn_data_wind/"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    
    task_config = {'model': {'class': 'LGBModel',
                             'module_path': 'qlib.contrib.model.gbdt',
                             'kwargs': {'loss': 'mse',
                                        'colsample_bytree': 0.8879,
                                        'learning_rate': 0.0421,
                                        'subsample': 0.8789,
                                        'lambda_l1': 205.6999,
                                        'lambda_l2': 580.9768,
                                        'max_depth': 8,
                                        'num_leaves': 210,
                                        'num_threads': 20}},
                   'dataset': {'class': 'DatasetH',
                               'module_path': 'qlib.data.dataset',
                               'kwargs': {'handler': {'class': 'Alpha158',
                                                      'module_path': 'qlib.contrib.data.handler',
                                                      'kwargs': {'start_time': '2016-01-01',
                                                                 'end_time': '2022-03-01',
                                                                 'fit_start_time': '<dataset.kwargs.segments.train.0>',
                                                                 'fit_end_time': '<dataset.kwargs.segments.train.1>',
                                                                 'instruments': 'sh000300'}},
                                          'segments': {'train': ('2016-01-01', '2018-12-31'),
                                                       'valid': ('2019-01-01', '2019-12-31'),
                                                       'test': ('2020-01-01', '2022-03-01')}}}}

    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2020-01-01",
            "end_time": "2022-03-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }


    # start exp
    with R.start(experiment_name="gbdt_csi300_test", recorder_id='test_001'):
        R.log_params(**flatten_dict(task_config))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
