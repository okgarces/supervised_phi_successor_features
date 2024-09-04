from time import gmtime, strftime

import numpy as np
import pandas as pd
import json

class Logger:
    SOURCE_TASK = 'source'
    TARGET_TASK = 'target'

    HEADERS = ['task_id', 'reward', 'step', 'accum_loss', 'q_loss', 'psi_loss', 'phi_loss']

    def __init__(self, root_path, prefix=None):
        prefix = prefix+'_' if prefix is not None else ''
        self.log_task_file = f'{root_path}results/{prefix}log_performance_{strftime("%d_%b_%Y_%H_%M_%S", gmtime())}.csv'

    def log(self, log_dictionary):
        filename = self.log_task_file
        with open(filename, 'a') as f:
            f.write(json.dumps(str(log_dictionary)) + '\n')
            # np.savetxt(f, json.dumps(log_dictionary), delimiter=',', newline='\n')

    def log_agent_performance(self, task, reward, step, accum_loss, *args, **kwargs):
        values = np.array([task, reward, step, accum_loss, *args])
        type_task = kwargs.get('type_task', self.SOURCE_TASK)
        filename = self.source_tasks_file if type_task == self.SOURCE_TASK else self.target_tasks_file

        with open(filename, 'a') as f:
            np.savetxt(f, np.column_stack(values), delimiter=',', newline='\n')

    def load_text(self, type_task='source'):
        filename = self.source_tasks_file if type_task == self.SOURCE_TASK else self.target_tasks_file
        return pd.DataFrame(np.loadtxt(filename, delimiter=','))

