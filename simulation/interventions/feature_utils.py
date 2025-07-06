import json
import numpy as np
import sys
sys.path.append('/home/morg/students/ohavbarbi/multiAgent/')
from classifiers.metric_utils import get_score, METRIC_COLUMNS

def try_int(x):
    try:
        return int(x)
    except Exception as e:
        return -1

def get_gen_metrics(log):
    metrics = {'entropy': [], 'varentropy': [], 'kurtosis': []}
    gen_length = len(log)
    max_text = ''
    max_score = 0
    for i in range(gen_length):
        cur_comp = log[i]
        if 'ton' in cur_comp[0][0]:
            for d in range(1, 5):
                if try_int(log[i-d][0][0]) < 0:
                    d = d -1
                    break
            if d == 0:
                continue
            text = ''.join([t[0][0] for t in log[i-d:i+1]])
            
            for feature in metrics.keys():
                logprobs_in_text = np.array([[opt[1] for opt in log[i-j]] for j in range(1, d+1)])
                cur_scores = []
                for token in range(logprobs_in_text.shape[0]):
                    cur_scores.append(get_score(logprobs_in_text[token, :], score_method=feature))
                cur_score = max(cur_scores)
                metrics[feature].append(cur_score)
                if feature == 'entropy' and (cur_score > max_score):
                    max_score = cur_score
                    max_text = text
                    
    return metrics, max_text