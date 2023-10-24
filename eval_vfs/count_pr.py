import numpy as np
import json
import glob

def format_float(f):
    return '{:.2f}'.format(f * 100)

    
def print_stats(exp_name, fp, gt):
    with open(fp, 'r') as fp:
        pr = json.load(fp)
    print(exp_name, len(pr))
    

    precs = []
    recalls = []
    num_wrong = 0
    valid_precs = 0
    valid_recs = 0
    num_gt_corrupt =0
    num_invalid_prec = 0
    for vii, vkey in enumerate(gt):
        try:
            if vkey not in pr:
                # for many reasons, the video could not be successfully processed, In this case, we consider this as a failure case.
                precs.append(0)
                recalls.append(0)
                continue

            _cur_precision = []
            _cur_recall = []
            for _pre in pr[vkey]['pre']:
                if _pre['result'].lower().strip() == 'yes':
                    _cur_precision.append(1)
                elif _pre['result'].lower().strip() == 'no':
                    _cur_precision.append(0)
                else:
                    _cur_precision.append(0)
                    num_wrong += 1

            for _rec in pr[vkey]['recall']:
                if _rec['result'].lower().strip() == 'yes':
                    _cur_recall.append(1)
                elif _rec['result'].lower().strip() == 'no':
                    _cur_recall.append(0)
                else:
                    _cur_recall.append(0)
                    num_wrong += 1

            # when _cur_recall is empty, it means that GT is corrupted.
            # Therefore we just that video

            # when _cur_precision is empty, it means that the model is responsible for the error
            # therefore we add 0 for cur_precision, cur_recall 0 when penalize_empty is True

            if len(_cur_recall) == 0:
                num_gt_corrupt += 1
                continue
            
            if len(_cur_precision) == 0:
                # empty _cur_precision means failure of the model
                num_invalid_prec += 1
                precs.append(0)
                recalls.append(0)
          
            else:
                valid_precs += 1
                precs.append(np.mean(_cur_precision))
                recalls.append(np.mean(_cur_recall))
           
        
        except:
            import traceback
            traceback.print_exc()
            print(f'error at {vkey}')

    mean_recall = np.mean(recalls)
    mean_precision =  np.mean(precs)
    print(f'mean_precision: {format_float(mean_precision)} for valid precs {valid_precs} ')
    print(f'mean_recall: {format_float(mean_recall)} for valid recs {valid_recs} ')
    # print(f'num_gt_corrupt: {num_gt_corrupt}, num_invalid_prec: {num_invalid_prec}')
    
    
    f1 = (2*mean_precision*mean_recall)/(mean_precision+mean_recall)
    print('f1: ', format_float(f1))
    print(f'num_wrong: {num_wrong}')
    return

if __name__ == '__main__':

    json_files = glob.glob('/home/ubuntu/ray/pegasus-1-eval/vfs_result/*.json')
    for _jf in json_files:
        exp_name = _jf.split('/')[-1].replace('.json', '')
        dataname = exp_name.split('_')[1]
        if dataname == 'vgpt':
            continue
        if dataname == 'msrvtt':
            with open('../vfs_resource/GT_msrvtt_src.json', 'r') as fp:
                gt = json.load(fp)
        elif dataname == 'vgpt':
            with open('../vfs_resource/GT_vgpt_src.json', 'r') as fp:
                gt = json.load(fp)
        else:
            raise NotImplementedError
        # with open(_jf, 'r') as fp:
        #     pr = json.load(fp)
        print_stats(exp_name, _jf, gt)
        print('------------')

'''
asronly_vgpt_precision_recall 497
mean_precision: 13.08 for valid precs 492 
mean_recall: 16.38 for valid recs 0 
num_gt_corrupt: 1, num_invalid_prec: 4
num_wrong: 1
------------
vendorA_vgpt_precision_recall 498
mean_precision: 12.82 for valid precs 416 
mean_recall: 6.21 for valid recs 0 
num_gt_corrupt: 1, num_invalid_prec: 81
num_wrong: 9
------------
videogpt_vgpt_precision_recall 499
mean_precision: 43.61 for valid precs 465 
mean_recall: 26.49 for valid recs 0 
num_gt_corrupt: 1, num_invalid_prec: 33
num_wrong: 0
------------
pegasus_vgpt_precision_recall 499
mean_precision: 58.09 for valid precs 498 
mean_recall: 39.96 for valid recs 0 
num_gt_corrupt: 1, num_invalid_prec: 0
num_wrong: 0
------------
'''

'''
asronly_msrvtt_precision_recall 1000
mean_precision: 28.16 for valid precs 874 
mean_recall: 25.43 for valid recs 0 
num_gt_corrupt: 0, num_invalid_prec: 126
num_wrong: 1
------------
vendorA_msrvtt_precision_recall 1000
mean_precision: 29.28 for valid precs 692 
mean_recall: 24.38 for valid recs 0 
num_gt_corrupt: 0, num_invalid_prec: 308
num_wrong: 2
------------
videogpt_msrvtt_precision_recall 1000
mean_precision: 37.58 for valid precs 920 
mean_recall: 27.24 for valid recs 0 
num_gt_corrupt: 0, num_invalid_prec: 80
num_wrong: 0
------------
pegasus_msrvtt_precision_recall 1000
mean_precision: 57.67 for valid precs 1000 
mean_recall: 46.88 for valid recs 0 
num_gt_corrupt: 0, num_invalid_prec: 0
num_wrong: 0
------------
'''