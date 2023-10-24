import argparse
import json
import os

INSTRUCTION = "Make the summary in a factual but detailed way."

def preprocess(args):
    # load prediction and gt summaries
    with open(args.pred_path) as f:
        pred = json.load(f)
    with open(args.gt_path) as f:
        gt = json.load(f)

    gt = {os.path.basename(key): val for key, val in gt.items()}
    print(f"# prediction: {len(pred)}, # gt: {len(gt)}")

    out = []
    for vid, pred_i in pred.items():
        vid = vid.replace('.','_')
        if vid[-4:] in ['_mp4', '_mkv']:
            vid = vid[:-4]
        gt_i = gt.pop(vid, None)
        if gt_i is None:
            print(f"skipping {vid}: gt not exists")
            continue
        if not isinstance(gt_i, str):
            gt_i = " ".join(gt_i)
        out.append({
            "video_name": vid,
            "Q": INSTRUCTION,
            "A": gt_i,
            "pred": pred_i
        })
    print(f"# processed: {len(out)}")

    if args.fill_na:
        # fill non-existing samples with None.
        for remaining in gt.keys():
            out.append({
                "video_name": remaining,
                "Q": INSTRUCTION,
                "A": gt[remaining],
                "pred": None
            })
        print(f"After filling NA: {len(out)}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, 'w') as f:
        f.write(json.dumps(out, indent=4))
    print("Finished")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocessing data for QEFVC evaluation.")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--gt_path", required=True, help="The path to file containing gt.")
    parser.add_argument("--out_path", required=True, help="The path to save preprocessed json files.")
    parser.add_argument("--fill_na", action='store_true', help="Fill non-existing samples with None.")
    args = parser.parse_args()

    preprocess(args)