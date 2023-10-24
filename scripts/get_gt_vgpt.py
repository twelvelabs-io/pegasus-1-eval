import argparse
import json
import os
from glob import glob
from tqdm import tqdm

def main(args):
    anns = glob(os.path.join(args.input_path,'*.txt'))
    anns = sorted(anns)
    print(f"# original samples: {len(anns)}")

    out = {}
    for ann in tqdm(anns):
        with open(ann,'r',encoding='utf-8-sig') as f:
            vid = '.'.join(os.path.basename(ann).split('.')[:-1])
            gt = f.read()
        out[vid] = gt

    with open(args.out_path, 'w') as f:
        f.write(json.dumps(out, indent=4))
    print(f"# processed samples: {len(out)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="make MSR-VTT gt json file.")
    parser.add_argument("--input_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--out_path", required=True, help="The path to save gt json files.")
    args = parser.parse_args()
    main(args)