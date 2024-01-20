import argparse

from pathlib import Path

from utils import read_and_vis_results

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path",type=str,default=Path("outputs/debug_T/output.csv"))
parser.add_argument("--img_path",type=str,default=Path("../data/1213_demo_data_v2/raw1"))
parser.add_argument("--pos_threshold",type=float,default=20)
arg=parser.parse_args()

read_and_vis_results(csv_path=arg.csv_path,img_path=arg.img_path,pos_threshold=arg.pos_threshold)