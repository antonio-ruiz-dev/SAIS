import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from operator import itemgetter
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f','--videoname',nargs='*',type=str)
parser.add_argument('-p','--path',type=str)
args = parser.parse_args()

starttime = time.time()

dataset = 'Custom'
# Other possible datasets: 'VUA_EASE', 'VUA_EASE_Stitch', 'NS_DART', 'NS_Gestures_Classification', 'VUA_Gestures_Classification', 'DVC_UCL_Gestures_Classification', 'JIGSAWS_Suturing_Gestures_Classification', 'NS_vs_VUA', 'CinVivo_OutView' 
# NS, VUA, NS_Gronau, VUA_Gronau, RAPN, VUA_COH, VUA_HMH, VUA_Lab, JIGSAWS_Suturing, DVC_UCL in extract_representations.py:478
# task-level names like VUA_EASE, NS_DART, Custom_Gestures, etc. in run_experiments.py:21
savepath = os.path.join(args.path,'paths') # project directory
if not os.path.exists(savepath):
    os.mkdir(savepath) 

# Generate Frame Paths
search_path = os.path.join(args.path,'images') # path to the images directory
load_path = 'images'
df = pd.DataFrame(columns=['path','category','label'])
# If --videoname is provided, use that list; otherwise use all video folders in images.
if args.videoname:
    cases = args.videoname
else:
    cases = sorted(
        case for case in os.listdir(search_path)
        if os.path.isdir(os.path.join(search_path, case))
    )
print(f'Generating paths for {len(cases)} video(s)...')
for case in tqdm(cases):
    casepath = os.path.join(search_path,case)
    files = sorted(os.listdir(casepath))
    print(f'Found {len(files)} frames in {casepath}')
    filepaths = list(map(lambda file:os.path.join(load_path,case,file),files))
    
    curr_df = pd.DataFrame(filepaths,columns=['path'])
    curr_df['category'] = case
    curr_df['label'] = case
    df = pd.concat((df,curr_df),axis=0)

df.to_csv(os.path.join(savepath,'%s_Paths.csv' % dataset))

# Generate Flow Paths
df = pd.DataFrame(columns=['path1','path2','category','label'])
#cases = sorted(os.listdir(path))
jump_frames = 15 # number of frames to skip = fps // 2
for case in tqdm(cases):
    casepath = os.path.join(search_path,case)
    files = sorted(os.listdir(casepath))
    indices = np.arange(0,len(files)-jump_frames,jump_frames)
    files = [files[idx] for idx in indices]
    filepaths = list(map(lambda file:os.path.join(load_path,case,file),files))

    frames = list(map(lambda file:int(file.split('_')[-1].strip('.jpg')),files))
    next_frames = list(map(lambda frame:frame + jump_frames,frames))
    next_files = list(map(lambda frame:'frames_' + '0'*(8-len(str(frame))) + str(frame) + '.jpg',next_frames))
    next_filepaths = list(map(lambda file:os.path.join(load_path,case,file),next_files))

    curr_df = pd.DataFrame(filepaths,columns=['path1'])
    curr_df['path2'] = next_filepaths
    curr_df['category'] = case
    curr_df['label'] = case
    df = pd.concat((df,curr_df),axis=0)

df.to_csv(os.path.join(savepath,'%s_FlowPaths.csv' % dataset))

df = pd.read_csv(os.path.join(savepath,'%s_FlowPaths.csv' % dataset),index_col=0)
df['nflow'] = df[['path1','label']].apply(lambda row:int(row['path1'].split('frames_')[-1].strip('.jpg')) // jump_frames, axis=1)
df['flowpath'] = df[['path1','label','nflow']].apply(lambda row:os.path.join('flows',row['label'],'flows_' + '0'*(8-len(str(row['nflow']))) + str(row['nflow']) + '.jpg'),axis=1)
df.drop(labels=['nflow'],axis=1,inplace=True)

df.to_csv(os.path.join(savepath,'%s_FlowPaths.csv' % dataset))

diff = time.time() - starttime
print("Time taken (s): %.3f" % diff)
