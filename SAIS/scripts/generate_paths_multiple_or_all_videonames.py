import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f','--videoname',nargs='*',type=str)
parser.add_argument('-p','--path',type=str)
args = parser.parse_args()

starttime = time.time()

# dataset = 'Custom'
dataset = 'JIGSAWS'
savepath = os.path.join(args.path,'paths') # project directory
if not os.path.exists(savepath):
    os.mkdir(savepath) 

# Generate Frame Paths
search_path = os.path.join(args.path,'images') # path to the images directory
videos_path = os.path.join(args.path,'videos') # path to original videos directory
load_path = 'images'
df = pd.DataFrame(columns=['path','category','label'])

# If video names are provided, use them. Otherwise, process all videos in ./SAIS/SAIS/videos.
if args.videoname:
    cases = args.videoname
else:
    if not os.path.isdir(videos_path):
        raise FileNotFoundError(f'Videos directory not found: {videos_path}')
    cases = sorted(
        os.path.splitext(file)[0]
        for file in os.listdir(videos_path)
        if os.path.isfile(os.path.join(videos_path, file)) and file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.mpeg', '.mpg', '.m4v'))
    )

if not cases:
    raise ValueError('No video names provided and no video files found in videos directory.')

print(f'Generating paths for {len(cases)} video(s)...')
for case in tqdm(cases):
    casepath = os.path.join(search_path,case)
    if not os.path.isdir(casepath):
        print(f'Skipping missing frame directory: {casepath}')
        continue
    files = sorted(os.listdir(casepath))
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
    if not os.path.isdir(casepath):
        continue
    files = sorted(os.listdir(casepath))
    if len(files) <= jump_frames:
        continue
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
