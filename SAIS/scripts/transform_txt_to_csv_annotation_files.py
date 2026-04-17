import os
import pandas as pd


def _infer_subject(video_name):
    # Example: Suturing_B001 -> Subject02 (B->2), Knot_Tying_H005 -> Subject08
    token = video_name.split('_')[-1]
    letter = token[0]
    if 'A' <= letter <= 'Z':
        subject_num = ord(letter) - ord('A') + 1
        return f'Subject{subject_num:02d}'
    return 'Subject00'


def _infer_supertrial(video_name):
    # Example: Suturing_B001 -> Trial001
    token = video_name.split('_')[-1]
    digits = ''.join(ch for ch in token if ch.isdigit())
    if digits:
        return f'Trial{int(digits):03d}'
    return 'Trial000'


def generate_jigsaws_loader_csv(dataset_root, output_dir):
    """Generate one CSV per surgical task under output_dir.

    Output schema:
    id, label, Path, Gesture, Subject, SuperTrial, StartFrame, EndFrame

    'label' == 'Path' (video identifier expected by extract_representations.py)
    """

    tasks = [
        ('Knot_Tying',     'Knot_Tying'),
        ('Needle_Passing', 'Needle_Passing'),
        ('Suturing',       'Suturing'),
    ]
    
    print(f'Generating CSV annotation files for tasks: {[t[1] for t in tasks]}')

    os.makedirs(output_dir, exist_ok=True)

    for task_folder, task_name in tasks:
        transcription_dir = os.path.join(
            dataset_root, task_folder, task_folder, 'transcriptions'
        )
        print(f'Processing task: {task_name} from {transcription_dir}...')
        if not os.path.isdir(transcription_dir):
            print(f'Skipping missing directory: {transcription_dir}')
            continue

        rows = []
        row_id = 1

        for file_name in sorted(os.listdir(transcription_dir)):
            if not file_name.endswith('.txt'):
                continue

            video_name = os.path.splitext(file_name)[0]
            print(f'Processing {video_name} from {transcription_dir}...')

            video_path = os.path.join('SAIS', 'videos', f'{video_name}.mp4')
            # Keep backslashes so current loader parsing remains compatible.
            video_path = video_path.replace('/', '\\')
            print(f'Expected video path: {video_path}')

            annotation_file = os.path.join(transcription_dir, file_name)
            with open(annotation_file, 'r') as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue

                    start_frame = int(parts[0])
                    end_frame   = int(parts[1])
                    gesture     = parts[2].strip()

                    rows.append(
                        {
                            'id':         row_id,
                            'label':      video_path,   # <-- required by extract_representations.py
                            'Path':       video_path,
                            'Gesture':    gesture,
                            'Subject':    _infer_subject(video_name),
                            'SuperTrial': _infer_supertrial(video_name),
                            'StartFrame': start_frame,
                            'EndFrame':   end_frame,
                        }
                    )
                    row_id += 1

        if not rows:
            print(f'No data found for task {task_name}, skipping CSV generation.')
            continue

        df = pd.DataFrame(
            rows,
            columns=['id', 'label', 'Path', 'Gesture', 'Subject', 'SuperTrial', 'StartFrame', 'EndFrame']
        )

        output_csv = os.path.join(output_dir, f'JIGSAWS_{task_name}_FlowPaths.csv')
        df.to_csv(output_csv, index=False)
        print(f'Saved {len(df)} rows -> {output_csv}')

    print('Done.')


if __name__ == '__main__':
    dataset_root = './SAIS/datasets/jigsaw'
    output_dir   = './SAIS/paths'
    generate_jigsaws_loader_csv(dataset_root, output_dir)
