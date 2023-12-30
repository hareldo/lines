import argparse
import os
import pandas as pd


def get_all_lines(line_series):
    for line in line_series:
        x = line['x']
        y = line['y']



def create_dataset(root, output_dir):
    for filename in os.listdir(root):
        if filename[-3:] == "csv":
            df = pd.read_csv(os.path.join(root, filename), index_col=False)
            df['region_shape_attributes'] = df['region_shape_attributes'].apply(lambda x: eval(x))
            df['x'] = [region['all_points_x'] for region in df['region_shape_attributes'].to_list()]
            df['y'] = [region['all_points_y'] for region in df['region_shape_attributes'].to_list()]
            images = df['filename'].drop_duplicates()
            for image in images:
                annotation_df = df[df['filename'] == image]
                lines = get_all_lines(annotation_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Dataset for line detection')
    parser.add_argument('csv_root_dir', type=str, default='')
    parser.add_argument('output_dir', type=str, default='')
    parser.add_argument('version', type=str, default='0.1')
    opt = parser.parse_args()
    opt.output_dir = os.path.join(opt.output_dir, opt.version)
    os.makedirs(opt.output_dir, exist_ok=True)
    create_dataset(opt.csv_root_dir, opt.output_dir)
