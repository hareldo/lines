import argparse
import os
import cv2
import numpy as np
import pandas as pd


def get_all_lines(line_series):
    lines = []
    for i, line in line_series.iterrows():
        x = line['x']
        y = line['y']
        if len(x) > 2 or len(y) > 2:
            continue
        lines.append([[x[0], y[0]], [x[1], y[1]]])
    return np.array(lines)


def to_int(x):
    return tuple(map(int, x))


def save_heatmap(prefix, image, lines):
    im_rescale = (512, 512)
    heatmap_scale = (128, 128)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]

    lcmap = np.zeros(heatmap_scale, dtype=np.float32)
    lcoff = np.zeros((2,) + heatmap_scale, dtype=np.float32)
    lleng = np.zeros(heatmap_scale, dtype=np.float32)
    angle = np.zeros(heatmap_scale, dtype=np.float32)
    lpos = []

    # the coordinate of lines can not equal to 128 (less than 128).
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    lines = lines[:, :, ::-1]  # change position of x and y --> (r, c)

    junc = []
    jids = {}

    # collecting junction endpoints (jun) and number them in dictionary (junc, jids).
    def jid(jun):
        jun = tuple(jun[:2])
        if jun in jids:
            return jids[jun]
        jids[jun] = len(junc)
        junc.append(np.array(jun + (0,)))
        return len(junc) - 1

    for v0, v1 in lines:
        v = (v0 + v1) / 2
        vint = to_int(v)
        lcmap[vint] = 1
        lcoff[:, vint[0], vint[1]] = v - vint - 0.5
        lleng[vint] = np.sqrt(np.sum((v0 - v1) ** 2)) / 2  # L
        lpos.append([junc[jid(v0)], junc[jid(v1)]])

        if v0[0] <= v[0]:
            vv = v0
        else:
            vv = v1

        # the angle under the image coordinate system (r, c)
        # theta means the component along the c direction on the unit vector
        if np.sqrt(np.sum((vv - v) ** 2)) <= 1e-4:
            continue
        angle[vint] = np.sum((vv - v) * np.array([0., 1.])) / np.sqrt(np.sum((vv - v) ** 2))  # theta

    image = cv2.resize(image, im_rescale)

    lpos = np.array(lpos, dtype=np.float32)
    np.savez_compressed(
        f"{prefix}_line.npz",
        lcmap=lcmap,
        lcoff=lcoff,
        lleng=lleng,
        lpos=lpos,
        angle=angle,
    )
    cv2.imwrite(f"{prefix}.png", image)


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
                im = cv2.imread(os.path.join(root, "images", image))
                save_heatmap(os.path.join(output_dir, image.split('.')[0]), im, lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Dataset for line detection')
    parser.add_argument('csv_root_dir', type=str, default='')
    parser.add_argument('output_dir', type=str, default='')
    parser.add_argument('version', type=str, default='0.1')
    opt = parser.parse_args()
    opt.output_dir = os.path.join(opt.output_dir, opt.version)
    os.makedirs(opt.output_dir, exist_ok=True)
    create_dataset(opt.csv_root_dir, opt.output_dir)
