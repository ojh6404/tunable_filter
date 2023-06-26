import argparse
import pprint
import cv2
import numpy as np

from tunable_filter.composite_zoo import BlurCropResolFilter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true', help='feedback mode')
    args = parser.parse_args()
    tuning = args.tune

    yaml_file_path = '/tmp/filter.yaml'
    img = cv2.imread('./media/dish.jpg', cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
    print('test img', img.shape)

    if tuning:
        tunable = BlurCropResolFilter.from_image(img)
        print('press q to finish tuning')
        tunable.launch_window()
        tunable.start_tuning(img)
        pprint.pprint(tunable.export_dict())
        tunable.dump_yaml(yaml_file_path)
    else:
        f = BlurCropResolFilter.from_yaml(yaml_file_path)
        img_filtered = f(img)
        cv2.imshow('debug', img_filtered)
        print('img_filtered shape')
        print(img_filtered.shape)
        print('press q to terminate')
        while True:
            if cv2.waitKey(50) == ord('q'):
                break
