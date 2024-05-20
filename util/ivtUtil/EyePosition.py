"""
This file contains the class PupileAlg which is used to detect the pupile in a given image.
This uses opencv to detect the pupile.
The pupil detection is contrast based and is not perfect.
The parameters are not optimally tuned.
"""

import cv2
import numpy as np


class PupileAlg:
    """
    This class is used to detect the pupile in a given image.
    """
    def get_pupile(self, img):
        """
        Get the pupile in a given image.
        :param img:  the image
        :return:  the x, y and radius of the pupile
        """
        img = np.array(img, dtype=np.uint8)
        inv = cv2.bitwise_not(img)
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv2.erode(inv, kernel, iterations=1)
        ret, thresh1 = cv2.threshold(erosion, 220, 255, cv2.THRESH_BINARY)
        cnts, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        radius = 4
        x, y = None, None
        if len(cnts) != 0:
            c = max(cnts, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(img, center, radius, (255, 0, 0), 2)
        return x, y, radius

    def get_pupile_from_vid(self, seq):
        """
        Get the pupile from a video sequence.
        :param seq:  the video sequence
        :return:  list of x, y and radius of the pupile
        """
        pos_buffer = []
        for i, img in enumerate(seq):
            x, y, radius = self.get_pupile(img)
            if x is not None and y is not None:
                pos_buffer.append((x, y, radius))
            elif len(pos_buffer) > 0:
                _x, _y, _ = pos_buffer[-1]
                pos_buffer.append((x if x is not None else _x, y if y is not None else _y, radius))
            else:
                pos_buffer.append((x, y, radius))
        return pos_buffer

    def get_pupile_from_batch(self, batch):
        """
        Get the pupile from a batch of videos.
        :param batch:  the batch of videos
        :return:  list of lists of x, y and radius of the pupile
        """
        batch = np.array(batch)
        batch_buffer = []
        for vid in batch:
            vid = np.moveaxis(vid, 0, -1)
            batch_buffer.append(self.get_pupile_from_vid(vid))
        return batch_buffer

    def get_from_loader(self, loader, with_images=False):
        """
        Get the pupile from a loader.
        Used if you want to use a loader and loop sequentially
        :param loader: the loader
        :param with_images: if True, return the images as well
        :return: the pupile positions
        """
        for batch in loader:
            if with_images:
                yield batch, self.get_pupile_from_batch(batch)
            else:
                yield self.get_pupile_from_batch(batch)


def test():
    """
    Test the pupile detection
    """

    import sys
    from pathlib import Path
    path_root = Path(__file__).parents[2]
    sys.path.append(str(path_root))
    from util.dataUtils.dataset_loader import OpenEDSLoader
    from util.dataUtils.transformations import Crop_top, Crop

    transformations = [
        Crop_top(20),  # centers the image better
        Crop((256, 256)),
    ]

    relative_path = "" #'../../'
    root = relative_path + 'data/openEDS/openEDS'
    save_path = relative_path + 'data/openEDS/openEDSSplit.npy'
    loader = OpenEDSLoader(root, batch_size=8, shuffle=True, max_videos=None, save_path=save_path,
                           save_anyway=False,
                           transformations=transformations, sim_clr=False, split_frames=6)
    train_loader, test_loader, _ = loader.get_loaders()
    pupile_alg = PupileAlg()
    for img, bach in pupile_alg.get_from_loader(train_loader, with_images=True):
        for vid, poss in zip(img, bach):
            video = np.moveaxis(np.array(vid, dtype=np.uint8), 0, -1)
            for i, pos in enumerate(poss):
                img_show = video[i]
                img_show = img_show.copy()
                center = (int(pos[0] if pos[0] is not None else 0), int(pos[1] if pos[1] is not None else 0))
                cv2.circle(img_show, center, int(pos[2]), (255, 255, 255), 2)
                cv2.imshow("pupile", img_show)
                cv2.waitKey(500)
        break


if __name__ == '__main__':
    test()