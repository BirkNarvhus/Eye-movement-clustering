"""
Implementation of the IVVt algorithm
used when plotting results from the eye tracking data
This is compared to results of other models in test_model.py
"""
import numpy as np

import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from util.ivtUtil.EyePosition import PupileAlg


class VelCalculator:
    """
    Calculates the velocity of the eye movement
    Uses pupil positions to calculate the velocity
    """
    def __init__(self, delt_time=6*1000/200):
        """
        the delta time is set to 6*1000/200 because the video is 6 seconds long and 200 frames
        :param delt_time: delta time between each frame in ms
        """
        self.delta_time = delt_time
        self.prev_time = 0
        self.prev_pos = (0, 0)
        self.velocity = 0

    def calculate(self, pos, time):
        """
        Calculate the velocity of the eye movement
        :param pos:  the position of the eye
        :param time: the delta time of the position
        :return: the velocity
        """
        if pos is None or pos[0] is None or pos[1] is None:
            if self.velocity is not None:
                return self.velocity
            return None
        if self.prev_pos == (0, 0) or self.prev_pos is None:
            self.prev_time = time
            self.prev_pos = pos
            return None

        dt = time - self.prev_time
        dx = pos[0] - self.prev_pos[0]
        dy = pos[1] - self.prev_pos[1]
        self.velocity = np.sqrt(dx ** 2 + dy ** 2) / dt
        self.prev_time = time
        self.prev_pos = pos
        return self.velocity

    def calculate_video(self, positions):
        """
        Calculate the velocity of the eye movement in a video
        :param positions: the positions of the eye
        :return: the velocities
        """
        buffer = []
        self.prev_time = 0
        self.prev_pos = (0, 0)
        self.velocity = None
        for i, pos in enumerate(positions):
            pos = (pos[0], pos[1])

            time = self.delta_time*i
            vel = self.calculate(pos, time)
            if vel is not None:
                buffer.append(vel)
        if len(buffer) == 0:
            buffer.append(None)
        return buffer

    def calculate_batch(self, bached_positions):
        """
        Calculate the velocity of the eye movement in a batch of videos
        :param bached_positions:
        :return: the velocities
        """
        return [self.calculate_video(positions) for positions in bached_positions]


class IvvtClassifier:
    """
    Classifies the velocity of the eye movement into 3 categories
    0 - fixation
    1 - sacade
    2 - smoothper
    """
    def __init__(self, sacadeThreshold=0.5, smoothperThreshold=0.2):
        self.sacadeThreshold = sacadeThreshold
        self.smoothperThreshold = smoothperThreshold

    def classify(self, velocity):
        """
        Classify the velocity into 3 categories: fixation, sacade, smoothper
        0 - fixation
        1 - sacade
        2 - smoothper
        None - could not find velocites in time frame

        Classifies based on number of occurrences pr time frame

        :param velocity: array of velocities in a time frame
        :return: int or None
        """
        buffer = []
        for vel in velocity:
            if vel is None:
                if len(buffer) > 0:
                    buffer.append(buffer[-1])
            elif vel > self.sacadeThreshold:
                buffer.append(1)
            elif vel < self.smoothperThreshold:
                buffer.append(0)
            else:
                buffer.append(2)

        if len(buffer) == 0:
            return 4

        fixationProb = buffer.count(0) / len(buffer)
        sacadeProb = buffer.count(1) / len(buffer)
        smoothperProb = buffer.count(2) / len(buffer)

        if fixationProb > sacadeProb and fixationProb > smoothperProb:
            return 0
        elif sacadeProb > fixationProb and sacadeProb > smoothperProb:
            return 1
        else:
            return 2

    def classify_batch(self, velocities):
        """
        Classify the velocities of a batch of videos
        :param velocities:  the velocities
        :return:  the classifications
        """
        return [self.classify(vel) for vel in velocities]


class IvvtHelper:
    """
    Helper class for the IVVt algorithm
    combines the velocity calculator, the ivvt classifier and the pupile algorithm
    """
    def __init__(self, delta_time=6*1000/200, sacadeThreshold=0.5, smoothperThreshold=0.2):
        """
        Setting threshold values and params for the IVVt algorithm
        :param delta_time: the delta time between each frame in ms
        :param sacadeThreshold: the threshold for sacade
        :param smoothperThreshold: the threshold for smoothper
        """
        self.delta_time = delta_time
        self.vel_calculator = VelCalculator(self.delta_time)
        self.ivvt_classifier = IvvtClassifier(sacadeThreshold=sacadeThreshold, smoothperThreshold=smoothperThreshold)
        self.pupil_alg = PupileAlg()

    def classify_bach(self, positions):
        """
        Classify a batch of videos
        :param positions: the positions of the eye
        :return: the classifications
        """
        pos = self.pupil_alg.get_pupile_from_batch(positions)
        velocities = self.vel_calculator.calculate_batch(pos)
        return self.ivvt_classifier.classify_batch(velocities)

    def classify_video(self, positions):
        """
        Classify a video
        :param positions:  the positions of the eye
        :return:  the classifications
        """
        pos = self.pupil_alg.get_pupile_from_vid(positions)
        velocities = self.vel_calculator.calculate_video(pos)
        return self.ivvt_classifier.classify(velocities)

    def from_loader(self, loader):
        """
        Classify a loader
        :param loader:  the loader
        :return:  the classifications
        """
        return [c for innerlist in [self.classify_bach(bach) for bach in
                                    loader] for c in innerlist]


def test():
    """
    testing the IVVt algorithm
    """
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


    ivvt_helper = IvvtHelper()
    classified_loader = ivvt_helper.from_loader(train_loader)
    classified = [c for innerlist in [ivvt_helper.classify_bach(bach) for bach in
                  train_loader] for c in innerlist]
    assert classified_loader == classified

    print(classified)


if __name__ == '__main__':
    test()