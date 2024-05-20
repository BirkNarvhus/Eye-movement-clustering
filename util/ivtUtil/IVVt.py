import numpy as np

import sys
sys.path.append('C:\\Users\\birkn\\Documents\\bachlor\\eye-movement-classification')

from util.ivtUtil.EyePosition import PupileAlg


class VelCalculator:
    def __init__(self, delt_time):
        """
        :param delt_time: delta time between each frame in ms
        """
        self.delta_time = delt_time
        self.prev_time = 0
        self.prev_pos = (0, 0)
        self.velocity = 0

    def calculate(self, pos, time):
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
        return [self.calculate_video(positions) for positions in bached_positions]


class IvvtClassifier:
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
        return [self.classify(vel) for vel in velocities]


class IvvtHelper:
    def __init__(self, delta_time=6*1000/200, sacadeThreshold=0.5, smoothperThreshold=0.2):
        self.delta_time = delta_time
        self.vel_calculator = VelCalculator(self.delta_time)
        self.ivvt_classifier = IvvtClassifier(sacadeThreshold=sacadeThreshold, smoothperThreshold=smoothperThreshold)
        self.pupil_alg = PupileAlg()
    def classify_bach(self, positions):
        pos = self.pupil_alg.get_pupile_from_batch(positions)
        velocities = self.vel_calculator.calculate_batch(pos)
        return self.ivvt_classifier.classify_batch(velocities)

    def classify_video(self, positions):
        pos = self.pupil_alg.get_pupile_from_vid(positions)
        velocities = self.vel_calculator.calculate_video(pos)
        return self.ivvt_classifier.classify(velocities)

    def from_loader(self, loader):
        return [c for innerlist in [self.classify_bach(bach) for bach in
                                    loader] for c in innerlist]
def test():
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