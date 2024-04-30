import numpy as np

class vel_calculator:
    def __init__(self):
        self.prev_time = 0
        self.prev_pos = (0, 0)
        self.velocity = 0

    def calculate(self, pos, time):
        if self.prev_time == 0:
            self.prev_time = time
            self.prev_pos = pos
            return 0

        dt = time - self.prev_time
        dx = pos[0] - self.prev_pos[0]
        dy = pos[1] - self.prev_pos[1]
        self.velocity = np.sqrt(dx ** 2 + dy ** 2) / dt
        self.prev_time = time
        self.prev_pos = pos
        return self.velocity

