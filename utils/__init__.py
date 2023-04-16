import random

from gym.wrappers import Monitor
from pyvirtualdisplay import Display
import torch
import numpy as np

display = None


def record_videos(env, path="videos"):
    global display
    if display is None:
        display = Display(visible=0, size=(1400, 900))
        display.start()
    monitor = Monitor(env, path, force=True, video_callable=lambda episode: True)
    # Capture intermediate frames
    env.unwrapped.set_monitor(monitor)
    return monitor


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
