import datetime
import math
import os
import pathlib
from functools import partial
import warnings

import pandas as pd
import torch.multiprocessing as mp
from joblib import Memory
from num2words import num2words
import numpy as np
from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config
from utils import seed_everything
import datasets

from main_simple_lib import *


im = load_image("/media/kristen/easystore/May2022_backups/2drill/front_camera_frames/frame002330.png")
query = "Is there rebar in this image?"
show_single_image(im)
code = get_code(query)

sudo apt-get install -y --no-install-recommends libnvinfer7=7.0.1-1+cuda11.6 \
    libnvinfer-dev=7.0.1-1+cuda11.6 \
    libnvinfer-plugin6=7.0.1-1+cuda11.6