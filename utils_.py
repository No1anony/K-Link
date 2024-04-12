
import os
import csv
import random
import numpy as np
from numpy.core.fromnumeric import transpose
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import interpolate
import math

import torch.utils.data as data
import logging
# from config import config
import torch







def sensor_names_RUL(index = False):
    full_abbre = {
        'LPC': 'low pressure compressor',
        'HPC': 'high pressure compressor',
        'LPT': 'low pressure turbine',
        'HPT': 'high pressure turbine'
    }
    if index:
        sensors_name_full = [str(i) for i in range(21)]

    else:

        sensors_name_full = ['total temperature at fan inlet', 'total temperature at low pressure compressor outlet',
                             'total temperature at high pressure compressor outlet',
                             'total temperature at low pressure turbine outlet',
                             'total dynamic pressure at fan inlet', 'total pressure in bypass-duct',
                             'total pressure at high pressure compressor outlet', 'the physical fan speed',
                             'physical core speed', 'engine pressure ratio',
                             'static pressure at high pressure compressor outlet', 'ratio of fuel flow to static pressure (Ps30)',
                             'corrected fan speed', 'corrected core speed', 'bypass Ratio', 'burner fuel-air ratio',
                             'bleed Enthalpy', 'demanded fan speed',
                             'demanded corrected fan speed', 'high pressure turbine coolant bleed',
                             'low pressure turbine coolant bleed']

    removed_index = [1, 5, 6, 10, 16, 18, 19]  ### index number

    # print(sensors_name_full)
    sensor4use = []
    for i, v in enumerate(sensors_name_full):
        if i + 1 not in removed_index:
            sensor4use.append(v)

    return sensor4use

def sensor_names_HAR(index = False):

    if index:
        sensors_name_full = [str(i) for i in range(9)]

    else:

        sensors_name_full = ['The x axis of the body acceleration signal obtained by subtracting the gravity from the total acceleration',
                             'The y axis of the body acceleration signal obtained by subtracting the gravity from the total acceleration',
                             'The z axis of the body acceleration signal obtained by subtracting the gravity from the total acceleration',
                             'The x axis of the angular velocity vector measured by the gyroscope with units radians/second',
                             'The y axis of the angular velocity vector measured by the gyroscope with units radians/second',
                             'The z axis of the angular velocity vector measured by the gyroscope with units radians/second',
                             'The x axis of the acceleration signal from the smartphone accelerometer X axis in standard gravity units',
                             'The y axis of the acceleration signal from the smartphone accelerometer X axis in standard gravity units',
                             'The z axis of the acceleration signal from the smartphone accelerometer X axis in standard gravity units',
                             ]



    return sensors_name_full

def sensor_names_SSC(index = False):

    if index:
        sensors_name_full = [str(i) for i in range(10)]

    else:

        sensors_name_full = ['C3-A2, the recording electrode (C3) is placed over the left central region of the scalp, and the reference electrode (A2) is placed on the right earlobe',
                             'C4-A1, the recording electrode (C4) is placed over the right central region of the scalp, and the reference electrode (A1) is placed on the left earlobe',
                             'F3-A2, the recording electrode (F3) is placed over the left frontal region of the scalp, and the reference electrode (A2) is placed on the right earlobe',
                             'F4-A1, the recording electrode (F4) is placed over the right frontal region of the scalp, and the reference electrode (A1) is placed on the left earlobe',
                             'O1-A2, the recording electrode (O1) is placed over the left occipital region of the scalp, and the reference electrode (A2) is placed on the right earlobe',
                             'O2-A1, the recording electrode (O2) is placed over the right occipital region of the scalp, and the reference electrode (A1) is placed on the left earlobe',
                             'LOC-A2, the recording electrode is placed at the left outer canthus (lateral corner of the left eye) to detect left eyes movements, and the reference electrode is placed on the right earlobe',
                             'ROC-A1, the recording electrode is placed at the right outer canthus (lateral corner of the right eye) to detect right eyes movements, and the reference electrode is placed on the left earlobe',
                             'X1, Chin EMG, the recording electrode is placed between the chin and the lower lip',
                             'X2, Electrocardiography (ECG or EKG), the recording electrode is placed on the right arm (right wrist or forearm)'
                             ]



    return sensors_name_full

