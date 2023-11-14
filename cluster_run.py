from data_col import *
from process import Process
from classify import *
from data_wrapper import ClassifierData, MultiClassifierData
import util

import sys
key = int(sys.argv[1].strip())


def run_exp(cal_d, tes_d):
    for i in tes_d:
        clf = EmpiricMultiClassifier([BasicClassifier(),EntropyClassifier(), GeoClassifier(aggregate_interval=None), MarginClassifier(), L2Classifier()])
        td = MultiClassifierData(i.get_file_name(), classifier=clf.get_clfs(), name=i.get_name())
        cd = MultiClassifierData(cal_d.get_file_name(), classifier=clf.get_clfs(), name=cal_d.get_name())
        p = Process(cd, td, clf)
        p.process()


if key == 1:
    calibration_data = ResCal[0]
    test_data = ResN[1:2]
    run_exp(calibration_data, test_data)

elif key == 2:
    calibration_data = ResCal[0]
    test_data = ResN[2:3]
    run_exp(calibration_data, test_data)

elif key == 3:
    calibration_data = ResCal[0]
    test_data = ResN[3:5]
    run_exp(calibration_data, test_data)

elif key == 4:
    calibration_data = ResCal[0]
    test_data = ResN[5:]
    run_exp(calibration_data, test_data)

elif key == 5:
    calibration_data = SquCal[0]
    test_data = SquN[1:2]
    run_exp(calibration_data, test_data)

elif key == 6:
    calibration_data = SquCal[0]
    test_data = SquN[2:3]
    run_exp(calibration_data, test_data)

elif key == 7:
    calibration_data = SquCal[0]
    test_data = SquN[3:5]
    run_exp(calibration_data, test_data)

elif key == 8:
    calibration_data = SquCal[0]
    test_data = SquN[5:]
    run_exp(calibration_data, test_data)

elif key == 9:
    calibration_data = ResCal[0]
    test_data = ResD[0:1]
    run_exp(calibration_data, test_data)

elif key == 10:
    calibration_data = ResCal[0]
    test_data = ResD[1:2]
    run_exp(calibration_data, test_data)

elif key == 11:
    calibration_data = ResCal[0]
    test_data = ResD[2:5]
    run_exp(calibration_data, test_data)

elif key == 12:
    calibration_data = ResCal[0]
    test_data = ResD[5:]
    run_exp(calibration_data, test_data)

elif key == 13:
    calibration_data = SquCal[0]
    test_data = SquD[0:1]
    run_exp(calibration_data, test_data)

elif key == 14:
    calibration_data = SquCal[0]
    test_data = SquD[1:2]
    run_exp(calibration_data, test_data)

elif key == 15:
    calibration_data = SquCal[0]
    test_data = SquD[2:5]
    run_exp(calibration_data, test_data)

elif key == 16:
    calibration_data = SquCal[0]
    test_data = SquD[5:]
    run_exp(calibration_data, test_data)