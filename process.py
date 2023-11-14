import util
import numpy as np
from data_wrapper import *


class Process:

    # alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    # batches = [1, 3, 5, 10, 30, 50, 100, 300, 500, 1000, 3000, 5000, 10000]
    # alphas = [ 0.1, 0.5]
    # batches = [5,10,30]
    alphas = [0.005, 0.01, 0.05, 0.1]
    batches = [10, 30, 50, 100, 300, 500, 1000, 3000, 5000, 10000]
    name = "basic"
    res_col = ["bs", "alpha", "iter", "result"]

    def __init__(self, calibration_data, test_data, classifier,
                 name=None, batches=None, alphas=None, persist=True, iterations=10000,
                 verbose=True, limit=True, result_data=None, progress_bar=False):
        self.limit = limit
        if alphas is not None:
            self.alphas = alphas
        if batches is not None:
            self.batches = batches
        if name is not None:
            self.name = name
        self.iterations = iterations

        self.classifier = classifier
        self.calibration_data = calibration_data
        self.test_data = test_data
        self.results = ResultData(self.get_name()) if result_data is None else result_data(self.get_name())
        self.persist = persist
        self.init = False
        self.verbose = verbose
        self.progress_bar = progress_bar

    def get_result_path(self):
        return self.results.get_path()

    def get_result_wrapper(self):
        return self.results

    def get_name(self):
        return "-".join([self.name, self.classifier.get_name(),
                         self.calibration_data.get_name(), self.test_data.get_name()])

    def get_test_data(self):
        return self.test_data.get(self.persist)

    def get_calibration_data(self):
        return self.calibration_data.get(self.persist)

    def get_avrg_distr(self, calibration_data=True, test_data=True):
        return {self.classifier.get_name(): {
            self.calibration_data.get_name(): np.mean(self.get_calibration_data(), axis=0) if calibration_data else None,
            self.test_data.get_name(): np.mean(self.get_test_data(), axis=0)} if test_data else None, }

    def get_clf_data(self, calibration_data=True, test_data=True, sort=True, true_names=False, as_cdf=False):
        if as_cdf:
            self.classifier.calibrate(self.get_calibration_data())
        if true_names:
            return self.classifier.get_data(calibration_data=self.get_calibration_data(), test_data=self.get_test_data(),
                                            calibration_data_flag=calibration_data, test_data_flag=test_data, sort=sort,
                                            calibration_name=self.calibration_data.get_name(),
                                            test_name=self.test_data.get_name(),
                                            as_cdf=as_cdf)
        else:
            return self.classifier.get_data(calibration_data=self.get_calibration_data(), test_data=self.get_test_data(),
                                            calibration_data_flag=calibration_data, test_data_flag=test_data, sort=sort,
                                            as_cdf=as_cdf)

    def get_cdf(self):
        return self.classifier.get_cdf(self.get_test_data())

    def get_results(self):
        return self.results.get(self.persist)

    def initialise(self, verbose=True):
        if not self.init:
            self.classifier.calibrate(self.calibration_data, process=self)
            self.test_data.get(self.persist)
            self.init = True
        if self.verbose:
            print("")
            print("#"*100)
            print(self.get_name(), " => Process Initialised")
            print("#" * 100)

    def process(self):
        results = []
        self.initialise()
        for alpha in self.alphas:
            reached = False
            for bs in self.batches:
                if reached:
                    result = self.iterations
                else:
                    result = self._process(bs, alpha)
                    if self.limit and result == self.iterations:
                        reached = True
                results.append([bs, alpha, self.iterations, result])
                if self.verbose:
                    print("-"*100)
                    print("Alpha {1} | BS: {0} | Result: {2}".format(bs, alpha, result))
                    print("-" * 100)
        self.results.set(results)
        self.results.write()

    def progress(self, val, i):
        if self.progress_bar:
            util.printProgressBar(i+1, self.iterations)
        return val

    def _process(self, batch_size, alpha):
        results = [self.progress(self.classifier.classify(self.test_data.sample(batch_size), alpha), i)
                   for i in range(self.iterations)]
        print("")
        return self.aggregate_results(results)

    @staticmethod
    def aggregate_results(results):
        pos = np.sum(results)
        return pos


class ClfProcess:

    def __init__(self, data, classifier):
        self.data = data
        self.classifier = classifier

    def process(self):
        cd = ClassifierData(self.data.get_file_name(), self.classifier)
        cd.set(self.classifier.aggregate_data(self.data.get()))
        cd.write()



