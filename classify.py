import numpy as np
from scipy import stats
import util
from statsmodels.stats.multitest import multipletests
from scipy.stats import ksone
from sklearn import metrics
from process import Process
from data_wrapper import ConfigData
import pandas as pd
import copy

class BasicClassifier:

    TVD = {(1e-05, 1.0): 0.999995, (1e-05, 3.0): 0.982894897461, (1e-05, 5.0): 0.912933349609,
           (1e-05, 10.0): 0.717041015625, (1e-05, 30.0): 0.436859130859, (1e-05, 50.0): 0.342071533203,
           (1e-05, 100.0): 0.243988037109, (1e-05, 300.0): 0.141807556152, (1e-05, 500.0): 0.110023498535,
           (1e-05, 1000.0): 0.077911376953, (1e-05, 3000.0): 0.045040130615, (1e-05, 5000.0): 0.034900665283,
           (1e-05, 10000.0): 0.024686813354, (5e-05, 1.0): 0.999975, (5e-05, 3.0): 0.970764160156,
           (5e-05, 5.0): 0.8798828125, (5e-05, 10.0): 0.674682617188, (5e-05, 30.0): 0.407958984375,
           (5e-05, 50.0): 0.319091796875, (5e-05, 100.0): 0.227416992188, (5e-05, 300.0): 0.132125854492,
           (5e-05, 500.0): 0.102508544922, (5e-05, 1000.0): 0.07258605957, (5e-05, 3000.0): 0.041961669922,
           (5e-05, 5000.0): 0.032516479492, (5e-05, 10000.0): 0.023000717163, (0.0001, 1.0): 0.99995,
           (0.0001, 3.0): 0.963165283203, (0.0001, 5.0): 0.861999511719, (0.0001, 10.0): 0.65478515625,
           (0.0001, 30.0): 0.394775390625, (0.0001, 50.0): 0.308624267578, (0.0001, 100.0): 0.219879150391,
           (0.0001, 300.0): 0.127731323242, (0.0001, 500.0): 0.099098205566, (0.0001, 1000.0): 0.070171356201,
           (0.0001, 3000.0): 0.040565490723, (0.0001, 5000.0): 0.031433105469, (0.0001, 10000.0): 0.022235870361,
           (0.0005, 1.0): 0.99995, (0.0005, 3.0): 0.93701171875, (0.0005, 5.0): 0.809631347656,
           (0.0005, 10.0): 0.604309082031, (0.0005, 30.0): 0.361938476562, (0.0005, 50.0): 0.282653808594,
           (0.0005, 100.0): 0.201263427734, (0.0005, 300.0): 0.116882324219, (0.0005, 500.0): 0.090675354004,
           (0.0005, 1000.0): 0.064208984375, (0.0005, 3000.0): 0.037120819092, (0.0005, 5000.0): 0.028762817383,
           (0.0005, 10000.0): 0.020347595215,
           (0.001, 1.0): 0.9995, (0.001, 3.0): 0.920654296875, (0.001, 5.0): 0.781372070312,
           (0.001, 10.0): 0.580444335938, (0.001, 30.0): 0.346740722656, (0.001, 50.0): 0.270690917969,
           (0.001, 100.0): 0.192687988281, (0.001, 300.0): 0.111877441406, (0.001, 500.0): 0.086791992188,
           (0.001, 1000.0): 0.061462402344, (0.001, 3000.0): 0.035533905029, (0.001, 5000.0): 0.027534484863,
           (0.001, 10000.0): 0.019477844238,
           (0.005, 1.0): 0.9975, (0.005, 3.0): 0.8642578125, (0.005, 5.0): 0.705444335938,
           (0.005, 10.0): 0.518737792969, (0.005, 30.0): 0.308166503906, (0.005, 50.0): 0.240386962891,
           (0.005, 100.0): 0.171051025391, (0.005, 300.0): 0.099304199219, (0.005, 500.0): 0.077041625977,
           (0.005, 1000.0): 0.054557800293, (0.005, 3000.0): 0.031539916992, (0.005, 5000.0): 0.024444580078,
           (0.005, 10000.0): 0.017292022705,
           (0.01, 1.0): 0.995, (0.01, 3.0): 0.828979492188, (0.01, 5.0): 0.668579101562,
           (0.01, 10.0): 0.488891601562, (0.01, 30.0): 0.289855957031, (0.01, 50.0): 0.226043701172,
           (0.01, 100.0): 0.160797119141, (0.01, 300.0): 0.093353271484, (0.01, 500.0): 0.07243347168,
           (0.01, 1000.0): 0.051292419434, (0.01, 3000.0): 0.02965927124, (0.01, 5000.0): 0.022983551025,
           (0.01, 10000.0): 0.016258239746,
           (0.05, 1.0): 0.975, (0.05, 3.0): 0.70751953125, (0.05, 5.0): 0.563232421875,
           (0.05, 10.0): 0.409240722656, (0.05, 30.0): 0.24169921875, (0.05, 50.0): 0.188415527344,
           (0.05, 100.0): 0.134033203125, (0.05, 300.0): 0.077835083008, (0.05, 500.0): 0.060394287109,
           (0.05, 1000.0): 0.042778015137, (0.05, 3000.0): 0.024742126465, (0.05, 5000.0): 0.019172668457,
           (0.05, 10000.0): 0.013565063477,
           (0.1, 1.0): 0.95, (0.1, 3.0): 0.635986328125, (0.1, 5.0): 0.509521484375,
           (0.1, 10.0): 0.36865234375, (0.1, 30.0): 0.217529296875, (0.1, 50.0): 0.169616699219,
           (0.1, 100.0): 0.120666503906, (0.1, 300.0): 0.070098876953, (0.1, 500.0): 0.054397583008,
           (0.1, 1000.0): 0.038528442383, (0.1, 3000.0): 0.022285461426, (0.1, 5000.0): 0.017272949219,
           (0.1, 10000.0): 0.012222290039,
           (0.5, 1.0): 0.75, (0.5, 3.0): 0.4345703125, (0.5, 5.0): 0.341796875, (0.5, 10.0): 0.2465,
           (0.5, 30.0): 0.145874023438, (0.5, 50.0): 0.113891601562, (0.5, 100.0): 0.081176757812,
           (0.5, 300.0): 0.047241210938, (0.5, 500.0): 0.036682128906, (0.5, 1000.0): 0.026000976562,
           (0.5, 3000.0): 0.01505279541, (0.5, 5000.0): 0.011672973633, (0.5, 10000.0): 0.00825881958}

    name = "ksconf"

    def __init__(self, calibration_data=None, threshold_value_dic=None, aggregate_interval=None, cal_min=0, cal_max=1):
        self._aggregate_interval = aggregate_interval
        self.cal_min = cal_min
        self.cal_max = cal_max
        self._calibration_data = None
        # if calibration_data is not None:
        #     self.calibrate(calibration_data)
        if threshold_value_dic is not None:
            self._threshold_value_dic = threshold_value_dic
        else:
            self._threshold_value_dic = self.TVD

    def get_cdf(self, data):
        return {self.get_name(): [[self.cdf(self._aggregate_data(data, True)), self.ycdf(len(data))],
                                  [self.cdf(self._calibration_data), self.ycdf(len(data))]]}

    def get_data(self, calibration_data=None, test_data=None, calibration_data_flag=True, test_data_flag=True,
                 calibration_name="calibration", test_name="test", sort=True, as_cdf=False):
        cal_data = self.get_calibration_data(calibration_data, sort, False, as_cdf=as_cdf) if calibration_data is not None and calibration_data_flag else None
        tes_data = self.get_test_data(test_data, sort, False, as_cdf=as_cdf) if test_data is not None and test_data_flag else None
        data = {self.get_name(): {calibration_name: cal_data, test_name: tes_data}}
        return data

    def get_calibration_data(self, calibration_data, sort=True, as_dic=True, as_cdf=False):
        if as_cdf:
            cal_data = [self.cdf(self._aggregate_data(calibration_data, sort)), self.ycdf(len(calibration_data))]
        else:
            cal_data = self._aggregate_data(calibration_data, sort)
        return {self.get_name(): cal_data} if as_dic else cal_data

    def get_test_data(self, test_data, sort=True, as_dic=True, as_cdf=False):
        if as_cdf:
            tes_data = [self.cdf(self._aggregate_data(test_data, sort)), self.ycdf(len(test_data))]
        else:
            tes_data = self._aggregate_data(test_data, sort)
        return {self.get_name(): tes_data} if as_dic else tes_data

    def cdf(self, p):
        return np.interp(p, self._calibration_data, self.ycdf(len(self._calibration_data)))

    def calibrate(self, calibration_data, sort=True, *args, **kwargs):
        calibration_data = calibration_data.get(True)
        data = self._aggregate_data(calibration_data)
        data = np.array([self.cal_min] + list(data) + [self.cal_max])
        self._calibration_data = np.sort(data)

    def classify(self, data, alpha, scipy=True):
        data = self._aggregate_data(data)
        data = self.cdf(data)
        val = stats.kstest(data, lambda x: x).statistic
        if scipy:
            return val > self.ks_critical_value(alpha, len(data))
        else:
            return val > self._threshold_value_dic[alpha, len(data)]

    def test_results(self, data):
        data = self._aggregate_data(data)
        data = self.cdf(data)
        return stats.kstest(data, lambda x: x)

    def get_name(self):
        add = "" if self._aggregate_interval is None else "_{0}_{1}".format(self._aggregate_interval[0],
                                                                            self._aggregate_interval[1])
        return self.name + add

    def aggregate_data(self, data, sort=False):
        return self._aggregate_data(data, sort)

    def _aggregate_data(self, data, sort=False):
        if data.ndim > 1:
            if self._aggregate_interval is None:
                data = self._aggregate_foo(data)
            else:
                data = self._aggregate_foo(data[self._aggregate_interval[0]:, :self._aggregate_interval[1] + 1])
        return np.sort(data) if sort else data

    @staticmethod
    def _aggregate_foo(data):
        return np.max(data, axis=1)

    @staticmethod
    def ycdf(n):
        n = n-2
        return [i/(n+1) for i in range(n+2)]

    @staticmethod
    def ks_critical_value(alpha, batch_size):
        return ksone.ppf(1 - alpha / 2, batch_size)


class Basic2Classifier(BasicClassifier):
    name = "p2"

    @staticmethod
    def _aggregate_foo(data):
        return 2*data[:, 1]


class Basic3Classifier(BasicClassifier):
    name = "p3"

    @staticmethod
    def _aggregate_foo(data):
        return 3*data[:, 2]


class L2Classifier(BasicClassifier):

    name = "l2"

    @staticmethod
    def _aggregate_foo(data):
        return np.linalg.norm(data, axis=1)


class EntropyClassifier(BasicClassifier):
    name = "entropy"

    @staticmethod
    def _aggregate_foo(data):
        return stats.entropy(data, axis=1) / np.log(data.shape[1])


class MarginClassifier(BasicClassifier):
    name = "margin"

    def __init__(self, calibration_data=None, threshold_value_dic=None, aggregate_interval=(0, 1)):
        super().__init__(calibration_data, threshold_value_dic, aggregate_interval)

    @staticmethod
    def _aggregate_foo(data):
        return data[:, 0] - data[:, 1]


class GeoClassifier(BasicClassifier):
    name = "geo"

    def __init__(self, calibration_data=None, threshold_value_dic=None, aggregate_interval=(0, 1)):
        super().__init__(calibration_data, threshold_value_dic, aggregate_interval)

    @staticmethod
    def _aggregate_foo(data):
        return stats.mstats.gmean(data, axis=1) / (1/data.shape[1])




class ConfSAvrClassifier(BasicClassifier):
    name = "confsavr"

    @staticmethod
    def _aggregate_foo(data):
        return np.average((data-np.roll(data, -1, axis=1))[:, :-1], axis=1, weights=data[:, :-1])


class LogL2Classifier(BasicClassifier):
    name = "logl2"

    @staticmethod
    def _aggregate_foo(data):
        return np.linalg.norm(np.log((data+1**(-6)) * (1/(1+(1**(-6)*data.shape[1])))), axis=1)

#
# class MSEClassifier(BasicClassifier):
#     name = "mse"
#
#     def __init__(self, calibration_data=None, threshold_value_dic=None, aggregate_interval=None):
#         super().__init__(calibration_data, threshold_value_dic, aggregate_interval)
#         self.avr_dist = None
#
#     def calibrate(self, calibration_data, sort=True):
#         if calibration_data is not None:
#             if self._aggregate_interval is None:
#                 self.avr_dist = np.mean(calibration_data, axis=0)
#             else:
#                 self.avr_dist = np.average(calibration_data[self._aggregate_interval[0]:, :self._aggregate_interval[1] + 1], axis=0)
#         super().calibrate(calibration_data, sort)
#
#     def _aggregate_foo(self, data):
#         return np.array([metrics.mean_squared_error(self.avr_dist, d) for d in data])

# ##############################################################################


class MultiClassifier:
    _name = "multi"

    def __init__(self, classifier, calibration_data=None):
        self._classifier = classifier
        # self.calibrate(calibration_data)
        # self.adjust_fpr = adjust_fpr
        # self.fpr_padding = 1

    # def calibrate_fpr(self, calibration_data):
    #     name = calibration_data.get_name() + "_" + self.get_name()
    #     data = ConfigData(name + ".csv", name)
    #     if data.exists():
    #         self.fpr_padding = self.compute_fpr_padding(data.get())
    #     else:
    #         self.adjust_fpr = False
    #         p = Process(calibration_data, calibration_data.copy(), self, result_data=data)
    #         p.process()
    #         self.fpr_padding = self.compute_fpr_padding(data.get())

    def get_clfs(self):
        return self._classifier

    def get_cdf(self):
        return {c.get_name(): c.get_cdf()[c.get_name()] for c in self._classifier}

    def get_data(self, data=None, calibration_data=True, test_data=True, sort=True):
        if data is None:
            return {c.get_name(): c.get_data(data, calibration_data, test_data, sort)[c.get_name()] for c in
                    self._classifier}
        else:
            return {c.get_name(): c.get_data(data.get_data(c), calibration_data, test_data, sort)[c.get_name()] for c in
                    self._classifier}

    def calibrate(self, calibration_data, *args, **kwargs):
        # if self.adjust_fpr:
        #     self.calibrate_fpr(calibration_data)
        for c in self._classifier:
            c.calibrate(calibration_data=calibration_data.get_data(c))

    def classify(self, data, alpha):
        results = [c.classify(data[c.get_name()], self.adjust_alpha(alpha, len(data[c.get_name()]))) for i, c in enumerate(self._classifier)]
        return self._aggregate_results(results)

    def get_name(self):
        return "_".join([self._name] + [c.get_name() for c in self._classifier])

    def cdf(self, p, clf_nr=None):
        if clf_nr is None:
            return [c.cdf(p) for c in self._classifier]
        else:
            return self._classifier[clf_nr].cdf(p)

    def adjust_alpha(self, alpha, bs):
        # print(alpha * (1 / len(self._classifier)))
        return alpha * (1 / len(self._classifier))
    #
    # @staticmethod
    # def compute_fpr_padding(data):
    #     df = pd.DataFrame(data, columns=["bs", "alpha", "iter", "result"])
    #     df["result"] = df["result"]/df["iter"]
    #     df["diff"] = df["alpha"]/df["result"]
    #     res = {}
    #     for i in df.groupby("alpha"):
    #         res[i[0]] = i[1]["diff"].mean()
    #     return res

    @staticmethod
    def _adjust_results(results, alpha):
        return multipletests([i.pvalue for i in results], alpha, "bonferroni")[0]

    @staticmethod
    def _aggregate_results(results):
        return sum(results) > 0


class HolmMultiClassifier(MultiClassifier):
    _name = "holm"

    @staticmethod
    def _adjust_results(results, alpha):
        return multipletests([i.pvalue for i in results], alpha, "holm")[0]


class EmpiricMultiClassifier(MultiClassifier):
    _name = "emulti"
    alphas = [0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.5]
    batches = [10, 30, 50, 100, 300, 500, 1000, 3000, 5000, 10000]
    iterations = 10000

    def __init__(self, classifier, calibration_data=None, alphas=None, batches=None, iterations=None, in_configuration=False):
        super().__init__(classifier, calibration_data)
        if alphas is not None:
            self.alphas = alphas
        if batches is not None:
            self.batches = batches
        if iterations is not None:
            self.iterations = iterations
        self.config = None
        self.in_configuration = in_configuration

    def calibrate(self, calibration_data, process=None, force=False,*args, **kwargs):
        if not self.in_configuration:
            if process is None:
                process = Process(calibration_data, calibration_data, self, result_data=ConfigData,
                                  alphas=self.alphas, batches=self.batches, iterations=self.iterations, *args, **kwargs)
            else:
                process = process.__class__(calibration_data, calibration_data, self, result_data=ConfigData,
                                            alphas=process.alphas, batches=process.batches, iterations=process.iterations,
                                            *args, **kwargs)
            conf = process.get_result_wrapper()
            if not conf.exists() or force:
                self.in_configuration = True
                print("#"*100)
                print("Start Calibration")
                print("#" * 100)
                process.process()
                self.in_configuration = False
            self.compute_config(conf.get())
        for c in self._classifier:
            c.calibrate(calibration_data=calibration_data.get_data(c))

    def compute_config(self, conf):
        df = pd.DataFrame(conf, columns=["bs", "alpha", "iter", "result"])
        self.config = {i[0]: [[0]+list(i[1]["alpha"].values), [0]+list(i[1]["result"]/i[1]["iter"].values)] for i in df.groupby("bs")}

    def empiric_fpr(self, alpha, bs):
        res = np.interp(alpha, self.config[bs][1], self.config[bs][0])
        return res

    def adjust_alpha(self, alpha, bs):
        if self.in_configuration:
            return alpha
        else:
            return self.empiric_fpr(alpha, bs)

    def configure(self, data, *args, **kwargs):
        p = Process(data, data, self, result_data=ConfigData, alphas=self.alphas, *args, **kwargs)
        p.process()