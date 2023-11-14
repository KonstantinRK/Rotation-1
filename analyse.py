import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
import util
from collections import OrderedDict
from process import Process
from matplotlib.colors import LogNorm
import re

class DataAnalysis:

    def __init__(self, processes, save_images=False, show=True):
        self.processes = OrderedDict([(i.get_name(), i) for i in processes])

        self.save_images = save_images
        self.show = show

    def _finalise_image(self, process_name, fig=None):
        if fig is None:
            if self.save_images:
                plt.savefig(fname=util.get_summary_path(process_name, ".jpg"), dpi=1000, format="jpg")
            if self.show:
                plt.show()
        else:
            if self.save_images:
                fig.savefig(fname=util.get_summary_path(process_name, ".jpg"), dpi=1000, format="jpg")
            if self.show:
                fig.show()

    def draw_cdf(self, process_names=None, one_calibration_curve=True, title=""):
        process_names = self.processes.keys() if process_names is None else process_names
        process_names = process_names if not isinstance(process_names, list) else [process_names]
        x = len(process_names) + 1 if one_calibration_curve else len(process_names) * 2
        col = sns.color_palette("hls", x)
        j = 0
        fig, axs = plt.subplots()
        for i, n in enumerate(process_names):
            if not one_calibration_curve or i == 0:
                data = self.processes[n].get_clf_data(calibration_data=True, test_data=True, true_names=True, as_cdf=True)
            else:
                data = self.processes[n].get_clf_data(calibration_data=False, test_data=True, true_names=True, as_cdf=True)
            for k, v in data.items():
                for h, d in v.items():
                    # print(i, n, k, h)
                    # print("-" * 100)
                    if d is not None:
                        sns.lineplot(x=d[0], y=d[1], color=col[j], ax=axs, label="-".join([k, h]))
                        j += 1
        axs.legend()
        axs.set_title(title)
        self._finalise_image("cdf-" + "-".join(process_names), fig)

        # self.processes[process_name].initialise()
        # data = self.processes[process_name].get_cdf()
        # keys = list(data.keys())
        # fig, axs = plt.subplots(len(keys), 1, squeeze=False)
        # for i in range(len(keys)):
        #     sns.lineplot(x=data[keys[i]][0][0], y=data[keys[i]][0][1], ax=axs[i, 0])
        #     sns.lineplot(x=data[keys[i]][1][0], y=data[keys[i]][1][1], ax=axs[i, 0])
        #     axs[i, 0].set_title(keys[i])
        # plt.show()

    def draw_hist(self, process_names=None, bins=100, one_calibration_curve=True, title=""):
        process_names = self.processes.keys() if process_names is None else process_names
        process_names = process_names if not isinstance(process_names, list) else [process_names]
        x = len(process_names)+1 if one_calibration_curve else len(process_names)*2
        col = sns.color_palette("hls", x)
        j = 0
        fig, axs = plt.subplots()
        for i, n in enumerate(process_names):
            if not one_calibration_curve or i == 0:
                data = self.processes[n].get_clf_data(calibration_data=True, test_data=True, true_names=True)
            else:
                data = self.processes[n].get_clf_data(calibration_data=False, test_data=True, true_names=True)
            for k, v in data.items():
                for h, d in v.items():
                    print(i, n, k, h, d)
                    print("-"*100)
                    if d is not None:
                        sns.histplot(x=d, bins=bins, stat="probability", color=col[j], ax=axs, label="-".join([k, h]))
                        j += 1
        axs.legend()
        axs.set_title(title)
        path = title if title != "" else "hist-" + "-".join(process_names)
        self._finalise_image(path, fig)

    def draw_avr_scores(self, process_names=None, one_calibration_curve=True, title="", cut_at=None):
        process_names = self.processes.keys() if process_names is None else process_names
        process_names = process_names if not isinstance(process_names, list) else [process_names]
        x = len(process_names)+1 if one_calibration_curve else len(process_names)*2
        col = sns.color_palette("hls", x)
        j = 0
        fig, axs = plt.subplots()
        for i, n in enumerate(process_names):
            if not one_calibration_curve or i == 0:
                data = self.processes[n].get_avrg_distr(calibration_data=True, test_data=True)
            else:
                data = self.processes[n].get_avrg_distr(calibration_data=False, test_data=True)
            for k, v in data.items():
                for h, d in v.items():
                    if d is not None:
                        print(i, n, k, h, d.shape)
                        print("-" * 100)
                        if cut_at is not None:
                            d = d[:cut_at]
                        sns.lineplot(y=d, x=list(range(d.shape[0])), color=col[j], ax=axs, label="-".join([k, h]))
                        j += 1
        # if cut_at is not None:
        #     axs.set_xlim(0, cut_at)
        axs.legend(fontsize="x-small")
        axs.set_title(title)
        plt.yscale("log")
        # plt.xscale("log")
        path = title if title != "" else "hist-" + "-".join(process_names)
        self._finalise_image(path, fig)

    # def compute_tv(self, bins=100):
    #     data = self.get_test_data()
    #     test_hist = np.array(np.histogram(data, bins)[0]/np.sum(data))
    #     cal_hist = np.array(np.histogram(self.experiment.cal_data, bins)[0]/np.sum(self.experiment.cal_data))
    #     index = chain.from_iterable(combinations(range(bins), r) for r in range(len(range(bins)) + 1))
    #     get_event = lambda x,y: 0 if len(y) == 0 else x[np.array(y)]
    #     return np.max([np.abs(np.sum(get_event(test_hist,i))-np.sum(get_event(cal_hist,i))) for i in index])
    #
    #


class Analysis(DataAnalysis):

    result_header = ["bs", "alpha", "iter"]

    def __init__(self, processes, validation_processes=None, save_images=False, show=True, load_results=True):
        super().__init__(processes, save_images, show)
        self.val_processes = None if validation_processes is None else OrderedDict(
            [(processes[i].get_name(), validation_processes[i]) for i in range(len(processes))])
        self.results = None
        self.val_results = None
        if load_results:
            self.results = self.compute_results(self.processes)
            if validation_processes is not None:
                self.val_results = self.compute_results(self.val_processes)

    def get_results(self, persist=True, flat=False):
        if self.results is None:
            results = self.compute_results(flat)
            if persist:
                self.results = results
            return results
        else:
            return self.results

    def compute_results(self, data, flat=False):
        if flat:
            return self.compute_flat_results(data)
        else:
            return self.compute_table_results(data)

    def compute_flat_results(self, data):
        df = None
        for k, v in data.items():
            if df is None:
                df = pd.DataFrame(v.get_results(), columns=self.result_header + [k])
                df["result"] = df[k]/df["iter"]
                df["process"] = k
            else:
                df2 = pd.DataFrame(v.get_results(), columns=self.result_header + [k])
                df["result"] = df[k] / df["iter"]
                df["process"] = k
                df = pd.merge(df, df2, how='left', left_on=self.result_header, right_on=self.result_header)
        return df

    def compute_table_results(self, data):
        df = None
        for k, v in data.items():
            if df is None:
                df = pd.DataFrame(v.get_results(), columns=self.result_header + [k])
                df[k] = df[k]/df["iter"]
            else:
                df2 = pd.DataFrame(v.get_results(), columns=self.result_header + [k])
                df2[k] = df2[k] / df2["iter"]
                df = pd.merge(df, df2, how='left', left_on=self.result_header, right_on=self.result_header)
        return df

    @staticmethod
    def default_process_map(val):
        s = val.split("-")[1]
        if s == "ksconf":
            s = "max"
        if s == "margin_0_1":
            s = "margin"
        return s

    def draw_roc(self, fix_label, value, from_to=None, log=True, process_map_foo=None, point=False):
        if fix_label == "alpha":
            var_label = "bs"
        else:
            var_label = "alpha"
        data = self.flatten_results()
        data = data[data[fix_label] == value]
        if process_map_foo is not False:
            if process_map_foo is None:
                process_map_foo = self.default_process_map
            data["process"] = [process_map_foo(i) for i in data["process"]]
        # print(data)
        if from_to is not None:
            data = data[(data[var_label] >= from_to[0]) & (data[var_label] <= from_to[1])]
        # min_data = pd.DataFrame([j[1].min().values for i in data.groupby("process") for j in i[1].groupby("alpha")], columns=data.columns)
        #max_data = pd.DataFrame([j[1].max().values for i in data.groupby("process") for j in i[1].groupby("alpha")], columns=data.columns)
        #sns.lineplot(data=min_data, x=var_label, y="result", hue="process")
        #sns.lineplot(data=max_data, x=var_label, y="result", hue="process")
        if point:
            sns.pointplot(data=data, x=var_label, y="result", hue="process", dodge=True)
        else:
            sns.lineplot(data=data, x=var_label, y="result", hue="process", ci=None)
            plt.ylim(0)
        if var_label == "bs" and log:
            plt.xscale('log', basex=10)
        # plt.ylim(0)
        plt.show()

    def draw_true_roc(self, bs, from_to=None, log=True, process_map_foo=None):
        fix_label = "bs"
        var_label = "fpr"
        value = bs
        data = self.flatten_val_results()
        data = data[data[fix_label] == value]
        if process_map_foo is not False:
            if process_map_foo is None:
                process_map_foo = self.default_process_map
        # print(data)
        if from_to is not None:
            data = data[(data[var_label] >= from_to[0]) & (data[var_label] <= from_to[1])]
        sns.lineplot(data=data, x=var_label, y="tpr", hue="process")
        if var_label == "bs" and log:
            plt.xscale('log', basex=10)
        plt.show()

    def translate_processes(self, df, process_map_foo):
        if process_map_foo is not False:
            if process_map_foo is None:
                process_map_foo = self.default_process_map
            df["process"] = [process_map_foo(i) for i in df["process"]]
        return df

    def compute_auc_auc_table(self, process_map_foo=None, index_numeric=False):
        df = self.compute_auc_auc_data()
        df["data"] = (df["process"].str.split("-", expand=True))[3]
        df = self.translate_processes(df, process_map_foo)
        if index_numeric:
            df["data"] = [int(re.findall(r"\d+", i)[0]) for i in df["data"]]
        df = df.pivot(index="data", columns="process", values="auc")
        df = df.sort_index()
        return df

    def heatmap_auc_auc_table(self, process_map_foo=None, index_numeric=False, vmin=None):
        df = self.compute_auc_auc_table(process_map_foo, index_numeric=index_numeric)
        if vmin is None:
            sns.heatmap(df, annot=True, cmap="YlGnBu",  fmt=".2f")
        else:
            sns.heatmap(df, annot=True, cmap="YlGnBu",  fmt=".2f", vmin=vmin, norm=LogNorm())
        plt.show()

    def draw_auc_curve(self, from_to=None, process_map_foo=None, point=False):
        df = self.compute_auc_data()
        if process_map_foo is not False:
            if process_map_foo is None:
                process_map_foo = self.default_process_map
            df["process"] = [process_map_foo(i) for i in df["process"]]
        if from_to is not None:
            df = df[(df["bs"] >= from_to[0]) & (df["bs"] <= from_to[1])]
        if point:
            sns.pointplot(data=df, x="bs", y="auc", hue="process", dodge=True)
        else:
            sns.lineplot(data=df, x="bs", y="auc", hue="process", ci=None)
            plt.ylim(0)
        plt.xscale('log', base=10)
        plt.show()

    def compute_auc_data(self):
        data = self.flatten_results()
        df = []
        for i in data.groupby("process"):
            for j in i[1].groupby("bs"):
                df.append([j[0], i[0], metrics.auc(j[1]["alpha"], j[1]["result"])])
        df = pd.DataFrame(df, columns=["bs", "process", "auc"])
        return df

    def compute_auc_auc_data(self):
        data = self.compute_auc_data()
        df = []
        for i in data.groupby("process"):
            df.append([i[0], metrics.auc(i[1]["bs"], i[1]["auc"])])
        df = pd.DataFrame(df, columns=["process", "auc"])
        return df

        # data = data[data[fix_label] == value]
        # if process_map_foo is not None:
        #     data["process"] = [process_map_foo(i) for i in data["process"]]
        # # print(data)
        # if from_to is not None:
        #     data = data[(data[var_label] >= from_to[0]) & (data[var_label] <= from_to[1])]
        # sns.lineplot(data=data, x=var_label, y="result", hue="process")
        # if var_label == "bs" and log:
        #     plt.xscale('log', basex=10)
        # plt.show()

    def flatten_val_results(self):
        df = pd.DataFrame(columns=self.result_header + ["process", "tpr", "fpr"])
        for k in self.processes.keys():
            df2 = self.get_results()[self.result_header + [k]].copy()
            df2["tpr"] = df2[k]
            df2["fpr"] = self.val_results[k]
            df2["process"] = k
            df2 = df2.drop(k, axis=1)
            df = df.append(df2)
        df = df.reset_index(drop=True)
        return df

        # def plot_roc(self, fix_label, value):
        #     if fix_label == "alpha":
        #         var_label = "bs"
        #     else:
        #         var_label = "alpha"
        #     data = self.results[self.results[fix_label] == value]
        #     for k in self.processes.keys():
        #         df = data[["bs", "alpha", k]]
        #         sns.lineplot(data=df, x=var_label, y=k)
        #     plt.show()

        # fig, axs = plt.subplots(1, 1, squeeze=False)
        # self.__draw_rocs(axs[0, 0], [data[var_label], self.results[self.processes.keys()],
        #                  title="{0}: {1}".format(fix_label, str(value)), log=None)
        # plt.show()

    def flatten_results(self):
        df = pd.DataFrame(columns=self.result_header + ["process","result"])
        for k in self.processes.keys():
            df2 = self.get_results()[self.result_header + [k]].copy()
            df2["result"] = df2[k]
            df2["process"] = k
            df2 = df2.drop(k, axis=1)
            df = df.append(df2)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def __draw_rocs(ax, x, y, labels=None, label_x="False", label_y="True", title="", log=None):
        if labels is not None:
            for i in range(len(x)):
                try:
                    auc = " ({0})".format(str(metrics.auc(x[i], y[i])))
                except ValueError:
                    auc = ""
                # print("-" * 100)
                # print(labels[i], auc)
                # print("-" * 100)
                # print(list(x[i]))
                # print(list(y[i]))
                # print("-" * 100)
                # print("" * 100)
                ax.plot(list(x[i]), list(y[i]), label=labels[i] + auc)
        else:
            for i in range(len(x)):
                ax.plot(x[i], y[i])

        if np.max(y) <= 1 and np.max(x) <= 1:
            ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.05])
        ax.set_xlabel(" ({0})".format(label_x))
        ax.set_ylabel(" ({0})".format(label_y))
        if log is not None:
            ax.set_xscale('log', basex=log)
        plt.title(title)
        if labels is not None:
            plt.legend(loc="lower right")

    def compute_auc(self):
        pass

    def compose_report(self):
        report = self.get_results().copy()
        return report[['bs', 'alpha'] + list(self.processes.keys())]


class TransformProcessAnalysis:

    def __init__(self, calibration_data, test_data, classifier):
        self.calibration_data = calibration_data
        self.test_data = test_data
        self.classifier = classifier
        self.processes = [[Process(self.calibration_data, j, i) for i in classifier]
                                   for j in test_data]
        self.analysis = [Analysis(i) for i in self.processes]
        self.reports = [a.compose_report() for a in self.analysis]
        self.common_data_name = self.compute_data_name()
        self.val_name = "sigma" if "N" in self.common_data_name else "percent"
        self.default_name = self.common_data_name + "-" + "-".join([c.get_name() for c in self.classifier])

    @staticmethod
    def adjust_names(report):
        report = report.replace("ksconf", "max")
        report = report.replace("margin_0_1", "margin")
        return report

    def compute_data_name(self):
        new_s = ""
        for i, d in enumerate(self.test_data):
            s = d.get_name()
            if i == 0:
                new_s = s
            n = min(len(new_s), len(s))
            new_s = "".join([new_s[j] for j in range(n) if new_s[j] == s[j]])
        return new_s

    def compute_first_100(self, as_df=True):
        data = []
        for j in range(len(self.reports)):
            df = self.reports[j]
            p = self.processes[j]
            for d in df.groupby("alpha"):
                for i in p:
                    if self.common_data_name is not None:
                        dp = int(i.test_data.get_name().replace(self.common_data_name, ""))
                    else:
                        dp = i.test_data.get_name()
                    if len(d[1][d[1][i.get_name()] >= 1]) > 0:
                        r = [dp, d[0], i.classifier.get_name(), d[1][d[1][i.get_name()] >= 1].iloc[0]["bs"]]
                    else:
                        r = [dp, d[0], i.classifier.get_name(), np.NaN]
                    data.append(r)
        if as_df:
            data = pd.DataFrame(data, columns=[self.val_name, "alpha", "process", "bs"])
        return self.adjust_names(data)

    def heatmap_first_100(self, alpha, save=False):
        df = self.compute_first_100()
        df = df[df["alpha"] == alpha].pivot(index=self.val_name, columns='process', values='bs')
        sns.heatmap(df, annot=True, cmap="YlGnBu_r", norm=LogNorm(), fmt=".0f")
        plt.title("First 100 ({0}): ".format(str(alpha)) + self.common_data_name)
        plt.tight_layout()
        if save:
            plt.savefig(fname=util.get_summary_path("heatmap-100" + self.default_name, ".jpg"), dpi=1000, format="jpg")
            plt.clf()
        else:
            plt.show()

    def heatmap_first_100_best(self, save=False):
        df = self.compute_first_100()
        tabl = [i for i in df.groupby(self.val_name)]
        for n in range(len(tabl)):
            obs = tabl[n][1].pivot(index='alpha', columns='process', values='bs')
            if n == 0:
                df = pd.DataFrame((obs.sub(obs.min(axis=1), axis=0) == 0).sum(), columns=[tabl[n][0]])
            else:
                df[tabl[n][0]] = (obs.sub(obs.min(axis=1), axis=0) == 0).sum()
        df.transpose()
        sns.heatmap(df.transpose(), annot=True, cmap="YlGnBu", fmt=".0f")
        plt.title("Aggregated 100: " + self.common_data_name)
        plt.tight_layout()
        if save:
            plt.savefig(fname=util.get_summary_path("heatmap-100_best" + self.default_name, ".jpg"), dpi=1000, format="jpg")
            plt.clf()
        else:
            plt.show()

