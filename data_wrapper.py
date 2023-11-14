import util
import numpy as np
import os

class GenericData:

    _name = "generic"

    def __init__(self, path, name=None, file_name=None):
        self._path = self._compute_path(path)
        self._data = None
        self.file_name = file_name if file_name is not None else path

        if name is not None:
            self._name = name

    def get_name(self):
        return self._name

    def exists(self):
        return os.path.exists(self._path)

    def get(self, persist=False):
        if self._data is None:
            data = self.load(persist)
            return data
        return self._data

    def set(self, data):
        self._data = data

    def sample(self, size):
        index = np.random.choice(list(range(len(self._data))), size)
        return self._data[index]

    def load(self, persist=True):
        data = util.load_file(self._path)
        if persist:
            self._data = data
        return data

    def write(self):
        if self._data is not None:
            util.write_file(self._path, self._data)

    def get_path(self):
        return self._path

    def get_file_name(self):
        return self.file_name

    @staticmethod
    def _compute_path(path):
        return path


class RawData(GenericData):

    _name = "raw_data"

    @staticmethod
    def _compute_path(path):
        return util.get_data_path(path)


class ResultData(GenericData):

    _name = "result_data"

    @staticmethod
    def _compute_path(path):
        return util.get_result_path(path)


class ConfigData(ResultData):

    _name = "config_data"

    @staticmethod
    def _compute_path(path):
        return util.get_config_path(path)


class SummaryData(GenericData):
    _name = "summary_data"

    @staticmethod
    def _compute_path(path):
        return util.get_summary_path(path)


class ClassifierData(GenericData):
    _name = "classifier_data"

    def __init__(self, path, classifier, name=None):
        path = classifier.get_name() + "-" + path
        super().__init__(path, name)
        self._path = self._compute_path(path)
        self._data = None

        if name is not None:
            self._name = name

    @staticmethod
    def _compute_path(path):
        return util.get_clf_path(path)


class MultiClassifierData:
    _name = "multi_classifier_data"

    def __init__(self, path, classifier, name=None):
        self.classifier_names = [k.get_name() for k in classifier]
        if name is not None:
            self._name = name
        if isinstance(name, dict):
            self.data = {k.get_name(): ClassifierData(path, k, name[k.get_name()]) for k in classifier}
        else:
            self.data = {k.get_name(): ClassifierData(path, k, name) for k in classifier}

    def _get_data_list(self):
        return [self.data[k] for k in self.classifier_names]

    def get_data(self, classifier=None):
        return self.select_foo(lambda c: self.data[c], classifier)

    def get_name(self, aggregate=False):
        if aggregate:
            dic = self.select_foo(ClassifierData.get_name)
            return "-".join(dic[k] for k in self.classifier_names)
        else:
            return self._name

    def select_foo(self, foo, clf=None, *args, **kwargs):
        as_list = False
        if isinstance(clf, list):
            as_list = True
        else:
            if clf is None:
                clf = self.classifier_names
            elif isinstance(clf, str):
                clf = [clf]
            else:
                clf = [clf.get_name()]
        if len(clf) > 1 or as_list:
            return {c: foo(self.data[c], *args, **kwargs) for c in clf}
        else:
            return foo(clf[0], *args, **kwargs)

    def exists(self, classifier=None):
        return self.select_foo(ClassifierData.exists, classifier)

    def get(self, persist=False, classifier=None):
        return self.select_foo(ClassifierData.get, classifier, persist=persist)

    def set(self, data, classifier=None):
        return self.select_foo(ClassifierData.set, classifier, data=data)

    def sample(self, size, classifier=None):
        return self.select_foo(ClassifierData.sample, classifier, size=size)

    def load(self, persist=True, classifier=None):
        return self.select_foo(ClassifierData.get, classifier, persist=persist)

    def write(self, classifier=None):
        return self.select_foo(ClassifierData.write, classifier)

    def get_path(self, classifier=None):
        return self.select_foo(ClassifierData.get_path, classifier)

    def get_file_name(self, classifier=None):
        return self.select_foo(ClassifierData.get_file_name, classifier)

    @staticmethod
    def _compute_path(path):
        return path