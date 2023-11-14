import numpy as np
import os
import scipy
import shutil

class PreProcess:
    import tensorflow.compat.v1 as tf  # for TF 2
    import tensornets as nets

    model_map = {"VGG19": nets.VGG19, "ResNet50": nets.ResNet50, "SqueezeNet": nets.SqueezeNet,
                 "MobileNet25": nets.MobileNet25, "NASNetAlarge": nets.NASNetAlarge}

    BASE_PATH = "/home/krk/Dropbox/IST/year_1/rotations/rotation_1/data"
    DATA_PATH = "./data"

    def __init__(self, model):
        self.model_name = model
        self.tf.disable_v2_behavior()
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        self.transformations = {"blur": PreProcess.img_blur, "noise": PreProcess.img_noise,
                           "dead": PreProcess.img_dead, "expos": PreProcess.img_expos}
        self.data = None
        self.inputs = self.tf.placeholder(self.tf.float32, [None, 224, 224, 3])
        self.model = self.model_map[self.model_name](self.inputs)
        # self.tf.reset_default_graph()

    @staticmethod
    def img_blur(sigma):
        pass

    @staticmethod
    def img_noise(img, sigma):
        rand = np.random.standard_normal(img.shape)
        return np.clip(img+sigma*rand, 0, 255)

    @staticmethod
    def img_dead(img, percent):
        for i in range(img.shape[0]):
            x = img.shape[1]
            y = img.shape[2]
            ind = np.random.choice(x * y, replace=False, size=int(np.round(percent * x * y, 0)))
            n2 = int(np.floor(0.5 * len(ind)))
            ind_0 = ind[:n2]
            ind_1 = ind[n2:]
            img[i, np.unravel_index(ind_0, (x, y))[0], np.unravel_index(ind_0, (x, y))[1], :] = 0
            img[i, np.unravel_index(ind_1, (x, y))[0], np.unravel_index(ind_1, (x, y))[1], :] = 255
        return img

    @staticmethod
    def img_expos(sigma):
        pass

    @staticmethod
    def path_addition(transform_name, *args, **kwargs):
        if transform_name is None:
            return ""
        return "+" + "_".join([transform_name] + list(args) + [str(k)+str(v) for k, v in kwargs.items()])

    def get_data(self, path_foo, store):
        if store:
            if self.data is None:
                self.data = path_foo()
            data = self.data
        else:
            data = path_foo()
        return data

    def get_transform_foo(self, foo_name, *args, **kwargs):
        foo = self.transformations.get(str(foo_name), None)
        if foo is None:
            return foo
        else:
            return lambda img: foo(img, *args, **kwargs)

    def extract_ilsvrc_cal_paths(self):
        path = os.path.join(self.BASE_PATH, "test")
        data = sorted([os.path.join(path, i) for i in os.listdir(path)])
        return data

    def extract_ilsvrc_val_paths(self):
        path = os.path.join(self.BASE_PATH, "val")
        data = sorted([os.path.join(i, j) for i in [os.path.join(path, p) for p in os.listdir(path)] for j in os.listdir(i)])
        return data

    def extract_awa_paths(self):
        path = os.path.join(self.BASE_PATH, "ooc")
        data = sorted([os.path.join(i, j) for i in [os.path.join(path, p) for p in os.listdir(path)] for j in os.listdir(i)])
        return data

    def extract_ilsvrc_cal_scores(self, n=None, batch_size=1000, start_batch=0, store=False, batch_process=True, transform=None,
                                  classes=False, store_batches=True, end_batch=None, *args, **kwargs):
        data = self.get_data(self.extract_ilsvrc_cal_paths, store)
        path = os.path.join(self.DATA_PATH, self.model_name + "_ILSVRC_cal_data{0}".format(self.path_addition(transform, *args, **kwargs)))
        self.__extract_aid(data, path, n, batch_size, start_batch=start_batch, batch_process=batch_process,
                           transform_foo=self.get_transform_foo(transform, *args, **kwargs),
                           classes=classes, store_batches=store_batches, end_batch=end_batch)

    def extract_ilsvrc_val_scores(self, n=None,  batch_size=1000, start_batch=0, store=False, batch_process=True, transform=None,
                                  classes=False, store_batches=True, end_batch=None, *args, **kwargs):
        data = self.get_data(self.extract_ilsvrc_val_paths, store)
        path = os.path.join(self.DATA_PATH, self.model_name + "_ILSVRC_val_data{0}".format(self.path_addition(transform, *args, **kwargs)))
        self.__extract_aid(data, path, n, batch_size, start_batch=start_batch, batch_process=batch_process,
                           transform_foo=self.get_transform_foo(transform, *args, **kwargs),
                           classes=classes, store_batches=store_batches, end_batch=end_batch)

    def extract_awa_scores(self, n=None, batch_size=1000, start_batch=0, store=False, batch_process=True, transform=None,
                                      classes=False, store_batches=True, end_batch=None, *args, **kwargs):
        data = self.get_data(self.extract_awa_paths, store)
        path = os.path.join(self.DATA_PATH, self.model_name + "_AwA_data{0}".format(self.path_addition(transform, *args, **kwargs)))
        self.__extract_aid(data, path, n, batch_size, start_batch=start_batch, batch_process=batch_process,
                           transform_foo=self.get_transform_foo(transform, *args, **kwargs),
                           classes=classes, store_batches=store_batches, end_batch=end_batch)

    def __extract_aid(self, data, path, n=None, batch_size=1000, start_batch=0, batch_process=True, transform_foo=None,
                      classes=False, store_batches=True, end_batch=None):
        if not batch_process:
            batch_size = len(data)
        if n is None:
            conf_scores = self.batch_extract_confidence_scores(data, batch_size=batch_size, start_batch=start_batch,
                                                               path=path if batch_process else None,
                                                               transform_foo=transform_foo,
                                                               classes=classes, store_batches=store_batches,
                                                               end_batch=end_batch)
        else:
            conf_scores = self.batch_extract_confidence_scores(data[:n], batch_size=batch_size, start_batch=start_batch,
                                                               path=path if batch_process else None,
                                                               transform_foo=transform_foo,
                                                               classes=classes, store_batches=store_batches,
                                                               end_batch=end_batch)
        # if not batch_process:
        #     conf_scores = np.array(conf_scores)
        #     np.savetxt(path, conf_scores, delimiter=",")

    def batch_extract_confidence_scores(self, data, batch_size=1000, start_batch=0, path=None, transform_foo=None,
                                        classes=False, store_batches=True, end_batch=None):

        batches = range(0, len(data), batch_size)
        print("TB", len(batches))
        print("SB", start_batch)
        if start_batch == 0:
            try:
                shutil.rmtree("temp")
            except Exception:
                pass
        try:
            os.mkdir("temp")
        except FileExistsError:
            pass
        #self.tf.reset_default_graph()
        for i in range(start_batch, len(batches)):
            if end_batch is not None and i == end_batch:
                break
            print("-"*100)
            print("Batch: {0} start.".format(i))
            if i+1 < len(batches):
                d = data[batches[i]:batches[i+1]]
            else:
                d = data[batches[i]:]
            res, cla = self.extract_confidence_scores(d, transform_foo, classes=classes)
            np.savetxt(os.path.join("temp", str(i) + ".csv"), res, delimiter=",")
            if classes:
                np.savetxt(os.path.join("temp", str(i) + "_labels.csv"), cla, delimiter=",", fmt="%s")
            print("Batch: {0} fin.".format(i))
            print("-" * 100)
            print("")
        if store_batches:
            res = np.concatenate([np.loadtxt(os.path.join("temp", i), delimiter=",") for i in os.listdir("temp") if "_labels" not in i])
            np.savetxt(path + ".csv", res, delimiter=",")
            fin = res
            if classes:
                cla = np.concatenate([np.loadtxt(os.path.join("temp", i), delimiter=",", dtype="str") for i in os.listdir("temp") if "_labels" in i])
                np.savetxt(path + "_labels.csv", cla, delimiter=",", fmt="%s")
                fin = res, cla
            # self.tf.reset_default_graph()
            return fin

    def batch_extract_confidence_scores2(self, data, batch_size=1000, start_batch=0, path=None, transform_foo=None,
                                        classes=False, store_batches=True):
        batches = range(0, len(data), batch_size)
        res_scores = []
        cla_scores = []
        print("TB", len(batches))
        print("SB", start_batch)
        for i in range(start_batch, len(batches)):
            print("-"*100)
            print("Batch: {0} start.".format(i))
            if i+1 < len(batches):
                d = data[batches[i]:batches[i+1]]
            else:
                d = data[batches[i]:]
            res, cla = self.extract_confidence_scores(d, transform_foo, classes=classes)
            if not store_batches or path is None:
                res_scores += res
                if classes:
                    cla_scores += cla
            else:
                # if i != 0:
                #     res = np.concatenate([np.loadtxt(path + ".csv", delimiter=","), res])
                #     if classes:
                #         cla = np.concatenate([np.loadtxt(path + "_labels.csv", delimiter=",", dtype="str"), cla])
                # print("Batch: {0} not saved.".format(i))
                np.savetxt(path + str(i) + ".csv", res, delimiter=",")
                if classes:
                    np.savetxt(path + str(i) + "_labels.csv", cla, delimiter=",", fmt="%s")
            print("Batch: {0} fin.".format(i))
            print("-" * 100)
            print("")
        if not store_batches and path is not None:
            np.savetxt(path + ".csv", res_scores, delimiter=",")
            if classes:
                np.savetxt(path + "_labels.csv", cla_scores, delimiter=",", fmt="%s")
        if store_batches:
            res = np.concatenate([np.loadtxt(path + str(i) + ".csv", delimiter=",") for i in range(start_batch, len(batches))])
            np.savetxt(path + ".csv", res_scores, delimiter=",")
            fin = res
            if classes:
                cla = np.concatenate([np.loadtxt(path + str(i) + "_labels.csv", delimiter=",", dtype="str") for i in range(start_batch, len(batches))])
                np.savetxt(path + "_labels.csv", cla_scores, delimiter=",", fmt="%s")
                fin = res, cla
            for i in range(start_batch, len(batches)):
                try:
                    os.remove(path + str(i) + ".csv")
                    os.remove(path + str(i) + "_labels.csv")
                except FileNotFoundError:
                    pass
            return fin
        else:
            if classes:
                return res_scores, cla_scores
            else:
                return res_scores

    def extract_confidence_scores(self, data, transform_foo=None, classes=False):
        # self.tf.reset_default_graph()
        img = self.nets.utils.load_img(data, target_size=(256, 256), crop_size=224)
        if transform_foo is not None:
            img = transform_foo(img)
        with self.tf.Session() as sess:
            img = self.model.preprocess(img)  # equivalent to img = nets.preprocess(model, img)
            sess.run(self.model.pretrained())  # equivalent to nets.pretrained(model)
            preds = sess.run(self.model, {self.inputs: img})
            res = [[j[2] for j in i] for i in self.nets.utils.decode_predictions(preds, top=0)]
            clas = None
            if classes:
                clas = [[j[0] for j in i] for i in self.nets.utils.decode_predictions(preds, top=0)]
        return res, clas

    @staticmethod
    def check_data(data):
        for i in data:
            if ".JPEG" not in i:
                 print(i)