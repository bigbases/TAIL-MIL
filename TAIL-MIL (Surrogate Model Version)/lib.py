#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler


def get_data_dim(dataset):
    """
    :param dataset: Name of dataset
    :return: Number of dimensions in data
    """
    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 55
    elif str(dataset).startswith("machine"):
        return 38
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_target_dims(dataset):
    """
    :param dataset: Name of dataset
    :return: index of data dimension that should be modeled (forecasted and reconstructed),
                     returns None if all input dimensions should be modeled
    """
    if dataset == "SMAP":
        return [0]
    elif dataset == "MSL":
        return [0]
    elif dataset == "SMD":
        return None
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_data(dataset, max_train_size=None, max_test_size=None,
             normalize=False, spec_res=False, train_start=0, test_start=0):
    """
    Get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    prefix = "datasets"
    if str(dataset).startswith("machine"):
        prefix += "/ServerMachineDataset/processed"
    elif dataset in ["MSL", "SMAP"]:
        prefix += "/data/processed"
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print("load data of:", dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None

    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None)
        test_data, _ = normalize_data(test_data, scaler=scaler)

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", None if test_label is None else test_label.shape)
    return (train_data, None), (test_data, test_label)


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.plot(losses["train_recon"], label="Recon loss")
    plt.plot(losses["train_total"], label="Total loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.plot(losses["val_recon"], label="Recon loss")
    plt.plot(losses["val_total"], label="Total loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()


def load(model, PATH, device="cpu"):
    """
    Loads the model's parameters from the path mentioned
    :param PATH: Should contain pickle file
    """
    model.load_state_dict(torch.load(PATH, map_location=device))


def get_series_color(y):
    if np.average(y) >= 0.95:
        return "black"
    elif np.average(y) == 0.0:
        return "black"
    else:
        return "black"


def get_y_height(y):
    if np.average(y) >= 0.95:
        return 1.5
    elif np.average(y) == 0.0:
        return 0.1
    else:
        return max(y) + 0.1


def adjust_anomaly_scores(scores, dataset, is_train, lookback):
    """
    Method for MSL and SMAP where channels have been concatenated as part of the preprocessing
    :param scores: anomaly_scores
    :param dataset: name of dataset
    :param is_train: if scores is from train set
    :param lookback: lookback (window size) used in model
    """

    # Remove errors for time steps when transition to new channel (as this will be impossible for model to predict)
    if dataset.upper() not in ['SMAP', 'MSL']:
        return scores

    adjusted_scores = scores.copy()
    if is_train:
        md = pd.read_csv(f'./datasets/data/{dataset.lower()}_train_md.csv')
    else:
        md = pd.read_csv('./datasets/data/labeled_anomalies.csv')
        md = md[md['spacecraft'] == dataset.upper()]

    md = md[md['chan_id'] != 'P-2']

    # Sort values by channel
    md = md.sort_values(by=['chan_id'])

    # Getting the cumulative start index for each channel
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    sep_cuma = sep_cuma[:-1]
    buffer = np.arange(1, 20)
    i_remov = np.sort(np.concatenate((sep_cuma, np.array([i+buffer for i in sep_cuma]).flatten(),
                                      np.array([i-buffer for i in sep_cuma]).flatten())))
    i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
    i_remov = np.sort(np.unique(i_remov))
    if len(i_remov) != 0:
        adjusted_scores[i_remov] = 0

    # Normalize each concatenated part individually
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    s = [0] + sep_cuma.tolist()
    for c_start, c_end in [(s[i], s[i+1]) for i in range(len(s)-1)]:
        e_s = adjusted_scores[c_start: c_end+1]

        e_s = (e_s - np.min(e_s))/(np.max(e_s) - np.min(e_s))
        adjusted_scores[c_start: c_end+1] = e_s

    return adjusted_scores


"""
Created on Mon Dec 12 10:08:16 2016
@author: Alban Siffer
@company: Amossys
@license: GNU GPLv3
Code from https://github.com/NetManAIOps/OmniAnomaly
"""


from math import floor, log

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy.optimize import minimize

# colors for plot
deep_saffron = "#FF9933"
air_force_blue = "#5D8AA8"

"""
================================= MAIN CLASS ==================================
"""


class SPOT:
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)

    Attributes
    ----------
    proba : float
            Detection level (risk), chosen by the user

    extreme_quantile : float
            current threshold (bound between normal and abnormal events)

    data : numpy.array
            stream

    init_data : numpy.array
            initial batch of observations (for the calibration/initialization step)

    init_threshold : float
            initial threshold computed during the calibration step

    peaks : numpy.array
            array of peaks (excesses above the initial threshold)

    n : int
            number of observed values

    Nt : int
            number of observed peaks
    """

    def __init__(self, q=1e-4):
        """
        Constructor
        Parameters
        ----------
        q
                Detection level (risk)

        Returns
        ----------
        SPOT object
        """
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def __str__(self):
        s = ""
        s += "Streaming Peaks-Over-Threshold Object\n"
        s += "Detection level q = %s\n" % self.proba
        if self.data is not None:
            s += "Data imported : Yes\n"
            s += "\t initialization  : %s values\n" % self.init_data.size
            s += "\t stream : %s values\n" % self.data.size
        else:
            s += "Data imported : No\n"
            return s

        if self.n == 0:
            s += "Algorithm initialized : No\n"
        else:
            s += "Algorithm initialized : Yes\n"
            s += "\t initial threshold : %s\n" % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += "Algorithm run : Yes\n"
                s += "\t number of observations : %s (%.2f %%)\n" % (
                    r,
                    100 * r / self.n,
                )
            else:
                s += "\t number of peaks  : %s\n" % self.Nt
                s += "\t extreme quantile : %s\n" % self.extreme_quantile
                s += "Algorithm run : No\n"
        return s

    def fit(self, init_data, data):
        """
        Import data to SPOT object

        Parameters
        ----------
        init_data : list, numpy.array or pandas.Series
                initial batch to calibrate the algorithm

        data : numpy.array
                data for the run (list, np.array or pd.series)

        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print("This data format (%s) is not supported" % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print("The initial data cannot be set")
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data

        Parameters
        ----------
        data : list, numpy.array, pandas.Series
                data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print("This data format (%s) is not supported" % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        """
        Run the calibration (initialization) step

        Parameters
        ----------
        level : float
                (default 0.98) Probability associated with the initial threshold t
        verbose : bool
                (default = True) If True, gives details about the batch initialization
        verbose: bool
                (default True) If True, prints log
        min_extrema bool
                (default False) If True, find min extrema instead of max extrema
        """
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level

        level = level - floor(level)

        n_init = self.init_data.size

        S = np.sort(self.init_data)  # we sort X to get the empirical quantile
        self.init_threshold = S[int(level * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print("Initial threshold : %s" % self.init_threshold)
            print("Number of peaks : %s" % self.Nt)
            print("Grimshaw maximum log-likelihood estimation ... ", end="")

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print("[done]")
            print("\t" + chr(0x03B3) + " = " + str(g))
            print("\t" + chr(0x03C3) + " = " + str(s))
            print("\tL = " + str(l))
            print("Extreme quantile (probability = %s): %s" % (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
                scalar function
        jac : function
                first order derivative of the function
        bounds : tuple
                (min,max) interval for the roots search
        npoints : int
                maximum number of roots to output
        method : str
                'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval

        Returns
        ----------
        numpy.array
                possible roots of the function
        """
        if method == "regular":
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == "random":
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(
            lambda X: objFun(X, fun, jac),
            X0,
            method="L-BFGS-B",
            jac=True,
            bounds=[bounds] * len(X0),
        )

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        Y : numpy.array
                observations
        gamma : float
                GPD index parameter
        sigma : float
                GPD scale parameter (>0)
        Returns
        ----------
        float
                log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
                numerical parameter to perform (default : 1e-8)
        n_points : int
                maximum number of candidates for maximum likelihood (default : 10)
        Returns
        ----------
        gamma_best,sigma_best,ll_best
                gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = SPOT._rootsFinder(
            lambda t: w(self.peaks, t),
            lambda t: jac_w(self.peaks, t),
            (a + epsilon, -epsilon),
            n_points,
            "regular",
        )

        right_zeros = SPOT._rootsFinder(
            lambda t: w(self.peaks, t),
            lambda t: jac_w(self.peaks, t),
            (b, c),
            n_points,
            "regular",
        )

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q

        Parameters
        ----------
        gamma : float
                GPD parameter
        sigma : float
                GPD parameter
        Returns
        ----------
        float
                quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True, dynamic=True):
        """
		Run SPOT on the stream

		Parameters
		----------
		with_alarm : bool
			(default = True) If False, SPOT will adapt the threshold assuming \
			there is no abnormal values
		Returns
		----------
		dict
			keys : 'thresholds' and 'alarms'

			'thresholds' contains the extreme quantiles and 'alarms' contains \
			the indexes of the values which have triggered alarms

		"""
        if self.n > self.init_data.size:
            print(
                "Warning : the algorithm seems to have already been run, you \
            should initialize before running again"
            )
            return {}

        # list of the thresholds
        th = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):

            if not dynamic:
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                # If the observed value exceeds the current threshold (alarm case)
                if self.data[i] > self.extreme_quantile:
                    # if we want to alarm, we put it in the alarm list
                    if with_alarm:
                        alarm.append(i)
                    # otherwise we add it in the peaks
                    else:
                        self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                        # self.peaks = self.peaks[1:]
                        self.Nt += 1
                        self.n += 1
                        # and we update the thresholds

                        g, s, l = self._grimshaw()
                        self.extreme_quantile = self._quantile(g, s)

                # case where the value exceeds the initial threshold but not the alarm ones
                elif self.data[i] > self.init_threshold:
                    # we add it in the peaks
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    # self.peaks = self.peaks[1:]
                    self.Nt += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                else:
                    self.n += 1

            th.append(self.extreme_quantile)  # thresholds record

        return {"thresholds": th, "alarms": alarm}

    def plot(self, run_results, with_alarm=True):
        """
        Plot the results of given by the run

        Parameters
        ----------
        run_results : dict
                results given by the 'run' method
        with_alarm : bool
                (default = True) If True, alarms are plotted.
        Returns
        ----------
        list
                list of the plots

        """
        x = range(self.data.size)
        K = run_results.keys()

        (ts_fig,) = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if "thresholds" in K:
            th = run_results["thresholds"]
            (th_fig,) = plt.plot(x, th, color=deep_saffron, lw=2, ls="dashed")
            fig.append(th_fig)

        if with_alarm and ("alarms" in K):
            alarm = run_results["alarms"]
            al_fig = plt.scatter(alarm, self.data[alarm], color="red")
            fig.append(al_fig)

        plt.xlim((0, self.data.size))

        return fig


"""
============================ UPPER & LOWER BOUNDS =============================
"""


class biSPOT:
    """
    This class allows to run biSPOT algorithm on univariate dataset (upper and lower bounds)

    Attributes
    ----------
    proba : float
            Detection level (risk), chosen by the user

    extreme_quantile : float
            current threshold (bound between normal and abnormal events)

    data : numpy.array
            stream

    init_data : numpy.array
            initial batch of observations (for the calibration/initialization step)

    init_threshold : float
            initial threshold computed during the calibration step

    peaks : numpy.array
            array of peaks (excesses above the initial threshold)

    n : int
            number of observed values

    Nt : int
            number of observed peaks
    """

    def __init__(self, q=1e-4):
        """
        Constructor
        Parameters
        ----------
        q
                Detection level (risk)

        Returns
        ----------
        biSPOT object
        """
        self.proba = q
        self.data = None
        self.init_data = None
        self.n = 0
        nonedict = {"up": None, "down": None}

        self.extreme_quantile = dict.copy(nonedict)
        self.init_threshold = dict.copy(nonedict)
        self.peaks = dict.copy(nonedict)
        self.gamma = dict.copy(nonedict)
        self.sigma = dict.copy(nonedict)
        self.Nt = {"up": 0, "down": 0}

    def __str__(self):
        s = ""
        s += "Streaming Peaks-Over-Threshold Object\n"
        s += "Detection level q = %s\n" % self.proba
        if self.data is not None:
            s += "Data imported : Yes\n"
            s += "\t initialization  : %s values\n" % self.init_data.size
            s += "\t stream : %s values\n" % self.data.size
        else:
            s += "Data imported : No\n"
            return s

        if self.n == 0:
            s += "Algorithm initialized : No\n"
        else:
            s += "Algorithm initialized : Yes\n"
            s += "\t initial threshold : %s\n" % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += "Algorithm run : Yes\n"
                s += "\t number of observations : %s (%.2f %%)\n" % (
                    r,
                    100 * r / self.n,
                )
                s += "\t triggered alarms : %s (%.2f %%)\n" % (
                    len(self.alarm),
                    100 * len(self.alarm) / self.n,
                )
            else:
                s += "\t number of peaks  : %s\n" % self.Nt
                s += "\t upper extreme quantile : %s\n" % self.extreme_quantile["up"]
                s += "\t lower extreme quantile : %s\n" % self.extreme_quantile["down"]
                s += "Algorithm run : No\n"
        return s

    def fit(self, init_data, data):
        """
        Import data to biSPOT object

        Parameters
        ----------
        init_data : list, numpy.array or pandas.Series
                initial batch to calibrate the algorithm ()

        data : numpy.array
                data for the run (list, np.array or pd.series)

        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print("This data format (%s) is not supported" % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print("The initial data cannot be set")
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data

        Parameters
        ----------
        data : list, numpy.array, pandas.Series
                data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print("This data format (%s) is not supported" % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, verbose=True):
        """
        Run the calibration (initialization) step

        Parameters
        ----------
        verbose : bool
                (default = True) If True, gives details about the batch initialization
        """
        n_init = self.init_data.size

        S = np.sort(self.init_data)  # we sort X to get the empirical quantile
        self.init_threshold["up"] = S[int(0.98 * n_init)]  # t is fixed for the whole algorithm
        self.init_threshold["down"] = S[int(0.02 * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks["up"] = self.init_data[self.init_data > self.init_threshold["up"]] - self.init_threshold["up"]
        self.peaks["down"] = -(
            self.init_data[self.init_data < self.init_threshold["down"]] - self.init_threshold["down"]
        )
        self.Nt["up"] = self.peaks["up"].size
        self.Nt["down"] = self.peaks["down"].size
        self.n = n_init

        if verbose:
            print("Initial threshold : %s" % self.init_threshold)
            print("Number of peaks : %s" % self.Nt)
            print("Grimshaw maximum log-likelihood estimation ... ", end="")

        l = {"up": None, "down": None}
        for side in ["up", "down"]:
            g, s, l[side] = self._grimshaw(side)
            self.extreme_quantile[side] = self._quantile(side, g, s)
            self.gamma[side] = g
            self.sigma[side] = s

        ltab = 20
        form = "\t" + "%20s" + "%20.2f" + "%20.2f"
        if verbose:
            print("[done]")
            print("\t" + "Parameters".rjust(ltab) + "Upper".rjust(ltab) + "Lower".rjust(ltab))
            print("\t" + "-" * ltab * 3)
            print(form % (chr(0x03B3), self.gamma["up"], self.gamma["down"]))
            print(form % (chr(0x03C3), self.sigma["up"], self.sigma["down"]))
            print(form % ("likelihood", l["up"], l["down"]))
            print(
                form
                % (
                    "Extreme quantile",
                    self.extreme_quantile["up"],
                    self.extreme_quantile["down"],
                )
            )
            print("\t" + "-" * ltab * 3)
        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
                scalar function
        jac : function
                first order derivative of the function
        bounds : tuple
                (min,max) interval for the roots search
        npoints : int
                maximum number of roots to output
        method : str
                'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval

        Returns
        ----------
        numpy.array
                possible roots of the function
        """
        if method == "regular":
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == "random":
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(
            lambda X: objFun(X, fun, jac),
            X0,
            method="L-BFGS-B",
            jac=True,
            bounds=[bounds] * len(X0),
        )

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        Y : numpy.array
                observations
        gamma : float
                GPD index parameter
        sigma : float
                GPD scale parameter (>0)
        Returns
        ----------
        float
                log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, side, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
                numerical parameter to perform (default : 1e-8)
        n_points : int
                maximum number of candidates for maximum likelihood (default : 10)
        Returns
        ----------
        gamma_best,sigma_best,ll_best
                gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks[side].min()
        YM = self.peaks[side].max()
        Ymean = self.peaks[side].mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = biSPOT._rootsFinder(
            lambda t: w(self.peaks[side], t),
            lambda t: jac_w(self.peaks[side], t),
            (a + epsilon, -epsilon),
            n_points,
            "regular",
        )

        right_zeros = biSPOT._rootsFinder(
            lambda t: w(self.peaks[side], t),
            lambda t: jac_w(self.peaks[side], t),
            (b, c),
            n_points,
            "regular",
        )

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = biSPOT._log_likelihood(self.peaks[side], gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks[side]) - 1
            sigma = gamma / z
            ll = biSPOT._log_likelihood(self.peaks[side], gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, side, gamma, sigma):
        """
        Compute the quantile at level 1-q for a given side

        Parameters
        ----------
        side : str
                'up' or 'down'
        gamma : float
                GPD parameter
        sigma : float
                GPD parameter
        Returns
        ----------
        float
                quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        if side == "up":
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold["up"] + (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold["up"] - sigma * log(r)
        elif side == "down":
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold["down"] - (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold["down"] + sigma * log(r)
        else:
            print("error : the side is not right")

    def run(self, with_alarm=True):
        """
		Run biSPOT on the stream

		Parameters
		----------
		with_alarm : bool
			(default = True) If False, SPOT will adapt the threshold assuming \
			there is no abnormal values
		Returns
		----------
		dict
			keys : 'upper_thresholds', 'lower_thresholds' and 'alarms'

			'***-thresholds' contains the extreme quantiles and 'alarms' contains \
			the indexes of the values which have triggered alarms

		"""
        if self.n > self.init_data.size:
            print(
                "Warning : the algorithm seems to have already been run, you \
            should initialize before running again"
            )
            return {}

        # list of the thresholds
        thup = []
        thdown = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):

            # If the observed value exceeds the current threshold (alarm case)
            if self.data[i] > self.extreme_quantile["up"]:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks["up"] = np.append(self.peaks["up"], self.data[i] - self.init_threshold["up"])
                    self.Nt["up"] += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw("up")
                    self.extreme_quantile["up"] = self._quantile("up", g, s)

            # case where the value exceeds the initial threshold but not the alarm ones
            elif self.data[i] > self.init_threshold["up"]:
                # we add it in the peaks
                self.peaks["up"] = np.append(self.peaks["up"], self.data[i] - self.init_threshold["up"])
                self.Nt["up"] += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._grimshaw("up")
                self.extreme_quantile["up"] = self._quantile("up", g, s)

            elif self.data[i] < self.extreme_quantile["down"]:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks["down"] = np.append(
                        self.peaks["down"],
                        -(self.data[i] - self.init_threshold["down"]),
                    )
                    self.Nt["down"] += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw("down")
                    self.extreme_quantile["down"] = self._quantile("down", g, s)

            # case where the value exceeds the initial threshold but not the alarm ones
            elif self.data[i] < self.init_threshold["down"]:
                # we add it in the peaks
                self.peaks["down"] = np.append(self.peaks["down"], -(self.data[i] - self.init_threshold["down"]))
                self.Nt["down"] += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._grimshaw("down")
                self.extreme_quantile["down"] = self._quantile("down", g, s)
            else:
                self.n += 1

            thup.append(self.extreme_quantile["up"])  # thresholds record
            thdown.append(self.extreme_quantile["down"])  # thresholds record

        return {"upper_thresholds": thup, "lower_thresholds": thdown, "alarms": alarm}

    def plot(self, run_results, with_alarm=True):
        """
        Plot the results of given by the run

        Parameters
        ----------
        run_results : dict
                results given by the 'run' method
        with_alarm : bool
                (default = True) If True, alarms are plotted.
        Returns
        ----------
        list
                list of the plots

        """
        x = range(self.data.size)
        K = run_results.keys()

        (ts_fig,) = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if "upper_thresholds" in K:
            thup = run_results["upper_thresholds"]
            (uth_fig,) = plt.plot(x, thup, color=deep_saffron, lw=2, ls="dashed")
            fig.append(uth_fig)

        if "lower_thresholds" in K:
            thdown = run_results["lower_thresholds"]
            (lth_fig,) = plt.plot(x, thdown, color=deep_saffron, lw=2, ls="dashed")
            fig.append(lth_fig)

        if with_alarm and ("alarms" in K):
            alarm = run_results["alarms"]
            al_fig = plt.scatter(alarm, self.data[alarm], color="red")
            fig.append(al_fig)

        plt.xlim((0, self.data.size))

        return fig


"""
================================= WITH DRIFT ==================================
"""


def backMean(X, d):
    M = []
    w = X[:d].sum()
    M.append(w / d)
    for i in range(d, len(X)):
        w = w - X[i - d] + X[i]
        M.append(w / d)
    return np.array(M)


class dSPOT:
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper-bound)

    Attributes
    ----------
    proba : float
            Detection level (risk), chosen by the user

    depth : int
            Number of observations to compute the moving average

    extreme_quantile : float
            current threshold (bound between normal and abnormal events)

    data : numpy.array
            stream

    init_data : numpy.array
            initial batch of observations (for the calibration/initialization step)

    init_threshold : float
            initial threshold computed during the calibration step

    peaks : numpy.array
            array of peaks (excesses above the initial threshold)

    n : int
            number of observed values

    Nt : int
            number of observed peaks
    """

    def __init__(self, q, depth):
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0
        self.depth = depth

    def __str__(self):
        s = ""
        s += "Streaming Peaks-Over-Threshold Object\n"
        s += "Detection level q = %s\n" % self.proba
        if self.data is not None:
            s += "Data imported : Yes\n"
            s += "\t initialization  : %s values\n" % self.init_data.size
            s += "\t stream : %s values\n" % self.data.size
        else:
            s += "Data imported : No\n"
            return s

        if self.n == 0:
            s += "Algorithm initialized : No\n"
        else:
            s += "Algorithm initialized : Yes\n"
            s += "\t initial threshold : %s\n" % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += "Algorithm run : Yes\n"
                s += "\t number of observations : %s (%.2f %%)\n" % (
                    r,
                    100 * r / self.n,
                )
                s += "\t triggered alarms : %s (%.2f %%)\n" % (
                    len(self.alarm),
                    100 * len(self.alarm) / self.n,
                )
            else:
                s += "\t number of peaks  : %s\n" % self.Nt
                s += "\t extreme quantile : %s\n" % self.extreme_quantile
                s += "Algorithm run : No\n"
        return s

    def fit(self, init_data, data):
        """
        Import data to DSPOT object

        Parameters
        ----------
        init_data : list, numpy.array or pandas.Series
                initial batch to calibrate the algorithm

        data : numpy.array
                data for the run (list, np.array or pd.series)

        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print("This data format (%s) is not supported" % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print("The initial data cannot be set")
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data

        Parameters
        ----------
        data : list, numpy.array, pandas.Series
                data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print("This data format (%s) is not supported" % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, verbose=True):
        """
        Run the calibration (initialization) step

        Parameters
        ----------
        verbose : bool
                (default = True) If True, gives details about the batch initialization
        """
        n_init = self.init_data.size - self.depth

        M = backMean(self.init_data, self.depth)
        T = self.init_data[self.depth :] - M[:-1]  # new variable

        S = np.sort(T)  # we sort X to get the empirical quantile
        self.init_threshold = S[int(0.98 * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks = T[T > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print("Initial threshold : %s" % self.init_threshold)
            print("Number of peaks : %s" % self.Nt)
            print("Grimshaw maximum log-likelihood estimation ... ", end="")

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print("[done]")
            print("\t" + chr(0x03B3) + " = " + str(g))
            print("\t" + chr(0x03C3) + " = " + str(s))
            print("\tL = " + str(l))
            print("Extreme quantile (probability = %s): %s" % (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
                scalar function
        jac : function
                first order derivative of the function
        bounds : tuple
                (min,max) interval for the roots search
        npoints : int
                maximum number of roots to output
        method : str
                'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval

        Returns
        ----------
        numpy.array
                possible roots of the function
        """
        if method == "regular":
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == "random":
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(
            lambda X: objFun(X, fun, jac),
            X0,
            method="L-BFGS-B",
            jac=True,
            bounds=[bounds] * len(X0),
        )

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        Y : numpy.array
                observations
        gamma : float
                GPD index parameter
        sigma : float
                GPD scale parameter (>0)
        Returns
        ----------
        float
                log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
                numerical parameter to perform (default : 1e-8)
        n_points : int
                maximum number of candidates for maximum likelihood (default : 10)
        Returns
        ----------
        gamma_best,sigma_best,ll_best
                gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = SPOT._rootsFinder(
            lambda t: w(self.peaks, t),
            lambda t: jac_w(self.peaks, t),
            (a + epsilon, -epsilon),
            n_points,
            "regular",
        )

        right_zeros = SPOT._rootsFinder(
            lambda t: w(self.peaks, t),
            lambda t: jac_w(self.peaks, t),
            (b, c),
            n_points,
            "regular",
        )

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = dSPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q

        Parameters
        ----------
        gamma : float
                GPD parameter
        sigma : float
                GPD parameter
        Returns
        ----------
        float
                quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True):
        """
		Run biSPOT on the stream

		Parameters
		----------
		with_alarm : bool
			(default = True) If False, SPOT will adapt the threshold assuming \
			there is no abnormal values
		Returns
		----------
		dict
			keys : 'upper_thresholds', 'lower_thresholds' and 'alarms'

			'***-thresholds' contains the extreme quantiles and 'alarms' contains \
			the indexes of the values which have triggered alarms

		"""
        if self.n > self.init_data.size:
            print(
                "Warning : the algorithm seems to have already been run, you \
            should initialize before running again"
            )
            return {}

        # actual normal window
        W = self.init_data[-self.depth :]

        # list of the thresholds
        th = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):
            Mi = W.mean()
            # If the observed value exceeds the current threshold (alarm case)
            if (self.data[i] - Mi) > self.extreme_quantile:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks = np.append(self.peaks, self.data[i] - Mi - self.init_threshold)
                    self.Nt += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)  # + Mi
                    W = np.append(W[1:], self.data[i])

            # case where the value exceeds the initial threshold but not the alarm ones
            elif (self.data[i] - Mi) > self.init_threshold:
                # we add it in the peaks
                self.peaks = np.append(self.peaks, self.data[i] - Mi - self.init_threshold)
                self.Nt += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._grimshaw()
                self.extreme_quantile = self._quantile(g, s)  # + Mi
                W = np.append(W[1:], self.data[i])
            else:
                self.n += 1
                W = np.append(W[1:], self.data[i])

            th.append(self.extreme_quantile + Mi)  # thresholds record

        return {"thresholds": th, "alarms": alarm}

    def plot(self, run_results, with_alarm=True):
        """
        Plot the results given by the run

        Parameters
        ----------
        run_results : dict
                results given by the 'run' method
        with_alarm : bool
                (default = True) If True, alarms are plotted.
        Returns
        ----------
        list
                list of the plots

        """
        x = range(self.data.size)
        K = run_results.keys()

        (ts_fig,) = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        #        if 'upper_thresholds' in K:
        #            thup = run_results['upper_thresholds']
        #            uth_fig, = plt.plot(x,thup,color=deep_saffron,lw=2,ls='dashed')
        #            fig.append(uth_fig)
        #
        #        if 'lower_thresholds' in K:
        #            thdown = run_results['lower_thresholds']
        #            lth_fig, = plt.plot(x,thdown,color=deep_saffron,lw=2,ls='dashed')
        #            fig.append(lth_fig)

        if "thresholds" in K:
            th = run_results["thresholds"]
            (th_fig,) = plt.plot(x, th, color=deep_saffron, lw=2, ls="dashed")
            fig.append(th_fig)

        if with_alarm and ("alarms" in K):
            alarm = run_results["alarms"]
            if len(alarm) > 0:
                plt.scatter(alarm, self.data[alarm], color="red")

        plt.xlim((0, self.data.size))

        return fig


"""
=========================== DRIFT & DOUBLE BOUNDS =============================
"""


class bidSPOT:
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper and lower bounds)

    Attributes
    ----------
    proba : float
            Detection level (risk), chosen by the user

    depth : int
            Number of observations to compute the moving average

    extreme_quantile : float
            current threshold (bound between normal and abnormal events)

    data : numpy.array
            stream

    init_data : numpy.array
            initial batch of observations (for the calibration/initialization step)

    init_threshold : float
            initial threshold computed during the calibration step

    peaks : numpy.array
            array of peaks (excesses above the initial threshold)

    n : int
            number of observed values

    Nt : int
            number of observed peaks
    """

    def __init__(self, q=1e-4, depth=10):
        self.proba = q
        self.data = None
        self.init_data = None
        self.n = 0
        self.depth = depth

        nonedict = {"up": None, "down": None}

        self.extreme_quantile = dict.copy(nonedict)
        self.init_threshold = dict.copy(nonedict)
        self.peaks = dict.copy(nonedict)
        self.gamma = dict.copy(nonedict)
        self.sigma = dict.copy(nonedict)
        self.Nt = {"up": 0, "down": 0}

    def __str__(self):
        s = ""
        s += "Streaming Peaks-Over-Threshold Object\n"
        s += "Detection level q = %s\n" % self.proba
        if self.data is not None:
            s += "Data imported : Yes\n"
            s += "\t initialization  : %s values\n" % self.init_data.size
            s += "\t stream : %s values\n" % self.data.size
        else:
            s += "Data imported : No\n"
            return s

        if self.n == 0:
            s += "Algorithm initialized : No\n"
        else:
            s += "Algorithm initialized : Yes\n"
            s += "\t initial threshold : %s\n" % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += "Algorithm run : Yes\n"
                s += "\t number of observations : %s (%.2f %%)\n" % (
                    r,
                    100 * r / self.n,
                )
                s += "\t triggered alarms : %s (%.2f %%)\n" % (
                    len(self.alarm),
                    100 * len(self.alarm) / self.n,
                )
            else:
                s += "\t number of peaks  : %s\n" % self.Nt
                s += "\t upper extreme quantile : %s\n" % self.extreme_quantile["up"]
                s += "\t lower extreme quantile : %s\n" % self.extreme_quantile["down"]
                s += "Algorithm run : No\n"
        return s

    def fit(self, init_data, data):
        """
        Import data to biDSPOT object

        Parameters
        ----------
        init_data : list, numpy.array or pandas.Series
                initial batch to calibrate the algorithm

        data : numpy.array
                data for the run (list, np.array or pd.series)

        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print("This data format (%s) is not supported" % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print("The initial data cannot be set")
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data

        Parameters
        ----------
        data : list, numpy.array, pandas.Series
                data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print("This data format (%s) is not supported" % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, verbose=True):
        """
        Run the calibration (initialization) step

        Parameters
        ----------
        verbose : bool
                (default = True) If True, gives details about the batch initialization
        """
        n_init = self.init_data.size - self.depth

        M = backMean(self.init_data, self.depth)
        T = self.init_data[self.depth :] - M[:-1]  # new variable

        S = np.sort(T)  # we sort T to get the empirical quantile
        self.init_threshold["up"] = S[int(0.98 * n_init)]  # t is fixed for the whole algorithm
        self.init_threshold["down"] = S[int(0.02 * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks["up"] = T[T > self.init_threshold["up"]] - self.init_threshold["up"]
        self.peaks["down"] = -(T[T < self.init_threshold["down"]] - self.init_threshold["down"])
        self.Nt["up"] = self.peaks["up"].size
        self.Nt["down"] = self.peaks["down"].size
        self.n = n_init

        if verbose:
            print("Initial threshold : %s" % self.init_threshold)
            print("Number of peaks : %s" % self.Nt)
            print("Grimshaw maximum log-likelihood estimation ... ", end="")

        l = {"up": None, "down": None}
        for side in ["up", "down"]:
            g, s, l[side] = self._grimshaw(side)
            self.extreme_quantile[side] = self._quantile(side, g, s)
            self.gamma[side] = g
            self.sigma[side] = s

        ltab = 20
        form = "\t" + "%20s" + "%20.2f" + "%20.2f"
        if verbose:
            print("[done]")
            print("\t" + "Parameters".rjust(ltab) + "Upper".rjust(ltab) + "Lower".rjust(ltab))
            print("\t" + "-" * ltab * 3)
            print(form % (chr(0x03B3), self.gamma["up"], self.gamma["down"]))
            print(form % (chr(0x03C3), self.sigma["up"], self.sigma["down"]))
            print(form % ("likelihood", l["up"], l["down"]))
            print(
                form
                % (
                    "Extreme quantile",
                    self.extreme_quantile["up"],
                    self.extreme_quantile["down"],
                )
            )
            print("\t" + "-" * ltab * 3)
        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
                scalar function
        jac : function
                first order derivative of the function
        bounds : tuple
                (min,max) interval for the roots search
        npoints : int
                maximum number of roots to output
        method : str
                'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval

        Returns
        ----------
        numpy.array
                possible roots of the function
        """
        if method == "regular":
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == "random":
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(
            lambda X: objFun(X, fun, jac),
            X0,
            method="L-BFGS-B",
            jac=True,
            bounds=[bounds] * len(X0),
        )

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        Y : numpy.array
                observations
        gamma : float
                GPD index parameter
        sigma : float
                GPD scale parameter (>0)
        Returns
        ----------
        float
                log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, side, epsilon=1e-8, n_points=8):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
                numerical parameter to perform (default : 1e-8)
        n_points : int
                maximum number of candidates for maximum likelihood (default : 10)
        Returns
        ----------
        gamma_best,sigma_best,ll_best
                gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks[side].min()
        YM = self.peaks[side].max()
        Ymean = self.peaks[side].mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = bidSPOT._rootsFinder(
            lambda t: w(self.peaks[side], t),
            lambda t: jac_w(self.peaks[side], t),
            (a + epsilon, -epsilon),
            n_points,
            "regular",
        )

        right_zeros = bidSPOT._rootsFinder(
            lambda t: w(self.peaks[side], t),
            lambda t: jac_w(self.peaks[side], t),
            (b, c),
            n_points,
            "regular",
        )

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = bidSPOT._log_likelihood(self.peaks[side], gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks[side]) - 1
            sigma = gamma / z
            ll = bidSPOT._log_likelihood(self.peaks[side], gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, side, gamma, sigma):
        """
        Compute the quantile at level 1-q for a given side

        Parameters
        ----------
        side : str
                'up' or 'down'
        gamma : float
                GPD parameter
        sigma : float
                GPD parameter
        Returns
        ----------
        float
                quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        if side == "up":
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold["up"] + (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold["up"] - sigma * log(r)
        elif side == "down":
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold["down"] - (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold["down"] + sigma * log(r)
        else:
            print("error : the side is not right")

    def run(self, with_alarm=True, plot=True):
        """
		Run biDSPOT on the stream

		Parameters
		----------
		with_alarm : bool
			(default = True) If False, SPOT will adapt the threshold assuming \
			there is no abnormal values
		Returns
		----------
		dict
			keys : 'upper_thresholds', 'lower_thresholds' and 'alarms'

			'***-thresholds' contains the extreme quantiles and 'alarms' contains \
			the indexes of the values which have triggered alarms

		"""
        if self.n > self.init_data.size:
            print(
                "Warning : the algorithm seems to have already been run, you \
            should initialize before running again"
            )
            return {}

        # actual normal window
        W = self.init_data[-self.depth :]

        # list of the thresholds
        thup = []
        thdown = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):
            Mi = W.mean()
            Ni = self.data[i] - Mi
            # If the observed value exceeds the current threshold (alarm case)
            if Ni > self.extreme_quantile["up"]:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks["up"] = np.append(self.peaks["up"], Ni - self.init_threshold["up"])
                    self.Nt["up"] += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw("up")
                    self.extreme_quantile["up"] = self._quantile("up", g, s)
                    W = np.append(W[1:], self.data[i])

            # case where the value exceeds the initial threshold but not the alarm ones
            elif Ni > self.init_threshold["up"]:
                # we add it in the peaks
                self.peaks["up"] = np.append(self.peaks["up"], Ni - self.init_threshold["up"])
                self.Nt["up"] += 1
                self.n += 1
                # and we update the thresholds
                g, s, l = self._grimshaw("up")
                self.extreme_quantile["up"] = self._quantile("up", g, s)
                W = np.append(W[1:], self.data[i])

            elif Ni < self.extreme_quantile["down"]:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks["down"] = np.append(self.peaks["down"], -(Ni - self.init_threshold["down"]))
                    self.Nt["down"] += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw("down")
                    self.extreme_quantile["down"] = self._quantile("down", g, s)
                    W = np.append(W[1:], self.data[i])

            # case where the value exceeds the initial threshold but not the alarm ones
            elif Ni < self.init_threshold["down"]:
                # we add it in the peaks
                self.peaks["down"] = np.append(self.peaks["down"], -(Ni - self.init_threshold["down"]))
                self.Nt["down"] += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._grimshaw("down")
                self.extreme_quantile["down"] = self._quantile("down", g, s)
                W = np.append(W[1:], self.data[i])
            else:
                self.n += 1
                W = np.append(W[1:], self.data[i])

            thup.append(self.extreme_quantile["up"] + Mi)  # upper thresholds record
            thdown.append(self.extreme_quantile["down"] + Mi)  # lower thresholds record

        return {"upper_thresholds": thup, "lower_thresholds": thdown, "alarms": alarm}

    def plot(self, run_results, with_alarm=True):
        """
        Plot the results given by the run

        Parameters
        ----------
        run_results : dict
                results given by the 'run' method
        with_alarm : bool
                (default = True) If True, alarms are plotted.
        Returns
        ----------
        list
                list of the plots

        """
        x = range(self.data.size)
        K = run_results.keys()

        (ts_fig,) = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if "upper_thresholds" in K:
            thup = run_results["upper_thresholds"]
            (uth_fig,) = plt.plot(x, thup, color=deep_saffron, lw=2, ls="dashed")
            fig.append(uth_fig)

        if "lower_thresholds" in K:
            thdown = run_results["lower_thresholds"]
            (lth_fig,) = plt.plot(x, thdown, color=deep_saffron, lw=2, ls="dashed")
            fig.append(lth_fig)

        if with_alarm and ("alarms" in K):
            alarm = run_results["alarms"]
            if len(alarm) > 0:
                al_fig = plt.scatter(alarm, self.data[alarm], color="red")
                fig.append(al_fig)

        plt.xlim((0, self.data.size))

        return fig
    

import numpy as np
import more_itertools as mit


def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                    A point is labeled as "anomaly" if its score is lower than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):
    Returns:
            np.ndarray: predict labels

    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    if label is None:
        predict = score > threshold
        return predict, None

    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    for i in range(len(predict)):
        if any(actual[max(i, 0) : i + 1]) and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def pot_eval(init_score, score, label, q=1e-3, level=0.99, dynamic=False):
    """
    Run POT method on given score.
    :param init_score (np.ndarray): The data to get init threshold.
                    For `OmniAnomaly`, it should be the anomaly score of train set.
    :param: score (np.ndarray): The data to run POT method.
                    For `OmniAnomaly`, it should be the anomaly score of test set.
    :param label (np.ndarray): boolean list of true anomalies in score
    :param q (float): Detection level (risk)
    :param level (float): Probability associated with the initial threshold t
    :return dict: pot result dict
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Running POT with q={q}, level={level}..")
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=False)  # Calibration step
    ret = s.run(dynamic=dynamic, with_alarm=False)

    print(len(ret["alarms"]))
    print(len(ret["thresholds"]))

    pot_th = np.mean(ret["thresholds"])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    if label is not None:
        p_t = calc_point2point(pred, label)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": pot_th,
            "latency": p_latency,
        }
    else:
        return {
            "threshold": pot_th,
        }


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Finding best f1-score by searching for threshold..")
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1.0, -1.0, -1.0)
    m_t = 0.0
    m_l = 0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target, latency = calc_seq(score, label, threshold)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            m_l = latency
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)

    return {
        "f1": m[0],
        "precision": m[1],
        "recall": m[2],
        "TP": m[3],
        "TN": m[4],
        "FP": m[5],
        "FN": m[6],
        "threshold": m_t,
        "latency": m_l,
    }


def calc_seq(score, label, threshold):
    predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    return calc_point2point(predict, label), latency


def epsilon_eval(train_scores, test_scores, test_labels, reg_level=1):
    best_epsilon = find_epsilon(train_scores, reg_level)
    pred, p_latency = adjust_predicts(test_scores, test_labels, best_epsilon, calc_latency=True)
    if test_labels is not None:
        p_t = calc_point2point(pred, test_labels)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": best_epsilon,
            "latency": p_latency,
            "reg_level": reg_level,
        }
    else:
        return {"threshold": best_epsilon, "reg_level": reg_level}


def find_epsilon(errors, reg_level=1):
    """
    Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
    Code from TelemAnom (https://github.com/khundman/telemanom)
    """
    e_s = errors
    best_epsilon = None
    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)

    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]
            # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon


import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)
    
import torch
import torch.nn as nn

# from modules import (
#     ConvLayer,
#     FeatureAttentionLayer,
#     TemporalAttentionLayer,
#     GRULayer,
#     Forecasting_Model,
#     ReconstructionModel,
# )


class MTAD_GAT(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2
    ):
        super(MTAD_GAT, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features

        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)

        return predictions, recons
    
    
class MTAD_GAT_RECON(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2
    ):
        super(MTAD_GAT_RECON, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features

        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp

        recons = self.recon_model(h_end)

        return recons