import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import h5py

now = datetime.now()


class PriorContainer:
    def __init__(self, name, initial_point, distribution, **kwargs):
        self.name = name
        self.initial_point = initial_point
        self.distribution = distribution


# class Sampler(PlotSamples):
class Sampler:
    def __init__(self, parameters, **kwargs):
        """
        priors - dict
        initial_point - ordered dict
        """

        prop_defaults = {
            "chain_file": "chain_output.hdf5",
            "sub_folder": None,
            "acceptance_file": "acceptance_output.txt",
            "folder_prefix": "hmc",
            "steps_per_output": 1000,
            "total_steps": 100,
            "verbose": False,
            "use_tqdm": False,
            "email_for_updates": None,
            "output_file_location": "/home/mlk667/sampler_tmp/",
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        outer_folder = "output_files/"
        if self.sub_folder is not None:
            try:
                os.mkdir(outer_folder + self.sub_folder)
            except FileExistsError:
                pass
            outer_folder = outer_folder + self.sub_folder + "/"
        self.file_dir = (
            outer_folder
            + self.folder_prefix
            + "_"
            + now.strftime("%Y_%m_%d_%H_%M_%S")
            + "/"
        )
        os.mkdir(self.file_dir)

        self.file_dir_short = (
            self.folder_prefix + "_" + now.strftime("%Y_%m_%d_%H_%M_%S")
        )

        for name in ["chain", "acceptance"]:
            setattr(self, name + "_file", self.file_dir + getattr(self, name + "_file"))

        self.parameters = parameters

        self.num_steps = 0
        self.num_proposals = 0
        self.key_order = list(parameters.keys())

        self._setup_priors()
        self._setup_place_holders()

    def _setup_priors(self):

        # setup prior and fixed dictionaries
        self.sampling_values = {}
        self.fixed_values = {}
        for key, item in self.parameters.items():
            if isinstance(item, float):
                self.fixed_values[key] = item

            elif isinstance(item, dict):
                if "initial_point" not in item:
                    raise ValueError(
                        "Parameter with a prior must have an initial_point."
                    )

                if "prior" not in item:
                    raise ValueError(
                        "Parameter with a prior must have a prior distribution."
                    )
                # make sure sampling_keys and initial_point keys are the same
                self.sampling_values[key] = PriorContainer(
                    key, item["initial_point"], item["prior"]
                )

            else:
                raise ValueError(
                    "item in the prior dictionary must be a float"
                    + "(float64) or a dictionary."
                )

        self.dim = len(self.sampling_values.keys())

    def get_current_point(self):
        fixed_points = {key: item for key, item in self.fixed_values.items()}
        sampling_points = {
            key: item.current_value for key, item in self.sampling_values.items()
        }
        return {**fixed_points, **sampling_points}

    def get_prior_value(self, x, return_all=False):
        prior_val = np.zeros_like(x)
        for i, ind in enumerate(self.test_inds):
            distribution = self.sampling_values[self.key_order[ind]].distribution
            prior_val[i] = -distribution.logpdf(x[i])

        if return_all:
            return prior_val
        return np.sum(prior_val)
