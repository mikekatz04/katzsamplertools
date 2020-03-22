import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import h5py

now = datetime.now()


class Chain:
    def __init__(self, name, initial_point, distribution, **kwargs):
        self.name = name
        self.initial_point = initial_point
        self.distribution = distribution

        self.current_value = initial_point
        self.chain = [initial_point]

    def update_chain(self, new_point):
        self.chain.append(new_point)
        self.current_value = new_point


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
        self.test_inds = np.asarray(
            [i for i, item in enumerate(parameters.values()) if isinstance(item, dict)]
        )

        self.num_test_inds = len(self.test_inds)
        print("Testing indices (from parameters):", self.test_inds)

        self.acceptance_fraction = []
        self.current_af = 0.0
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
                self.sampling_values[key] = Chain(
                    key, item["initial_point"], item["prior"]
                )

            else:
                raise ValueError(
                    "item in the prior dictionary must be a float"
                    + "(float64) or a dictionary."
                )

        self.dim = len(self.sampling_values.keys())

    def _setup_place_holders(self):
        self.chain_store = np.zeros((self.total_steps, len(self.test_inds)))
        self.acceptance_store = np.zeros((self.total_steps,), dtype=int)

        self.acceptance_header = now.strftime("%Y_%m_%d_%H_%M_%S") + "\n"
        self.acceptance_header += "Acceptance info from HMC evaluation\n"
        self.acceptance_header += "------------------------\n"

        self.chain_header = now.strftime("%Y_%m_%d_%H_%M_%S") + "\n"
        self.chain_header += "Chains from HMC evaluation\n"
        for key, item in self.fixed_values.items():
            self.chain_header += key + ": " + "{:.18e}".format(item) + "\n"
        self.chain_header += "------------------------\n"
        for key in [self.key_order[i] for i in self.test_inds]:
            self.chain_header += key + "\t"

    def run(self):
        iter = (
            tqdm(range(self.total_steps), desc="HMC loop")
            if self.use_tqdm
            else range(self.total_steps)
        )
        for step in iter:
            self.main_step_num = step
            self.step()
            if (step + 1) % self.steps_per_output == 0:
                print("Saving after {} steps.".format(step + 1))
                num_points = (step + 1) * self.steps_per_output
                with h5py.File(self.chain_file, "w") as f:
                    f.create_dataset(
                        "samples",
                        data=self.chain_store[:, :num_points, :],
                        shape=self.chain_store[:, :num_points, :].shape,
                        dtype=self.chain_store[:, :num_points, :].dtype,
                        compression="gzip",
                        compression_opts=9,
                    )
                    f.create_dataset(
                        "af",
                        data=self.acceptance_store[:num_points],
                        shape=self.acceptance_store[:num_points].shape,
                        dtype=self.acceptance_store[:num_points].dtype,
                        compression="gzip",
                        compression_opts=9,
                    )
                    f.attrs["header"] = self.chain_header

                if self.email_for_updates is not None:
                    run_update(
                        self.main_step_num + 1,
                        self.file_dir_short,
                        [self.key_order[k] for k in self.test_inds],
                        output_file_location=self.output_file_location,
                        receiver_email=self.email_for_updates,
                    )

    def _update_chains(self, parameter_space_location):
        for key, chain in self.sampling_values.items():
            chain.update_chain(parameter_space_location[key])

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
