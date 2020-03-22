import numpy as np
from ptemcee.sampler import make_ladder
import numpy as np
from numpy.random.mtrand import RandomState
from .sampler import Sampler
from scipy.stats import norm
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
from datetime import datetime
import time
import numpy as np

import h5py
from ptemcee import Sampler as PTSampler
from shutil import copy

now = datetime.now()


class Logl:
    def __init__(self, likelihood):
        self.likelihood = likelihood
        self.num_per_call = likelihood.ndevices * likelihood.nwalkers

    def __call__(self, x_all):
        if (x_all.shape[0] % self.num_per_call) != 0:
            raise ShapeError("Array shape must be integer multiple of num_per_call.")

        splits = x_all.shape[0] // self.num_per_call
        LL = np.concatenate(
            [-self.likelihood.getNLL(x.T) for x in np.split(x_all, splits)]
        )
        return LL


class Logp:
    def __init__(
        self,
        recycler,
        key_order,
        sampling_values,
        fix_spins=False,
        emri_fix_phases=False,
    ):
        self.recycler = recycler
        self.fix_spins = fix_spins
        self.emri_fix_phases = emri_fix_phases
        self.key_order = key_order
        self.sampling_values = sampling_values

    def __call__(self, x):
        prior_val = np.zeros(x.shape[0])
        x = self.recycler.recycle(x.T).T
        if self.fix_spins:
            x[:, 2] = 0.0
            x[:, 3] = 0.0
        if self.emri_fix_phases:
            x[:, 6] = 0.0
            x[:, 7] = 0.0
            x[:, 8] = 0.0
        for i, key in enumerate(self.key_order):
            distribution = self.sampling_values[key].distribution
            prior_trans = distribution.logpdf(x[:, i])  # no negative here
            prior_val[np.isinf(prior_trans)] = -np.inf

        return prior_val


class PTMCMC(Sampler):
    def __init__(self, likelihood, parameters, nwalkers, ntemps, **kwargs):

        prop_default = {
            "steps_per_emcee_run": 1000,
            "burn_steps": 5000,
            "proposals": {},
            "fix_spins": False,
            "emri_fix_phases": False,
            "start_at_all_modes": True,
            "sigma_factor": 1.0,
            "progress": False,
            "ndim": 11,
            "Tmax": np.inf,
            "scale_factor": 1.15,
            "start_at_random_prior_point": False,
        }

        self.likelihood = likelihood
        self.nwalkers = nwalkers
        self.ntemps = ntemps

        if (self.ntemps % 2) != 0:
            print("Warning: Changing ntemps up 1 to make it even.")
            self.ntemps += 1

        for prop, default in prop_default.items():
            setattr(self, prop, kwargs.get(prop, default))

        super().__init__(parameters, **kwargs)

        current_point_dict = self.get_current_point()
        self.current_point = np.array(
            [current_point_dict[key] for key in self.key_order]
        )
        self.injection = self.current_point.copy()

        # self.m_mu = self.likelihood.get_Mij(self.current_point)
        # self.m_mu = self.likelihood.get_Mij_with_hij(self.start_point)
        self.cov = np.linalg.pinv(self.likelihood.get_Fisher(self.current_point))
        # self.m_mu  = self.likelihood.get_Fisher(self.current_point).diagonal()
        # self.s_mu = self.m_mu**(-1./2.)

        current_point = []
        if self.start_at_random_prior_point:
            self.current_point = np.zeros((self.nwalkers * self.ntemps, self.ndim))
            for i, key in enumerate(self.key_order):
                vec = self.sampling_values[key].distribution.rvs(
                    self.nwalkers * self.ntemps
                )
                self.current_point[:, i] = vec

        else:
            self.current_point = self.current_point + np.random.multivariate_normal(
                np.zeros(len(self.test_inds)),
                self.sigma_factor * self.cov,
                size=self.nwalkers * self.ntemps,
            )

        if self.start_at_all_modes and self.start_at_random_prior_point is False:
            num_repeats = int(self.nwalkers * self.ntemps / 8)
            long_mode = np.repeat(np.array([0, 1, 2, 3]) * np.pi / 2, 2 * num_repeats)
            lat_mode = np.tile(np.array([-1, 1]), 4 * num_repeats)

            self.current_point[:, 7] += long_mode
            # must change psi as well
            self.current_point[:, 9] += long_mode
            self.current_point[:, 7] = self.current_point[:, 7] % (2 * np.pi)
            self.current_point[:, 8] *= lat_mode
            self.current_point[:, 9] = self.current_point[:, 9] * (lat_mode == 1) + (
                np.pi - self.current_point[:, 9]
            ) * (lat_mode == -1)
            self.current_point[:, 6] *= lat_mode

        self.current_point = self.current_point.reshape(
            self.ntemps, self.nwalkers, self.ndim
        )
        self.start_point = self.current_point.copy()

        self.logl = Logl(self.likelihood)
        self.logp = Logp(
            self.likelihood.recycler,
            self.key_order,
            self.sampling_values,
            self.fix_spins,
            self.emri_fix_phases,
        )

        self.sampler = PTSampler(
            self.nwalkers,
            self.ndim,
            self.logl,
            self.logp,
            adaptive=True,
            betas=make_ladder(self.ndim, self.ntemps, Tmax=self.Tmax),
            scale_factor=self.scale_factor,
        )

    def _setup_place_holders(self):
        self.chain_store = np.zeros(
            (
                self.steps_per_emcee_run * self.total_steps,
                self.ntemps,
                self.nwalkers,
                (self.ndim + 1),
            )
        )

        self.acceptance_store = np.zeros((self.total_steps, self.ntemps, self.nwalkers))
        self.acceptance_store = np.zeros((self.total_steps, self.ntemps, self.nwalkers))
        self.swap_acceptance_store = np.zeros((self.total_steps, self.ntemps - 1))

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

    def step(self):

        if self.main_step_num == 0:
            seed = np.random.randint(100000)
            self.chain = self.sampler.chain(self.current_point, RandomState(seed))
            self.chain.run(self.burn_steps)
            self.current_point = self.chain.x[-1]
            print("burn end")
            seed = np.random.randint(100000)
            self.chain = self.sampler.chain(self.current_point, RandomState(seed))

        self.chain.run(self.steps_per_emcee_run)
        self.current_point = self.chain.x[-1]

        start = (self.main_step_num) * self.steps_per_emcee_run
        end = (self.main_step_num + 1) * self.steps_per_emcee_run

        self.acceptance_store[self.main_step_num] = self.chain.jump_acceptance_ratio
        self.swap_acceptance_store[
            self.main_step_num
        ] = self.chain.swap_acceptance_ratio

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
                num_points = (step + 1) * self.steps_per_emcee_run
                temp = np.concatenate(
                    [self.chain.x, self.chain.logl[:, :, :, np.newaxis]], axis=-1
                )
                with h5py.File(self.chain_file, "w") as f:
                    f.create_dataset(
                        "samples",
                        data=temp,
                        shape=temp.shape,
                        dtype=temp.dtype,
                        compression="gzip",
                        compression_opts=9,
                    )
                    f.create_dataset(
                        "af",
                        data=self.acceptance_store[: step + 1],
                        shape=self.acceptance_store[: step + 1].shape,
                        dtype=self.acceptance_store[: step + 1].dtype,
                        compression="gzip",
                        compression_opts=9,
                    )
                    f.create_dataset(
                        "saf",
                        data=self.swap_acceptance_store[: step + 1],
                        shape=self.swap_acceptance_store[: step + 1].shape,
                        dtype=self.swap_acceptance_store[: step + 1].dtype,
                        compression="gzip",
                        compression_opts=9,
                    )
                    f.attrs["header"] = self.chain_header

                copy(self.chain_file, self.chain_file[:-5] + "_backup.hdf5")
