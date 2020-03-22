import numpy as np
from .sampler import Sampler
from scipy.stats import norm
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
from datetime import datetime
import time
import numpy as np
import dynesty
import pickle


now = datetime.now()


def loglike(x, phenomhm):
    LL = -phenomhm.getNLL(x.T)
    return LL


def ptform(u, sampling_values, key_order, test_inds):
    prior_val = np.zeros_like(u)
    for i, ind in enumerate(test_inds):
        distribution = sampling_values[key_order[ind]].distribution
        loc, scale = distribution.args
        prior_val[i] = scale * u[i] + loc
    return prior_val


class DynestySampler(Sampler):
    def __init__(self, likelihood, parameters, **kwargs):

        prop_default = {
            "nlive": 1000,
            "bound": "single",
            "which_sampler": "dynamic",
            "run_kwargs": {},
        }

        self.likelihood = likelihood

        for prop, default in prop_default.items():
            setattr(self, prop, kwargs.get(prop, default))

        for prop, default in prop_default.items():
            kwargs[prop] = kwargs.get(prop, default)

        super().__init__(parameters, **kwargs)

        current_point_dict = self.get_current_point()
        self.current_point = np.array(
            [current_point_dict[key] for key in self.key_order]
        )
        self.injection = self.current_point.copy()

        if self.which_sampler == "dynamic":
            print("Running dynamic sampler.")
            self.sampler = dynesty.DynamicNestedSampler(
                loglike,
                ptform,
                len(self.test_inds),
                logl_args=(self.likelihood,),
                ptform_args=(self.sampling_values, self.key_order, self.test_inds),
                **kwargs
            )
        elif self.which_sampler == "static":
            print("Running static sampler.")
            self.sampler = dynesty.NestedSampler(
                loglike,
                ptform,
                len(self.test_inds),
                logl_args=(self.likelihood,),
                ptform_args=(self.sampling_values, self.key_order, self.test_inds),
                **kwargs
            )
        else:
            raise ValueError("which_sampler must be dynamic or static.")

    def _setup_place_holders(self):
        pass
        """self.chain_store = np.zeros((self.steps_per_emcee_run*self.steps_per_output, (len(self.test_inds) + 1) *self.nwalkers))

        self.acceptance_store = np.zeros((self.total_steps, self.nwalkers,))

        self.acceptance_header = now.strftime("%Y_%m_%d_%H_%M_%S") + '\n'
        self.acceptance_header += 'Acceptance info from HMC evaluation\n'
        self.acceptance_header += '------------------------\n'


        self.chain_header = now.strftime("%Y_%m_%d_%H_%M_%S") + '\n'
        self.chain_header += 'Chains from HMC evaluation\n'
        for key, item in self.fixed_values.items():
            self.chain_header += key + ': ' + '{:.18e}'.format(item) + '\n'
        self.chain_header += '------------------------\n'
        for key in [self.key_order[i] for i in self.test_inds]:
            self.chain_header += key + '\t'"""

    def run(self):
        file_name = self.file_dir + "results_dict.pickle"
        key_defaults = dict(
            nlive_init=500,
            maxiter_init=None,
            maxcall_init=None,
            dlogz_init=0.01,
            logl_max_init=np.inf,
            nlive_batch=500,
            wt_function=None,
            wt_kwargs=None,
            maxiter_batch=None,
            maxcall_batch=None,
            maxiter=None,
            maxcall=None,
            maxbatch=None,
            stop_function=None,
            stop_kwargs=None,
            use_stop=True,
            save_bounds=True,
            print_progress=True,
            print_func=None,
            live_points=None,
            read_out=file_name,
        )

        for key, item in key_defaults.items():
            self.run_kwargs[key] = self.run_kwargs.get(key, item)

        print("run_kwargs", self.run_kwargs)
        self.sampler.run_nested(**self.run_kwargs)
        results = self.sampler.results
        with open(file_name, "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("DUMPED TO FILE {}".format(file_name))
        try:
            print(results.summary())
        except:
            pass
