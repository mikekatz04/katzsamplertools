import numpy as np
import pdb
from collections import OrderedDict
import time
import argparse
from shutil import copyfile


def run(args):
    # MUST BE IN SAME ORDER AS WAVE FUNCTION CALL
    seed_val = np.random.randint(1000)
    np.random.seed(seed_val)
    print("seed:", seed_val)

    parameters = OrderedDict([(key, None) for key in key_order])

    for key in parameters.keys():
        if key in priors:
            parameters[key] = {}
            parameters[key]["initial_point"] = initial_point[key]
            parameters[key]["prior"] = priors[key]

        else:
            parameters[key] = initial_point[key]

    indices_to_test = np.asarray([list(parameters.keys()).index(key) for key in priors])

    likelihood_kwargs["test_inds"] = indices_to_test
    likelihood_kwargs["num_params"] = len(parameters.keys())

    likelihood = likelihood_class(*likelihood_args, **likelihood_kwargs)

    sampler = sampler_class(
        likelihood,
        parameters,
        *specific_sampler_args,
        **{**sampler_kwargs, **specific_sampler_kwargs}
    )
    print("transfer files")
    for config in [
        args.sampler_settings,
        args.main_settings,
        args.binary,
        args.likelihood_settings,
    ]:
        copyfile(config, sampler.file_dir + config)

    try:
        np.save(sampler.file_dir + "data_freqs", likelihood.data_freqs)
    except AttributeError:
        pass

    out_data_stream = []
    for i in range(3):
        try:
            out_data_stream.append(getattr(likelihood, "data_channel{}".format(i + 1)))
        except AttributeError:
            pass
    np.save(sampler.file_dir + "data_stream", np.asarray(out_data_stream))

    st = time.perf_counter()
    sampler.run()
    print(time.perf_counter() - st, "sec")
    # sampler.plot_2d_walk('ln_mT', 'mr')
    # sampler.plot_chains()
    # sampler.make_corner()
    # pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_settings",
        "-ms",
        help="Add .py configuration file for main settings.",
        default="main_config.py",
    )
    parser.add_argument(
        "--sampler_settings",
        "-ss",
        help="Add .py configuration file for sampler settings.",
        required=True,
    )
    parser.add_argument(
        "--likelihood_settings",
        "-ls",
        help="Add .py configuration file for likelihood settings.",
        default="hmlike_config.py",
    )
    parser.add_argument(
        "--binary", "-b", help="Add .py configuration file for binary.", required=True
    )
    parser.add_argument(
        "--add_noise", "-an", help="add noise", action="store_true", default=False
    )
    parser.add_argument(
        "--sampling_frequency",
        "-fs",
        type=float,
        help="sampling frequency for noise",
        default=0.3,
    )
    parser.add_argument(
        "--minimum_frequency",
        "-mf",
        type=float,
        help="minimum frequency for noise data set.",
        default=1e-7,
    )
    args = parser.parse_args()

    for config in [
        args.sampler_settings,
        args.main_settings,
        args.binary,
        args.likelihood_settings,
    ]:
        exec("from {} import *".format(config[:-3]))

    if args.add_noise:
        likelihood_kwargs["add_noise"] = {
            "fs": args.sampling_frequency,
            "min_freq": args.minimum_frequency,
        }

        # TODO need to add non log spaced (does not change anything really)

    if likelihood_kwargs["data_params"] is None:
        likelihood_kwargs["data_params"] = initial_point

    run(args)
