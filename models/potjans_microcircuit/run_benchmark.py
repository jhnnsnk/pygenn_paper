import numpy as np 
import matplotlib.pyplot as plt 

from pygenn import genn_model, genn_wrapper
from scipy.stats import norm
from six import itervalues

from time import perf_counter_ns
from argparse import ArgumentParser
from pathlib import Path
from json import dump, dumps

# Get and check file path
parser = ArgumentParser()
parser.add_argument("file", type=str)
parser.add_argument("--path", type=str, default=None)
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

data_path = Path(args.path)

file_name = args.file + ".json"
file_path = data_path / file_name
assert data_path.is_dir() and not file_path.exists()

print(f"Arguments: {args}")

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
# Layer names
LAYER_NAMES = ["23", "4", "5", "6"]

# Population names
POPULATION_NAMES = ["E", "I"]

# Simulation timestep [ms]
DT_MS = 0.1

# Simulation duration [ms]
DURATION_MS = 10000.0

# set to false because we use our timers
# Should kernel timing be measured?
MEASURE_TIMING = True

# Should we use procedural rather than in-memory connectivity?
PROCEDURAL_CONNECTIVITY = False

# Should we rebuild the model rather than loading previous version
BUILD_MODEL = True

# Should we use zero-copy for legacy spike recording?
ZERO_COPY = False

# Should we use GeNN's built-in recording system
USE_GENN_RECORDING = False

# How many threads to use per spike for procedural connectivity?
NUM_THREADS_PER_SPIKE = 8

# Scaling factors for number of neurons and synapses
NEURON_SCALING_FACTOR = 1.0
CONNECTIVITY_SCALING_FACTOR = 1.0

# Background rate per synapse
BACKGROUND_RATE = 8.0  # spikes/s

# Relative inhibitory synaptic weight
G = -4.0

# Mean synaptic weight for all excitatory projections except L4e->L2/3e
MEAN_W = 87.8e-3  # nA
EXTERNAL_W = 87.8e-3   # nA

# Mean synaptic weight for L4e->L2/3e connections
# See p. 801 of the paper, second paragraph under 'Model Parameterization',
# and the caption to Supplementary Fig. 7
LAYER_23_4_W = 2.0 * MEAN_W   # nA

# Standard deviation of weight distribution relative to mean for
# all projections except L4e->L2/3e
REL_W = 0.1

# Standard deviation of weight distribution relative to mean for L4e->L2/3e
# This value is not mentioned in the paper, but is chosen to match the
# original code by Tobias Potjans
LAYER_23_4_RELW = 0.05

# Numbers of neurons in full-scale model
NUM_NEURONS = {
    "23":   {"E":20683, "I": 5834},
    "4":    {"E":21915, "I": 5479},
    "5":    {"E":4850,  "I": 1065},
    "6":    {"E":14395, "I": 2948}}

# Probabilities for >=1 connection between neurons in the given populations.
# The first index is for the target population; the second for the source population
CONNECTION_PROBABILTIES = {
    "23E":  {"23E": 0.1009, "23I": 0.1689,  "4E": 0.0437,   "4I": 0.0818,   "5E": 0.0323,   "5I": 0.0,      "6E": 0.0076,   "6I": 0.0},
    "23I":  {"23E": 0.1346, "23I": 0.1371,  "4E": 0.0316,   "4I": 0.0515,   "5E": 0.0755,   "5I": 0.0,      "6E": 0.0042,   "6I": 0.0},
    "4E":   {"23E": 0.0077, "23I": 0.0059,  "4E": 0.0497,   "4I": 0.135,    "5E": 0.0067,   "5I": 0.0003,   "6E": 0.0453,   "6I": 0.0},
    "4I":   {"23E": 0.0691, "23I": 0.0029,  "4E": 0.0794,   "4I": 0.1597,   "5E": 0.0033,   "5I": 0.0,      "6E": 0.1057,   "6I": 0.0},
    "5E":   {"23E": 0.1004, "23I": 0.0622,  "4E": 0.0505,   "4I": 0.0057,   "5E": 0.0831,   "5I": 0.3726,   "6E": 0.0204,   "6I": 0.0},
    "5I":   {"23E": 0.0548, "23I": 0.0269,  "4E": 0.0257,   "4I": 0.0022,   "5E": 0.06,     "5I": 0.3158,   "6E": 0.0086,   "6I": 0.0},
    "6E":   {"23E": 0.0156, "23I": 0.0066,  "4E": 0.0211,   "4I": 0.0166,   "5E": 0.0572,   "5I": 0.0197,   "6E": 0.0396,   "6I": 0.2252},
    "6I":   {"23E": 0.0364, "23I": 0.001,   "4E": 0.0034,   "4I": 0.0005,   "5E": 0.0277,   "5I": 0.008,    "6E": 0.0658,   "6I": 0.1443}}
    

# In-degrees for external inputs
NUM_EXTERNAL_INPUTS = {
    "23":   {"E": 1600, "I": 1500},
    "4":    {"E": 2100, "I": 1900},
    "5":    {"E": 2000, "I": 1900},
    "6":    {"E": 2900, "I": 2100}}

# Mean rates in the full-scale model, necessary for scaling
# Precise values differ somewhat between network realizations
MEAN_FIRING_RATES = {
    "23":   {"E": 0.971,    "I": 2.868},
    "4":    {"E": 4.746,    "I": 5.396},
    "5":    {"E": 8.142,    "I": 9.078},
    "6":    {"E": 0.991,    "I": 7.523}}

# Means and standard deviations of delays from given source populations (ms)
MEAN_DELAY = {"E": 1.5, "I": 0.75}

DELAY_SD = {"E": 0.75, "I": 0.375}

# seed for RNGs
RNGSEED = args.seed

# Print time progress
PRINT_TIME = False

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def get_scaled_num_neurons(layer, pop):
    return int(round(NEURON_SCALING_FACTOR * NUM_NEURONS[layer][pop]))

def get_full_num_inputs(src_layer, src_pop, trg_layer, trg_pop):
    num_src = NUM_NEURONS[src_layer][src_pop]
    num_trg = NUM_NEURONS[trg_layer][trg_pop]
    connection_prob = CONNECTION_PROBABILTIES[trg_layer + trg_pop][src_layer + src_pop]

    return int(round(np.log(1.0 - connection_prob) / np.log(float(num_trg * num_src - 1) / float(num_trg * num_src))) / num_trg)

def get_mean_weight(src_layer, src_pop, trg_layer, trg_pop):
    # Determine mean weight
    if src_pop == "E":
        if src_layer == "4" and trg_layer == "23" and trg_pop == "E":
            return LAYER_23_4_W
        else:
            return MEAN_W
    else:
        return G * MEAN_W

def get_scaled_num_connections(src_layer, src_pop, trg_layer, trg_pop):
    # Scale full number of inputs by scaling factor
    num_inputs = get_full_num_inputs(src_layer, src_pop, trg_layer, trg_pop) * CONNECTIVITY_SCALING_FACTOR
    assert num_inputs >= 0.0

    # Multiply this by number of postsynaptic neurons
    return int(round(num_inputs * float(get_scaled_num_neurons(trg_layer, trg_pop))))

def get_full_mean_input_current(layer, pop):
    # Loop through source populations
    mean_input_current = 0.0
    for src_layer in LAYER_NAMES:
        for src_pop in POPULATION_NAMES:
            mean_input_current += (get_mean_weight(src_layer, src_pop, layer, pop) *
                                   get_full_num_inputs(src_layer, src_pop, layer, pop) *
                                   MEAN_FIRING_RATES[src_layer][src_pop])

    # Add mean external input current
    mean_input_current += EXTERNAL_W * NUM_EXTERNAL_INPUTS[layer][pop] * BACKGROUND_RATE
    assert mean_input_current >= 0.0
    return mean_input_current


# Start timing
time_start = perf_counter_ns()


# ----------------------------------------------------------------------------
# Network creation
# ----------------------------------------------------------------------------
model = genn_model.GeNNModel("float", "potjans_microcircuit")
model.dT = DT_MS
model._model.set_merge_postsynaptic_models(True)
model._model.set_default_narrow_sparse_ind_enabled(True)
model.timing_enabled = MEASURE_TIMING
model.default_var_location = genn_wrapper.VarLocation_DEVICE
model.default_sparse_connectivity_location = genn_wrapper.VarLocation_DEVICE
# set seed for RNGs
model._model.set_seed(RNGSEED)
print("Seed: ", model._model.get_seed())


# added optimized option for membrane potential initialization
optimized_mean = [-68.28, -63.16, -63.33, -63.45, -63.11, -61.66, -66.72, -61.43]
optimized_std = [5.36, 4.57, 4.74, 4.94, 4.94, 4.55, 5.46, 4.48]
poisson_init = {"current": 0.0}

exp_curr_params = {"tau": 0.5}

quantile = 0.9999
normal_quantile_cdf = norm.ppf(quantile)
max_delay = {pop: MEAN_DELAY[pop] + (DELAY_SD[pop] * normal_quantile_cdf)
             for pop in POPULATION_NAMES}
print("Max excitatory delay:%fms , max inhibitory delay:%fms" % (max_delay["E"], max_delay["I"]))

# Calculate maximum dendritic delay slots
# **NOTE** it seems inefficient using maximum for all but this allows more aggressive merging of postsynaptic models
max_dendritic_delay_slots = int(round(max(itervalues(max_delay)) / DT_MS))
print("Max dendritic delay slots:%d" % max_dendritic_delay_slots)

print("Creating neuron populations:")
total_neurons = 0
neuron_populations = {}
counter = 0
for layer in LAYER_NAMES:
    for pop in POPULATION_NAMES:
        pop_name = layer + pop
        lif_init = {"V": genn_model.init_var("Normal", {"mean": optimized_mean[counter], "sd": optimized_std[counter]}), "RefracTime": 0.0}
        counter += 1

        # Calculate external input rate, weight and current
        ext_input_rate = NUM_EXTERNAL_INPUTS[layer][pop] * CONNECTIVITY_SCALING_FACTOR * BACKGROUND_RATE
        ext_weight = EXTERNAL_W / np.sqrt(CONNECTIVITY_SCALING_FACTOR)
        ext_input_current = 0.001 * 0.5 * (1.0 - np.sqrt(CONNECTIVITY_SCALING_FACTOR)) * get_full_mean_input_current(layer, pop)
        assert ext_input_current >= 0.0

        lif_params = {"C": 0.25, "TauM": 10.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh" : -50.0,
                      "Ioffset": ext_input_current, "TauRefrac": 2.0}
        poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": ext_input_rate}

        pop_size = get_scaled_num_neurons(layer, pop)
        neuron_pop = model.add_neuron_population(pop_name, pop_size, "LIF", lif_params, lif_init)
        model.add_current_source(pop_name + "_poisson", "PoissonExp", pop_name, poisson_params, poisson_init)

        # Enable spike recording
        neuron_pop.spike_recording_enabled = USE_GENN_RECORDING

        print("\tPopulation %s: num neurons:%u, external input rate:%f, external weight:%f, external DC offset:%f" % (pop_name, pop_size, ext_input_rate, ext_weight, ext_input_current))

        # Add number of neurons to total
        total_neurons += pop_size

        # Add neuron population to dictionary
        neuron_populations[pop_name] = neuron_pop

# Loop through target populations and layers
print("Creating synapse populations:")
total_synapses = 0
num_sub_rows = NUM_THREADS_PER_SPIKE if PROCEDURAL_CONNECTIVITY else 1
for trg_layer in LAYER_NAMES:
    for trg_pop in POPULATION_NAMES:
        trg_name = trg_layer + trg_pop

        # Loop through source populations and layers
        for src_layer in LAYER_NAMES:
            for src_pop in POPULATION_NAMES:
                src_name = src_layer + src_pop

                # Determine mean weight
                mean_weight = get_mean_weight(src_layer, src_pop, trg_layer, trg_pop) / np.sqrt(CONNECTIVITY_SCALING_FACTOR)

                # Determine weight standard deviation
                if src_pop == "E" and src_layer == "4" and trg_layer == "23" and trg_pop == "E":
                    weight_sd = mean_weight * LAYER_23_4_RELW
                else:
                    weight_sd = abs(mean_weight * REL_W)

                # Calculate number of connections
                num_connections = get_scaled_num_connections(src_layer, src_pop, trg_layer, trg_pop)

                if num_connections > 0:
                    num_src_neurons = get_scaled_num_neurons(src_layer, src_pop)
                    num_trg_neurons = get_scaled_num_neurons(trg_layer, trg_pop)

                    print("\tConnection between '%s' and '%s': numConnections=%u, meanWeight=%f, weightSD=%f, meanDelay=%f, delaySD=%f" 
                          % (src_name, trg_name, num_connections, mean_weight, weight_sd, MEAN_DELAY[src_pop], DELAY_SD[src_pop]))

                    # Build parameters for fixed number total connector
                    connect_params = {"total": num_connections}

                    # Build distribution for delay parameters
                    d_dist = {"mean": MEAN_DELAY[src_pop], "sd": DELAY_SD[src_pop], "min": 0.0, "max": max_delay[src_pop]}

                    total_synapses += num_connections

                    # Build unique synapse name
                    synapse_name = src_name + "_" + trg_name

                    matrix_type = "PROCEDURAL_PROCEDURALG" if PROCEDURAL_CONNECTIVITY else "SPARSE_INDIVIDUALG"

                    # Excitatory
                    if src_pop == "E":
                        # Build distribution for weight parameters
                        # **HACK** np.float32 doesn't seem to automatically cast 
                        w_dist = {"mean": mean_weight, "sd": weight_sd, "min": 0.0, "max": float(np.finfo(np.float32).max)}

                        # Create weight parameters
                        static_synapse_init = {"g": genn_model.init_var("NormalClipped", w_dist),
                                               "d": genn_model.init_var("NormalClippedDelay", d_dist)}

                        # Add synapse population
                        syn_pop = model.add_synapse_population(synapse_name, matrix_type, genn_wrapper.NO_DELAY,
                            neuron_populations[src_name], neuron_populations[trg_name],
                            "StaticPulseDendriticDelay", {}, static_synapse_init, {}, {},
                            "ExpCurr", exp_curr_params, {},
                            genn_model.init_connectivity("FixedNumberTotalWithReplacement", connect_params))

                        # Set max dendritic delay and span type
                        syn_pop.pop.set_max_dendritic_delay_timesteps(max_dendritic_delay_slots)

                        if PROCEDURAL_CONNECTIVITY:
                            syn_pop.pop.set_span_type(genn_wrapper.SynapseGroup.SpanType_PRESYNAPTIC)
                            syn_pop.pop.set_num_threads_per_spike(NUM_THREADS_PER_SPIKE)
                    # Inhibitory
                    else:
                        # Build distribution for weight parameters
                        # **HACK** np.float32 doesn't seem to automatically cast 
                        w_dist = {"mean": mean_weight, "sd": weight_sd, "min": float(-np.finfo(np.float32).max), "max": 0.0}

                        # Create weight parameters
                        static_synapse_init = {"g": genn_model.init_var("NormalClipped", w_dist),
                                               "d": genn_model.init_var("NormalClippedDelay", d_dist)}

                        # Add synapse population
                        syn_pop = model.add_synapse_population(synapse_name, matrix_type, genn_wrapper.NO_DELAY,
                            neuron_populations[src_name], neuron_populations[trg_name],
                            "StaticPulseDendriticDelay", {}, static_synapse_init, {}, {},
                            "ExpCurr", exp_curr_params, {},
                            genn_model.init_connectivity("FixedNumberTotalWithReplacement", connect_params))

                        # Set max dendritic delay and span type
                        syn_pop.pop.set_max_dendritic_delay_timesteps(max_dendritic_delay_slots)

                        if PROCEDURAL_CONNECTIVITY:
                            syn_pop.pop.set_span_type(genn_wrapper.SynapseGroup.SpanType_PRESYNAPTIC)
                            syn_pop.pop.set_num_threads_per_spike(NUM_THREADS_PER_SPIKE)
print("Total neurons=%u, total synapses=%u" % (total_neurons, total_synapses))


# Time to model definition
time_model_def = perf_counter_ns()


if BUILD_MODEL:
    print("Building Model")
    model.build()


# Time to build model
time_build = perf_counter_ns()


print("Loading Model")
duration_timesteps = int(round(DURATION_MS / DT_MS))
model.load(num_recording_timesteps=duration_timesteps)


# Time to load model
time_load = perf_counter_ns()


ten_percent_timestep = duration_timesteps // 10
print("Simulating")

# Simulation
# Loop through timesteps
while model.t < DURATION_MS:
    # Advance simulation
    model.step_time()

    # Indicate every 10%
    if PRINT_TIME and (model.timestep % ten_percent_timestep) == 0:
        print("%u%%" % (model.timestep / 100))


# Time to simulate
time_simulate =  perf_counter_ns()


time_dict = {
        "time_model_def": time_model_def - time_start,
        "time_build": time_build - time_model_def,
        "time_construct_no_load": time_build - time_start,
        "time_load": time_load - time_build,
        "time_construct": time_load - time_start,
        "time_simulate": time_simulate - time_load,
        "time_total": time_simulate - time_start,
        }

info_dict = {
        "conf": {
            "seed": args.seed,
            "num_neurons": total_neurons
        },
        "timers": time_dict
    }

with file_path.open("w") as f:
    dump(info_dict, f, indent=4)

print(dumps(info_dict, indent=4))

if MEASURE_TIMING:
    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tSynapse simulation:%f" % (1000.0 * model.presynaptic_update_time))
    print("\tOverall time:%f" % (1000.0 * (model.presynaptic_update_time + model.neuron_update_time + 
                                          model.init_sparse_time + model.init_time)))

