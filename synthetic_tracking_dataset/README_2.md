# Synthetic Tracking Dataset

This folder contains the code and dataset used to generate synthetic radar measurements, together with the corresponding ground-truth target states for tracking experiments.

An example target trajectory is shown below.

![Target trajectory](./figs/target_traj.png)

## Metadata and folder structure

We provide each dataset sample in both `.json` and `.mat` format.  
Each folder corresponds to one tracking event.

```bash
{Name_of_event_folder}/
├── log.json
└── log.mat
```



## Synthetic data generation

We currently provide three types of target motion scenarios:
* linear
* random_walk
* sinusoid

If needed, additional data can be generated using:

```bash
data_generator.m
```
This script can be used to generate more tracking events with different motion patterns and parameter settings.

## What is stored in `log.json`

The `log.json` file contains all information required to reproduce, inspect, or analyze one synthetic tracking event. It includes experiment-level metadata, radar network configuration, simulation constants, ground-truth target states, and simulated radar measurements.

1. Run-level metadata

The top-level fields describe the overall setup of the generated sample, including:

`RUN_NAME`: name of the sample

`NUM_TAR`: number of targets

`NUM_CPI_PER_MEA`: number of CPIs per measurement

`track_time`: total tracking duration

`mc`: Monte Carlo run index

`TYPE`: target motion type, e.g. linear, random_walk, or sinusoid

`PRE_WHITEN`: whether pre-whitening is enabled

`seed`: random seed used for data generation 


2. `network_topo`

This section stores the radar network topology and graph structure, including:

`numNodes`: number of radar nodes

`theta`: angular placement of nodes

`com_rad_CR`: communication radius

`radius`: deployment radius

`radar_pos`: 2D position of each radar node

`labels`: node labels

`adj_matrix`: adjacency matrix

`degree_matrix`: degree matrix

`laplacian_matrix`: graph Laplacian

`inc_matrix`: incidence matrix

`weights_matrix`: graph weights used in the network formulation

3. `constant`

This section contains physical constants and simulation parameters, such as:

`c`: speed of light

`lambda`: radar wavelength

`time_step`: simulation time step

`Bandwidth`: radar bandwidth of each node

`fs`: sampling frequency of each node

`SNR_idx`, SNR_lin: signal-to-noise-ratio settings

`range_var`, `doppler_var`: measurement variances

`range_sd`, `doppler_sd`: measurement standard deviations

`measurements_noise`: measurement noise covariance

`process_noise`: process noise covariance

`pre_whit_L`: pre-whitening matrix

`PRE_WHITEN`: whether pre-whitening is enabled

4. `target`

This section describes the target initialization and ground-truth states, including:

`angle_degrees`: target moving direction in degrees

`initial_position`: initial target position

`speed`: target speed

`direction`: target moving direction vector

`true_params`: initial target state in the form [x, y, vx, vy]

`target_position`: target ground-truth position sequence

`target_state`: target ground-truth full state sequence in the form [x, y, vx, vy]

5. `Trajectory`

This section stores the trajectory and measurement data used in the tracking task:

`true_params`: ground-truth target state sequence over time

`measurements`: simulated radar measurements collected by the sensor network

For the example file, the measurement tensor is organized by node, target, measurement type, and time. The two measurement channels correspond to range and Doppler.