ActivitySim
===========

[![Build Status](https://travis-ci.com/ActivitySim/activitysim.svg?branch=main)](https://travis-ci.org/github/ActivitySim/activitysim)[![Coverage Status](https://coveralls.io/repos/github/ActivitySim/activitysim/badge.svg?branch=main)](https://coveralls.io/github/ActivitySim/activitysim?branch=main)

The mission of the ActivitySim project is to create and maintain advanced, open-source, 
activity-based travel behavior modeling software based on best software development 
practices for distribution at no charge to the public.

The ActivitySim project is led by a consortium of Metropolitan Planning Organizations 
(MPOs) and other transportation planning agencies, which provides technical direction 
and resources to support project development. New member agencies are welcome to join 
the consortium. All member agencies help make decisions about development priorities 
and benefit from contributions of other agency partners. 

## Documentation

https://activitysim.github.io/activitysim  

# UAL FORK

UAL uses AcitivitySim as one component of an integrated transportation and land use simulation platform ([PILATES](https://github.com/ual/PILATES)). As such, most of our changes to the ActivitySim code are designed to facilitate dynamic reading and writing of data.

## UAL Settings
We've augmented the canonical ActivitySim **settings.yaml** file with default parameters that are specific to the new submodel steps we've created. These parameters are found at the top of all of the settings.yaml files, and should look something like this:
```
# output
s3_output: False

# geographic settings
state_fips: 06
local_crs: EPSG:7131 

# skims
create_skims_from_beam: True
beam_skims_url: https://<url of skims>

# urbansim data
create_inputs_from_usim_data: True
year: 2010
scenario: base
bucket_name: bayarea-activitysim
usim_data_store: model_data.h5
```

## UAL runtime args
Most of the time our ActivitySim code will be executed as a Docker image, so any run-specific configs that might need to change from one run to the next had to be able to be specified at runtime. For the most part these arguments give the user the ability to override the UAL-specific settings mentioned above, which serve as defaults.
```
usage: simulation.py [-b BUCKET_NAME] [-y YEAR] [-s SCENARIO] [-u SKIMS_URL] [-x PATH_TO_REMOTE_DATA] [-w] [-h HOUSEHOLD_SAMPLE_SIZE]

optional arguments:
  -b BUCKET_NAME, --bucket_name BUCKET_NAME
                        s3 bucket name
  -y YEAR, --year YEAR  data year
  -s SCENARIO, --scenario SCENARIO
                        scenario
  -u SKIMS_URL, --skims_url SKIMS_URL
                        url of skims .csv
  -x PATH_TO_REMOTE_DATA, --path_to_remote_data PATH_TO_REMOTE_DATA
                        url of urbansim .h5 model data
  -w, --write_to_s3     write output to s3?
  -h HOUSEHOLD_SAMPLE_SIZE, --household_sample_size HOUSEHOLD_SAMPLE_SIZE
                        household sample size
```
If none of these are specified then ActivitySim will attempt read these settings from **settings.yaml**. If `-x` is not specified then ActivitySim will attempt to read/write data from s3 using a dynamically generate filepath like:
`s3://<bucket_name>/<input/output>/<scenario>/<year>/model_data.h5`

## UAL submodels
Activitysim expects all of the input data to be there before it starts up. Since we're creating the data dynamically, we have to run a few preprocessing steps before kicking off ActivitySim in earnest. We currently do this with a couple custom ActivitySim submodels that we call manually from the main simulation.py script. Eventually these should get pulled out of ActivitySim and moved into PILATES.
- **initialize_skims_from_beam.py**: this module contains the `create_skims_from_beam` step which downloads the skims from a URL endpoint and transforms them into the format ActivitySim expects (**skims.omx**).
- **initalize_from_usim.py**: this module contains the `create_inputs_from_usim` step which reads in UrbanSim-formatted land use and population data (usually stored as .h5) and transforms them to create the **land_use.csv**, **households.csv**, and **persons.csv** input files that ActivitySim expects.

There is also two submodels that get run in-sequence with the main ActivitySim submodels as specified in **settings.yaml**:
- **generate_beam_plans.py**: this module contains the `generate_beam_plans` step which transforms the trips table into a person-plan based table of daily activities that can be read by BEAM.
- **write_outputs_to_s3.py**: this module contains the `write_outputs_to_s3` step which writes the activitysim outputs to AWS s3 so that the downstream simulation models (UrbanSim, BEAM, etc.) can use them as inputs for the next simulation iteration.


## Docker
- `docker build -t <dockeruser>/activitysim .`
- `docker run -w <working dir, e.g. "/activitysim/bay_area"> <dockeruser>/activitysim -b <s3 bucket> -y <input data year> -s <scenario> -u <skims url> -w`


## [TO DO](https://github.com/ual/activitysim/wiki/TO-DO)