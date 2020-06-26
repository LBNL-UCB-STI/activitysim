ActivitySim
===========

[![Build Status](https://travis-ci.org/ActivitySim/activitysim.svg?branch=master)](https://travis-ci.org/ActivitySim/activitysim)[![Coverage Status](https://coveralls.io/repos/github/ActivitySim/activitysim/badge.svg?branch=master)](https://coveralls.io/github/ActivitySim/activitysim?branch=master)

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


## TO DO:
- [ ] fix tour sequencing errors in plan generation due to trips with shared departure hours
- [ ] clean up auto_ownership and tour_mode_choice calibration notebooks
- [ ] merge changes from RSG activitysim fork into ours
   - [ ] ride hail/TNC
   - [ ] estimation notebooks
- [ ] create inputs from urbansim data
   - [x] ~~move to `abm/models/` (now **initialize_from_usim.py**)~~
   - [x] ~~split out skim conversion to its own .py (now **initialize_skims_from_beam.py**) ~~
   - [ ] replace orca calls with activitysim.core.inject methods
   - [x] ~~recycle zone assignment code for blocks, schools, colleges, etc.~~
   - [x] ~~improve block to hex mapping~~
   - [ ] improve land use data creation
      - [x] ~~improve `area_type` imputation (cbd vs. urban core vs. rural, etc.)~~
      - [ ] improve `terminal` time (i.e. walk time to vehicle) imputation
      - [ ] replace county ID dummies
         - should only effect free_parking and auto_ownership models
- [x] ~~Improve activity_plans.py ~~
   - [x] ~~move to `models/` dir (now **generate_beam_plans.py**)~~
   - [x] ~~preserve home coords from higher res urbansim data~~

