
# Benchmarking Feature Selection Methods in Radiomics

This repository contains the source code for the paper 'Benchmarking Feature Selection Methods in Radiomics', published in Investigative Radiology, July 2022.



## Setting up

We need some packages, a pip freeze can be found there with the exact versions used (Personally, I do not really believe in this, replicating runs 100% is sometimes hard,
and if a second run with slightly different version results in a huge difference, this should be investigated instead of swept under the rug).

I tried to fix all random numbers etc so that runs should be deterministic, but did not explicitly test for this.

To install all packages:

`$ pip install -r requirements.txt`

It *might* be possible that pymrmre needs to  be installed last. I installed it afterwards and needed to install other packages before pymrmre.

If you want to use a virtual environment for this, you can execute something like this, in case of python 3.6

`$ virtualenv -p /usr/bin/python3.6 /data/radFS/venv`
`$ source /data/radFS/venv/bin/activate`

We also need the scikit-feature package from github. Unfotunately, I had to modify a small thing to get rid of warnings, so the version to use is below ./skfeature/ . Installation is not needed, the code will be used directly.



## Experiment

The experiment is then started with ./startExperiment.py It will write all the artifacts into /data/radFS/mlruns
**One can change this path by changing the TrackingPath variable at the beginning of the file.**
Also, it uses 6 cores for running, this can be changed at the very bottom of the file.

Experiments already executed will not execute a second time. One my machine I had during development several strange crashed, I believe these stem from race conditions, so to avoid to restart everything, I implemented a simple check.

The mlflow ui can be started by

`$ mlflow ui --backend-store-uri file:///data/radFS/mlruns`

It can be used to either track the experiments or to just look at the metrics.



## Evaluation

Evaluation code is unfortunately completely messy, also revision, introducing 3 feature types, added to this. Some extra packages needs to be installed, e.g. cm-super, dvipng packages are needed for plotting. (Unfortunately no requirements.txt available, but this is not critical).

Evaluation needs access to the whole mlruns folders, because it needs to recompute some of the results, which were not computed during the experiment. The path can be found in at the beginning in `TrackingPath = "/data/radFS/mlruns".`

If the experiment is not re-executed, the mlruns needs to be _exactly_ at this place, else artifacts will not be found, as these seem to be hardlinked rin the meta.yml files in the mlrun folders.

**Note**: For convenience, the combined results are stored in the results folder of this repo, so the evaluation can take place without recomputing the whole experiment.

**Note**: TWO FILTERS, DCSF AND FCBF were removed AFTER experiments were finished. This stems from the fact that both seem to have always returned constant
feature scores, so that selectKBest did select the last features. Since this bug skewed the results, those two were removed.



## License

**Note: Data and scikit.features package have their own license. Please refer to the respective publications and to the scikit.features package**

Other code is licensed with the MIT license:

Copyright (c) 2022 Aydin Demircioglu
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
