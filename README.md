# BSNN

I've cleaned up the code a bit. Data loaders are under `dataloaders`, layers
should go under `layers`, and testing models should go under `models`. Models
should have a `run_model` function. 

I renamed `StochasticBinaryLayer.py` to `layers/bernoulli.py` and
`StochasticBinaryModel.py` to `models/linear_bernoulli.py`. In general, since
models will be dataset specific and (at least for the near future) exist to
test specific types of stochastic neurons, you should stick to the
`<dataset>_<layer type>.py` naming scheme for models (where `dataset` is the
name of a loader in `dataloaders`)

`main.py` runs the `linear_bernoulli` model. I will extend it for other models.
In general, write similar code in files at the repo root. 
