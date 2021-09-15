# Detecting sarcasm with RoBERTa

## Installation

* Create your own environment with conda and activate it.
* Use: pip install -r requirements.txt
* Install `torch` on your computer yourself, as this depends on your platform 
  and on the availability of CUDA.

## Data

The used data files are based on the Khodak et al. at [SARC set](https://nlp.cs.princeton.edu/SARC/2.0/),
but slightly changed to have no full ancestry, but rather a single ancestor in the `parent_comment` field.

The actual data files are not publicly 


## Exploring data and outcome

See notebooks:

* Exploration of data: data_exploration.ipynb
* Exploration of outcome: outcome_exploration.ipynb


## Configuration and running

As training the model might take hours, it is started from the command line so it can run as
a process in the background.

To start it:
> `$ python run.py`

To configure it - primitively - check out a number of variables at the top of `run.py`
and check what is actually being called at the bottom in the main routine.

 