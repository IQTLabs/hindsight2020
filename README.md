# HindSight2020

## Intro

This codebase matches sentences between two corpora of COVID19 academic articles (or their summaries).

We use this approach to claim-match a human-curated set of COVID19 research (corpora1, DHS) against over a hundred thousand academic articles on COVID19 (corpora2, CORD19-AllenAI) to trace the evolution of research for emerging infectious disease outbreaks. Specifically, we examine the stability around early evidence in the course of pandemic, and therfore, the feasibility and timelines of using early research conclusions for policy-making during novel pandemics.

For more details, see our [Hindsight2020 paper](http://FIXME).

## Abstract

We apply claim-matching to various subsets of the COVID-19 scientific literature. Our goal is to build a framework for characterizing uncertainty in emerging infectious disease (EID) outbreaks as a function of time, peer review, hypothesis-sharing, evidence collection practice, and interdisciplinary citation networks. In the healthiest of times, scientists face validity, methodology, and reproducibility challenges; following EIDs, the rush to publish and the risk of data misrepresentation, misinterpretation or worse misinformation puts an even greater onus on methodological rigor and proper understanding of the scientific method, which includes revisiting initial assumptions. This project seeks to understand how and when early evidence emerges for different types of recurring EID questions, outlined below, via a deep learning approach using SBERT for claim matching. In addition, it makes publicly available an expert-annotated dataset of 5,815 matched sentence pairs that can be used to fine-tune future COVID-19 natural language programming models.

## Claim Matching using SBERT 

We use a pre-trained SBERT model to efficiently compare pairs of sentences; our DHS search dataset contains about 600 ground-truth claims, which we are able to match against 7 million sentences from the CORD19 candidate dataset within less than 24 hours using this approach.

## HindSight2020 Setup
1. Navigate to the repository root directory in your terminal.
2. Create a new virtual environment.
   2. On MacOS/Linux, run `python3 -m venv hindsightEnv` in your terminal.
   2. On Windows, run `py -m venv hindsightEnv` in CMD or Powershell.
3. Activate the virtual environment. This should be done every time you run code.
   3. On MacOS/Linux, run `source hindsightEnv/bin/activate`.
   3. On Windows, run `.\hindsightEnv\Scripts\activate`.
4. From the repository root, install every package needed by running `pip install -r requirements.txt`.
5. In the repository root, run the setup script.
   5. On MacOS/Linux, run `python hindsight/setup.py`.
   5. On Windows, run `python .\hindsight\setup.py`.
6. When finished running commands, run `deactivate` to stop using the virtual environment. Be sure to reactivate the
virtual environment per the instructions given above when you choose to run the jupyter notebooks or claim matcher.

### Running HindSight2020 Claim Matcher
1. Navigate to your repository root.
2. Modify the `./config/config.yaml` file for the setup you would like to run (repo works with defaults there).
3. Activate the virtual environment from the setup above.
4. Run `time python hindsight/main.py -s 'manual' -c 'cord19_json' -d` from your repository root. This will extract the sentences from the
CORD19 dataset of json files, and claim-match them against the DHS ground truth evidence contained in config.yml. 

### HindSight2020 Pre-processing steps
If you want to do a different analysis than what is in `config/config.yaml`, you'll need to prepare all of the files that configuration file
refers to. Below is a reference for those pieces, along with the scripts in this repo that can be used to generate those files.

#### Setting up the CORD19 candidate document set of academic articles
The [CORD19 dataset](https://allenai.org/data/cord-19) is a publicly-available dataset of COVID19 academic articles and related research
that is updated daily/weekly, and made available in json format. We rely on some of the fields to be present in these json files
(such as abstract, date, paperID, etc) to perform our processing in `cord19_json_loader.py`; you can modify this file to conform to 
whatever json fields you'd like to read in. To use this dataset or one like it, specify the following fields in `config/config.yaml`:
1. `cord19_dir`: the path to json files, one for each CORD19 paper
2. `metadata_path`: the path to the metadata csv that comes with a CORD19 download


1. Navigate to your repository root.
2. Modify the `./config/config.yaml` file for the setup you would like to run (repo works with defaults there).
3. Activate the virtual environment from the setup above.
4. Run `time python hindsight/main.py -s 'manual' -c 'cord19_json' -d` from your repository root. This will extract the sentences from the
CORD19 dataset of json files, and claim-match them against the DHS ground truth evidence contained in config.yml. 

## Acknowledgements

Thanks to Ben RockLin et al. for IQTLab's [Infodemic Claim-Matching](https://github.com/IQTLabs/ClaimMatching) codebase, where were borrowed claim matching code using SBERT.

