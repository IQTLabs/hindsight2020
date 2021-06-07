"""
This driver will process the CORD19 dataset of academic articles and the DHS sentences-of-evidence
to perform claim matching between the latter and the former using SBERT.

Uses config.yml for setup.

To run:
activate the environment specified in the README, 
source hindsight/bin/activate
then:
time python claimMatching/main.py -s 'manual' -c 'cord19_json' 

Takes about 24 hours to run on the CORD19 dataset from Dec2020 (default settings in config/config.yaml)

@author: kdobolyi (code is an extension of ClaimMatching/Infodemic repo written by @brocklin)
"""
import func.match_claims as ClaimMatcher

import argparse, os, sys, yaml

if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', dest='config', type=str,
                        default='hindsight/config/config.yml',
                        help='configuration file\'s location')
    parser.add_argument('-s', '--search_set', dest='search_set', type=str,
                        default='manual', choices=['misc_json', 'manual'],
                        help='which items to find matching claims for')
    parser.add_argument('-c', '--candidate_set', dest='candidate_set', type=str,
                        default='cord19_json', choices=['misc_json', 'cord19_json'],
                        help='which items to use as potential matches for the search set items')
    arguments = parser.parse_args()

    # configuration setup
    CWD = os.getcwd()
    config_loc = os.path.join(CWD, arguments.config)
    with open(config_loc, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if not cfg:
        print('Config file not found, please supply a valid, non-empty config or use the default.')
        sys.exit(1)
    cfg['CWD'] = CWD
    ClaimMatcher.find_nearest_claims(arguments.search_set, arguments.candidate_set, cfg)
