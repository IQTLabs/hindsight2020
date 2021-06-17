"""
Performs claim matching via SBERT

@author: kdobolyi (this is modification of ClaimMatching/Infodemic repo written by @brocklin)
"""
from datetime import datetime
import os, sys, csv, re
import traceback

import numpy as np
import scipy
from sentence_transformers import SentenceTransformer

import func.cord19_json_loader as Cord19Loader


def encode_sets(search_docs, candidate_docs, model_weights, search_set, candidate_set):
    """
    Encodes all the items in the search and candidate sets using an
    SBERT sentence encoder with the given weights.

    :param search_docs: list of all documents from the search set
    :param candidate_docs: list of all documents from the candidate set
    :param model_weights: name of the pre-trained weights to use to encode the documents
    :param search_set: name of the search set
    :param candidate_set: name of the candidate set
    :return: a list of embeddings for the search set and a list of embeddings for the candidate set
    """
    embedder = SentenceTransformer(model_weights)

    if search_set == candidate_set:
        print("Encoding search and candidate set data...")
        search_embeddings = embedder.encode(search_docs)
        return search_embeddings, search_embeddings
    else:
        print("Encoding search set...")
        search_embeddings = embedder.encode(search_docs)

        print("Encoding candidate set...")
        if len(candidate_docs) == 2:  # if we are using CORD19 where we have to keep track of paperIDs, use just sentences
            candidate_docs = candidate_docs[0]
        candidate_embeddings = embedder.encode(candidate_docs)
        return search_embeddings, candidate_embeddings


def get_top_matches(search_embedding, candidate_embeddings, search_set, candidate_set, cfg):
    """
    Helper function to find the top matching items in the candidate set for a search set item.

    :param search_embedding: SBERT-generated embedding for search set document
    :param candidate_embeddings: SBERT-generated embeddings for all documents in the target set
    :param search_set: name of the search set
    :param candidate_set: name of the target set
    :param cfg: configuration dictionary
    :return: distances and indices of the closest candidate documents to the search document
    """
    distances = scipy.spatial.distance.cdist([search_embedding], candidate_embeddings, "cosine")[0]
    sorted_dists = np.argsort(distances)

    # if the search set is the same as the candidate set, keep the same element from showing up in the matches
    low_indices = None
    if search_set == candidate_set:
        low_indices = sorted_dists[1:cfg['num_matches'] + 1]
    else:
        low_indices = sorted_dists[:cfg['num_matches']]
    return distances, low_indices

def create_paperID_to_citation_mapping(title_to_id, cfg):
    """
    Tries to map the CORD19 paper_ID to the citation numbers provided via the DHS evidence.
    Allows us to connect the DHS paper mentions/citations to the CORD19 papers, as these do not share paper_IDs;
    we rely on matching the title of the paper across these two datasets for a match (TODO: this could be improved).

    Arguments:
        [dict] title_to_id : a dict that assosicates a list of CORD19 sha-s with each unique article title
        [dict] cfg :  the configuration file that points to the DHS_citations_path of the DHS evidence
                        specified in main.py via config.yml
    Returns:
        [dict] sentence_to_paperID_mapping: 
    """
    sentence_to_paperID_mapping = {}

    # open the file that contains all the citations as written by DHS in their evidence (bibliography/references of DHS)
    try:
        file = open(cfg['DHS_citations_path'])
        citations = file.readlines()
        file.close()
    except:
        print("citations file not found, leaving without it...")
        return sentence_to_paperID_mapping

    # mine the title of the paper from the DHS bibliography/references, and map it back to the DHS citation number
    citation_to_title = {}
    for c in citations:
        matchObj = re.match( r"([\d]+)\. (.+)(\.,)(.+)(\. )(.+\.)(.*)", c)
        if matchObj:
            title = matchObj.group(4).strip()
            citation_to_title[matchObj.group(1)] = title[:].split('.')[0].replace("<em>",'').replace("</em>",'')
        else:
            c = c.split(". ")
            citation_to_title[c[0]] = "None"

    # associate each DHS sentence of evidence with the CORD19 paper_ID, via the title of the paper(s) that sentence cites
    file = open(cfg['DHS_raw_path'])
    sentence_with_citations = file.readlines()
    file.close()
    for line in sentence_with_citations:
        sentence = line.split("#")[0].strip()
        citations = line.split('#')[1]
        papers = ""
        for c in citations.split(','):
            c = c.strip()
            if len(c) > 0 and citation_to_title[c] != "None":
                if citation_to_title[c] in title_to_id.keys():
                    paperID = title_to_id[citation_to_title[c]]
            else:
                paperID = "None"
            papers += paperID + ","
        sentence_to_paperID_mapping[sentence] = papers

    return sentence_to_paperID_mapping

def citations_csv(cfg):
    '''
    Helper function to write citation metadata for each sentence (plus the sentence) to a file as a prep for a dataframe
    Arguments:
        [dict] cfg : configuration data from yaml file
    '''
    print("loading metadata...")
    article_metadata, title_to_id = load_CORD19_metadata(cfg)
    DHS_sent_to_paperID_mapping = create_paperID_to_citation_mapping(title_to_id, cfg)
    question_mapping = load_questions_mapping(cfg)

    # write metadata and sentence to the specified csv
    csv = open(cfg['DHS_citations_dataframe_path'], "w")
    csv.write('sentence,paperID,date,journal,affiliation,question\n')
    for sentence in list(DHS_sent_to_paperID_mapping.keys()):
        print(sentence)
        paperIDs = DHS_sent_to_paperID_mapping[sentence]
        for paperID in paperIDs.replace(';',',').replace(' ','').split(','):
            if paperID != 'None' and len(paperID) > 5:
                date = article_metadata[paperID]['date']
                journal = article_metadata[paperID]['journal']
                affiliation = ''
                if 'affiliation' in article_metadata[paperID].keys():
                    affiliation = article_metadata[paperID]['affiliation']
                question = question_mapping[sentence.strip()]
                csv.write('"'+sentence+'",'+paperID+','+date+','+journal+','+affiliation+','+question+"\n")
    csv.close()
    print('wrote df to ' + cfg['DHS_citations_dataframe_path'])

def make_known_citation_folder(DHS_sent_to_paperID_mapping):
    """
    Helper function to create a directory of json files (of COVID articles) that are part of CORD19
    and also mentioned in the DHS citations

        Arguments:
            [dict] DHS_sent_to_paperID_mapping : maps DHS sentences to their sha-s (ids)
    """
    try:
        os.system("rm -r ./data/cord19_json/known_citations_json")
    except:
        pass

    os.system("mkdir ./data/cord19_json/known_citations_json/")
    for v in DHS_sent_to_paperID_mapping.values():
        shas = v.split(";")
        for sha in shas:
            sha = sha.replace(",","").strip()
            if sha != "None":
                os.system("cp ./data/cord19_json/pdf_json_jan4/"+sha+".json ./data/cord19_json/known_citations_json/"+sha+".json")

def write_search_doc_output(file, search_doc, candidate_docs, match, distances, low_indices, article_metadata, DHS_sent_to_paperID_mapping, question_mapping, cfg):
    """
    Helper function to write output for matches to each document in the search set.

    :param file: file to output to
    :param search_doc: document from the search set
    :param candidate_docs: all candidate documents
    :param match: matching keywords
    :param distances: distances to each nearby candidate
    :param low_indices: indices of each nearby candidate
    :param article_metadata: dict used to map article metadata to results file by paper_ID
    :param DHS_sent_to_paperID_mapping: 
    :param question_mapping: maps the DHS questions to their question number
    :param cfg: yaml file of configurations converted into a dict
    """

    delimiter = '\t'
    csv = open(cfg['results_csv_path'],"a+")

    if len(candidate_docs) == 2: # if we are dealing with the CORD19 dataset
        paperLookup = candidate_docs[1] # get the dict of paperIDs for the candidate docs
        candidate_docs = candidate_docs[0] # get the list of candidate docs

    # write information about the matched pair to a logging file
    file.write("Search set item:\n")
    file.write(search_doc.replace('\n', ' ') + '\n')
    file.write("----------------------------\n")
    if match:
        file.write("Matching keywords: " + str(match) + '\n')
    file.write("Distances to candidate: " + str(distances) + '\n')
    file.write("Top candidate set matches (high to low): \n")

    # for each matching sentence, print them out, along with metadata about the DHS sentence
    # and the matched CORD19 sentence, in a csv format result.cs
    ctr = 0
    for index in low_indices:

        # print out the DHS sentence and its metadata
        index = int(index)
        paperID = str(paperLookup[candidate_docs[index]])
        search_question = search_doc.replace('\n', ' ').strip()
        search_paperID = "None"
        if search_question in DHS_sent_to_paperID_mapping.keys():
            search_paperID = DHS_sent_to_paperID_mapping[search_question]
        file.write("- " + str(candidate_docs[index]).replace('\n', ' ') + "\t#paperID: " + paperID + '\n')

        # print out the matched CORD19 sentence and its metadata
        result = question_mapping[search_question] + delimiter
        result += search_question + delimiter
        result += str(candidate_docs[index]).replace('\n', ' ') + delimiter
        result += paperID + delimiter + search_paperID + delimiter
        result += article_metadata[paperID]['date'] + delimiter + article_metadata[paperID]['authors'] + delimiter 
        result += article_metadata[paperID]['journal'] + delimiter + article_metadata[paperID]['title'] + delimiter
        result += article_metadata[paperID]['affiliation'] + delimiter + str(distances[ctr])

        csv.write(result + "\n")
        ctr += 1

    file.write("\n")
    csv.close()

def retrieve_nearest(search_docs, candidate_docs, matches, search_set, candidate_set, cfg, article_metadata, DHS_sent_to_paperID_mapping, question_mapping):
    """
    Retrieves the nearest claims for each tweet passed in.

    :param search_docs: list of all documents in the search set
    :param candidate_docs: list of all documents in the candidate set
    :param matches: list of filtered tweets' matching keywords
    :param search_set: string denoting what data to use for the search set
    :param candidate_set: string denoting what data to use for the candidate set
    :param cfg: yaml file of configurations converted into a dict
    :param article_metadata: dict used to map article metadata to results file by paper_ID
    :param DHS_sent_to_paperID_mapping: 
    :param question_mapping: maps the DHS questions to their question number
    """

    print("encoding...")
    search_embeddings, candidate_embeddings = encode_sets(search_docs, candidate_docs, cfg['model'], search_set,
                                                          candidate_set)
    if len(matches) == 0:
        matches = [None] * len(search_embeddings)

    print("Writing output...")
    if not os.path.exists((cfg['output_dir'])):
        cwd = os.getcwd()    
        os.mkdir(os.path.join(cwd,cfg['output_dir']))

    with open(os.path.join(cfg['output_dir'], search_set + '-' + candidate_set + '-' +
                                              datetime.now().strftime("%m%d%y-%H%M%S") + '.txt'), "w") as f:
        f.write("Model: " + cfg['model'] + "\n")
        f.write("Search Set: " + search_set + "\n")
        f.write("Candidate Set: " + candidate_set + "\n")
        for search_doc, search_embedding, match in zip(search_docs, search_embeddings, matches):
            distances, low_indices = get_top_matches(search_embedding, candidate_embeddings, search_set,
                                                     candidate_set, cfg)
            write_search_doc_output(f, search_doc, candidate_docs, match, distances[low_indices], low_indices, article_metadata, DHS_sent_to_paperID_mapping, question_mapping, cfg)

def load_search_keywords(search_keywords, cfg):
    """
    Loads the NERs mined from each sentence with some cleanup into the incoming argument search_keywords
    using the NERs_path in the configuration dict cfg.

    Arguments:
        [dict] search_keywords : a dictionary that maps a sentence to its list of non-stopword keywords
        [dict] cfg : a dictionary of the configuration information in the yaml file

    """
    file = open(cfg['NERs_path'])
    data = file.readlines()
    file.close()

    stoplist = ['acute respiratory syndrome', 'coronavirus', 'SARS-CoV-2', 'Based', 'modeling',
                'estimates', 'small study', 'Interpretation', 'small scale', 'acute respiratory distress syndrome',
                'ARDS', 'Individual', 'statistics', 'Signs', 'Pandemic', 'Deaths', 'COVID-19', 'Situations',
                'Associated with', 'Illness', 'Controlling', 'Several studies', 'Research', 'Parameters', 'Row',
                'Evidence', 'Suggestions', 'Discrepancies', 'human', 'severe acute respiratory syndrome coronavirus 2']

    for line in csv.reader(data, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        keywords = line[0].split(', ')
        sentence = line[1].strip()
        keywords_list = []
        for k in keywords:
            if k not in stoplist:
                keywords_list.append(k)
        search_keywords[sentence] = keywords_list

    print("finished loading keywords...")

def filter_candidate_docs(candidate_docs, search_keywords, MIN_KEYWORDS=3):
    """
    Filter the list of candidate sentences to just those that have at least MIN_KEYWORDS matching the DHS sentence;
    if we don't do this, SBERT will take too long to run. Generally would like to keep the number of
    sentences that we claim match against less than a few thousand in most cases.

    Arguments:
        [dict] candidate_docs : a list of size two, who's first element contains all the sentences we could match against
        [dict] search_keywords : a dictionary that maps a sentence to its list of non-stopword keywords
        [int] MIN_KEYWORDS: the minimum number of keywords that need to match between DHS and CORD19 sentences to keep the latter
    """

    filtered_candidate_docs = []
    ctr = 0
    print("Filtering " + str(len(candidate_docs[0])) + " candidate_docs...")
    for doc in candidate_docs[0]:
        if ctr % 50000000 == 0:
            print(ctr)

        # only keep sentences that have at least MIN_KEYWORDS in common with the DHS sentence keywords
        found_key = 0
        matches = []
        for k in search_keywords:
            if k.lower() in doc.lower():
                found_key += 1 + k.count(' ')   # if the keywords is made up of multiple words, add them to the count
                matches.append(k.lower())
            if found_key >= MIN_KEYWORDS:
                filtered_candidate_docs.append(doc)
                break
        ctr += 1

    print("done filtering candidate_docs from " + str(len(candidate_docs)) + " to " + str(len(filtered_candidate_docs)) + "...")
    return filtered_candidate_docs, matches

def find_nearest_claims(search_set, candidate_set, cfg):
    """
    Finds and prints the nearest claims for each tweet. Serves as a
    controller function that dispatches work to various other helper
    functions.

    :param search_set: string denoting what data to use for the search set
    :param candidate_set: string denoting what data to use for the candidate set
    :param prune_duplicates: boolean denoting whether or not to remove duplicates from the search and candidate sets
    :param multimodal: boolean denoting whether or not multimodal data should be used
    :param cfg: configuration dictionary
    :return: none
    """

    # wipe and prepare the csv format for results (uses tab instead of comma because sentences might have commas)
    file = open(cfg['results_csv_path'],'w')
    file.close()
    file = open(cfg['results_csv_path'], "a")
    header = "question\tground_truth\tmatched_claim\tmatched_claim_paperID\tground_truth_paperID\tmatched_date\t"
    header += "matched_authors\tmatched_journal\tmatched_title\tmatched_affiliation\tdistance"
    file.write(header + "\n")
    file.close()

    # load the json files (sentences from articles plus metadata) from CORD19 and their associated mappings to the DHS sentences
    misc_data, article_metadata, question_mapping = None, None, None
    if candidate_set == 'misc_json' or search_set == 'misc_json':
        misc_data = [misc_point.get('content') for misc_point in MiscLoader.get_json_data(cfg)]
    if candidate_set == 'cord19_json' or search_set == 'cord19_json':
        sentence_to_paperID = {}
        cord19_data, article_metadata, title_to_id, question_mapping = Cord19Loader.get_json_data(cfg, sentence_to_paperID)
        DHS_sent_to_paperID_mapping = create_paperID_to_citation_mapping(title_to_id, cfg)

        file = open(cfg['DHS_path'])
        data = file.readlines()
        file.close()
        hasCitation = 0
        for sentence in data:
            if sentence.replace('  - ', '').strip() in DHS_sent_to_paperID_mapping.keys() or sentence.replace('  - ', '') in DHS_sent_to_paperID_mapping.keys():
                hasCitation += 1
        print("Percent of DHS sentences that have a citation:", str(hasCitation * 1.0 / (len(data) - 16)))

    # for the paper, count how many of the DHS citations were matched in the CORD19 dataset; we have to rely on 
    # the titles matching here, as these two datasets use different paper IDs
    #make_known_citation_folder(DHS_sent_to_paperID_mapping)

    # setup candidate docs (the things that might be matched against our ground truth); if CORD19 data, we need
    # to not only pass the sentences, but also the mapping from sentence to paperID in order to pull paper 
    # metadata later for our reporting
    candidate_docs = None
    if candidate_set == 'misc_json':
        candidate_docs = misc_data
    elif candidate_set == 'cord19_json':
        candidate_docs = [cord19_data, sentence_to_paperID]

    # set up the search docs; these are the DHS sentences, location specified in the config file
    search_docs = None
    if search_set == 'manual':
        search_docs = cfg['sentences']
    elif search_set == 'misc_json':
        search_docs = misc_data

    print("Retrieving nearest with", len(search_docs), "search documents and", len(candidate_docs[0]),
          "candidate documents.")

    # load the keywords for each DHS sentence
    search_keywords = {}
    load_search_keywords(search_keywords, cfg)

    # log all the DHS sentences that had to be skipped because of too many or too few CORD19 sentences that were filtered
    # to match by keyword for that DHS sentence
    skipped = open(cfg['skipped_path'],"w")

    # go through every DHS sentence, filter the CORD19 sentences by overlapping DHS keywords, find the nearest matches, 
    # write the results to files (including logging files if unable to match).
    for search in search_docs:
        print('DHS sentence', search)
        try:
            MIN_KEYWORDS = 3
            local_candidate_docs, keyword_matches = [filter_candidate_docs(candidate_docs, search_keywords[search], MIN_KEYWORDS), candidate_docs[1]]
            print('keyword matched sentences', len(local_candidate_docs[0]))

            # if there are an ureasonable number of CORD19 sentences to process, log and skip this sentence
            # later, we would manually edit the keywords for these sentences and try again to match/filter
            # otherwise, do the SBERT matching
            if len(local_candidate_docs[0]) > 80000:
                skipped.write(search + "\n")
                skipped.write(str(search_keywords[search]) + "\n")
                skipped.write(str(len(local_candidate_docs)) + "\n")
                skipped.write("--------------------------------------------------------------------")
                print("TOO LONG")
                continue
            else:
                retrieve_nearest([search], local_candidate_docs, keyword_matches, search_set, candidate_set, cfg, article_metadata, DHS_sent_to_paperID_mapping, question_mapping)
        except Exception as e: # other exceptions due to inability to match any sentences, etc: log and reprocess later
            print(e)
            traceback.print_exc() 
            skipped.write("FAILED")
            skipped.write(search + "\n")
            skipped.write(str(e) + "\n")
            skipped.write(str(search_keywords[search]) + "\n")
            skipped.write(str(len(local_candidate_docs)) + "\n")
            skipped.write("--------------------------------------------------------------------")
    skipped.close()


