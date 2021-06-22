"""
This file loads the CORD19 metadata using a pre-specified configuration (passed in
as a cfg dictionary to the functions below)

@author: kdobolyi
"""
import json, os, sys, csv

def load_CORD19_metadata(cfg):
    '''
    Returns the article metadata from the CORD19 dataset path specified in cfg. Note that one article may
    be associated with multiple sha-s (unique ids) -- this code creates an entry for each unique sha.

    Arguments:
        [dict] cfg :  the configuration file that points to the metadata_path of the CORD19 metadata csv
                    specified in main.py via config.yml
    Returns:
        [dict] article_metadata :   a dict with the article sha (unique ID) as the key, and the values are
                    dictionaries of article metadata for each unique sha.
        [dict] title_to_id :        a dict that assosicates a list of sha-s with each unique article title
    '''

    file = open(cfg['metadata_path'])
    data = file.readlines()[1:]
    file.close()

    article_metadata = {}
    title_to_id = {}

    for line in csv.reader(data, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        article = {}
        article['shas'] = line[1]
        article['date'] = line[9]
        article['authors'] = line[10]
        article['journal'] = line[11]
        article['title'] = line[3]

        # there may be multiple shas per unique article; we will make an entry for all of them
        for sha in article['shas'].split(';'):
            article_metadata[sha.strip()] = article

        # associate each unique title with its shas
        title_to_id[article['title']] = article['shas']

    return article_metadata, title_to_id

def load_questions_mapping(cfg):
    '''
    Maps each DHS sentence to its corresponding DHS question, using the path to the DHS
    sentences specified in the config file [cfg]. Prints out the number of DHS citations that appear
    in the CORD19 dataset

    Arguments:
        [dict] cfg :  the configuration file that points to the DHS_path of the sentences + questions mined from
                            DHS' updates, this path is specified in main.py via config.yml
    Returns:
        [dict] question_mapping :   maps each DHS sentence to its corresponding DHS question
    '''
    file = open(cfg['DHS_path'])
    data = file.readlines()
    file.close()

    # DHS questions
    questions = [

        '1. Infectious Dose – How much agent will make a healthy individual ill?',
        '2. Transmissibility – How does it spread from one host to another? How easily is it spread?',
        '3. Host Range – How many species does it infect? Can it transfer from species to species?',
        '4. Incubation Period – How long after infection do symptoms appear? Are people infectious during this time?',
        '5. Clinical Presentation – What are the signs and symptoms of an infected person?',
        '6. Protective Immunity – How long does the immune response provide protection from reinfection?',
        '7. Clinical Diagnosis – Are there tools to diagnose infected individuals? When during infection are they effective?',
        '8. Medical Treatments – Are there effective treatments?',
        '9. Vaccines – Are there effective vaccines?',
        '10. Non-pharmaceutical Interventions (NPIs) – Are public health control measures effective at reducing spread?',
        '11. Environmental Stability – How long does the agent live in the environment?',
        '12. Decontamination – What are effective methods to kill the agent in the environment?',
        '13. PPE – What PPE is effective, and who should be using it?',
        '14. Forensics – Natural vs intentional use? Tests to be used for attribution.',
        '15. Genomics – How does the disease agent compare to previous strains?',
        '16. Forecasting – What forecasting models and methods exist?'
        ]

    question_mapping = {}
    for d in data:
        d = d.replace(" - ", '').strip()
        for q in questions:
            if d in q:
                current_question = q
        question_mapping[d[:]] = current_question

    return question_mapping

def get_json_data(cfg, sentence_to_paperID):
    '''
    Processes the CORD19 articles and metadata (in json format) into the dataset formats needed for this library. Ignores
    any articles that are too old to be about COVID19 (pre 2020). Grabs all sentences from both the abstracts and the
    paper bodies.

    Arguments:
        [dict] cfg : the configuration file that points to the cord19_dir, where the CORD19 dataset has been downloaded;
                    we point it to the folder of jsons for pdfs in this download of CORD19
        [dict] sentence_to_paperID : maps each sentence from an academic article to the paper_ID of CORD19 (not the same as sha)
    Returns:
        [list] data : List of sentences mined from all CORD19 articles
        [dict] article_metadata : a dict with the article sha (unique ID) as the key, and the values are
                    dictionaries of article metadata for each unique sha.
        [dict] title_to_id : a dict that assosicates a list of sha-s with each unique article title
        [dict] question_mapping : maps each DHS sentence to its corresponding DHS question
        [dict] sentence_to_paperID : not explicitly returned, but this argument is updated in this function
    '''
    if not os.path.exists(cfg['cord19_dir']):
        print("Did you download the CORD19 dataset and its metadata.csv, and update their paths in ./config/config.yml? And/or, please ensure you followed the README directions for JSON data.")
        sys.exit(1)
    json_files = [os.path.join(cfg['cord19_dir'], file) for file in os.listdir(cfg['cord19_dir']) if file.endswith('.json')]
    if len(json_files) == 0:
        print("Did you download the CORD19 dataset and its metadata.csv, and update their paths in ./config/config.yml? And/or, please ensure you followed the README directions for JSON data.")
        sys.exit(1)

    question_mapping = load_questions_mapping(cfg)
    print("finished loading questions mapping...")
    
    article_metadata, title_to_id = load_CORD19_metadata(cfg)
    print("finished loading CORD19 metadata...")

    data = []
    ctr = 0
    for file in json_files:
        with open(file) as f:
            dict_json = json.load(f)

            # article is too old to be about covid19
            if not (article_metadata[dict_json['paper_id']]['date'].endswith("20") 
                or article_metadata[dict_json['paper_id']]['date'].endswith("21")
                or "2020" in article_metadata[dict_json['paper_id']]['date']
                or "2021" in article_metadata[dict_json['paper_id']]['date']):
                continue

            # process the article body text
            body = dict_json['body_text']
            sentences = ""
            for b in body:
                sentences += b['text'] + " "

            # if the abstract exists, process it as well
            if len(dict_json['abstract']) != 0:
                sentences += dict_json['abstract'][0]['text']
            sentences = sentences.split(". ")
            paper_id = dict_json['paper_id']
            if ctr % 10000 == 0:
                print(str(ctr) + " out of " + str(len(json_files)))
            ctr += 1

            # map each sentence to its CORD19 paper_id
            for s in sentences:
                sentence_to_paperID[s] = paper_id
            data.extend(list(set(sentences)))

            # add additional article metadata
            org = ""
            for author in dict_json['metadata']['authors']:
                if 'affiliation' in author.keys():
                    affiliation = author['affiliation']
                    if 'laboratory' in affiliation.keys():
                        org = affiliation['laboratory'] + ';' 
                    if 'institution' in affiliation.keys():
                        org += affiliation['institution']
            if dict_json['paper_id'] in article_metadata.keys():
                article_metadata[dict_json['paper_id']]['affiliation'] = org

    print("finished loading CORD19 sentences to paperID mapping...")
    return data, article_metadata, title_to_id, question_mapping


