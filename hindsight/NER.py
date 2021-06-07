# https://allenai.github.io/scispacy/
'''
	Processes DHS sentences into a file of keywords,sentences for each sentence. We typically then
	convert these to a desired csv format using Excel.

'''

import scispacy
import spacy

nlp = spacy.load("en_core_sci_sm")

file = open("./DHS_processed.txt")
data = file.readlines()
file.close()

file = open("DHS_NERs.txt", "w")
for text in data:
     doc = nlp(text)
     result = (str(doc.ents)[1:-1] + "###" + text[:-1])
     file.write(result + "\n")
file.close()

