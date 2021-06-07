    
# (certainty) kdobolyi@mp-gpu-server04:~/uncertainty$ python main.py predict --granularity=sentence "I wonder if I am the walrus."
# (entailment) python predict.py data/models/mednli.infersent.glovebioasqmimic.128.n8d0l13c.pkl data/inputs.txt data/predictions_test.csv

'''
Script for calling the certainty-labelling code; requires an input.txt of sentences to label.

'''

import subprocess
import traceback

file = open("./inputs.txt")
data = file.readlines()
file.close()

print('processing ' + str(len(data)) + " sentences...")
results = open("results.txt", "w")
ctr = 0
for d in data:
	print(str(ctr) + " out of " + str(len(data)))
	try:
		output = subprocess.check_output("python -W ignore main.py predict --granularity=sentence '" + d[:-1] + "'", shell=True)
		results.write(str(output) + "\n")
		print(str(output))
	except Exception as e:
		traceback.print_exc() 
		print("FAILED on ", ctr)
		print(d)
		break
	ctr += 1
results.close()

