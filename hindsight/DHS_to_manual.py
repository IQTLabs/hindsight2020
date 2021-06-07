'''
	script to convert the raw DHS sentences with citations into just sentences (without citations) that can be copied into the
	yaml file for further processing.
'''

file = open('DHS_raw.txt')
data = file.readlines()
file.close()

result = open('DHS_processed.txt','w')
for d in data:
	d = d.split("#")
	if len(d[0]) > 0:
		result.write('  - ' + d[0] + '\n')
result.close()