from client import *
import os 
import csv

audio_dir = 'cmu_us_bdl_arctic/wav/'
audio_files  = os.listdir(audio_dir)

model = 'models/output_graph.pbmm'
alphabet = 'models/alphabet.txt'
lm = 'models/lm.binary'
trie =  'models/trie'
f = open('result.csv', 'w')
 
for idx, audio in enumerate(audio_files):
	text   = predict(model, alphabet, lm, trie, os.path.join(audio_dir, audio))
	writer = csv.writer(f)
	writer.writerow([str(idx), text])
	print('Done convering audio :', idx)
	
