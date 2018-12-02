from client import *

files = ['arctic_a0001.wav']
model = 'models/output_graph.pbmm'
alphabet = 'models/alphabet.txt'
lm = 'models/lm.binary'
trie =  'models/trie'

for audio in files:
	predict(model, alphabet, lm, trie, audio)
		
