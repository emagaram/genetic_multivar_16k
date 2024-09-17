import words

corpus_freqs = words.CorpusFrequencies()
bgs = words.get_bigrams()
th = bgs["th"]
tgs = words.get_trigrams()
s = 0
for word, freq in tgs.items():
    if word.find("th") != -1:
       s+=freq

print("TH:", th)

print("s:", s)