# A Word Puzzle: a grid of randomly chosen
# letters with rules for creating words out of the 
# letters; this puzzle is known as "Target."
# +---+---+---+  How many words of (1) four letters or more
# | e | g | i |  from those shown here? (2) Each letter may
# +---+---+---+  be used once per word. (3) Each word must
# | v | r | v |  contain the center letter and (4)there
# +---+---+---+  must be at least one nine-letter word.
# | o | n | l |  21 words, good;
# 32 words, very good; 42 words, excellent.
import nltk
nltk.download('words')

wordlist = nltk.corpus.words.words()
puzzleLetters = nltk.FreqDist('egivrvonl')
# print(puzzleLetters.keys())
must_contain = 'r'
words = [w for w in wordlist if (must_contain in w) and (len(w) >= 4) and nltk.FreqDist(w) <= puzzleLetters]
# 'e', 'g', 'i', 'v', 'r', 'v', 'o', 'n', 'l' 각각의 철자를 포함하고 있으면 True, 앞의 10개의 letter중 혹은 이 외의 문자가 하나라도 더 추가되면 False 리턴
# nltk.FreqDist('v') <= puzzleLetters 
# puzzleLetters
print(words)
print(len(words))