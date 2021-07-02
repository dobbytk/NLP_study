# Part of Speech tagging (POS tag)
import nltk
nltk.download('punkt')      # '/root/nltk_data/tokenizers'
nltk.download('averaged_perceptron_tagger')

sentence = """
Natural language processing (NLP) is a subfield of computer science, information engineering, 
and artificial intelligence concerned with the interactions between computers and human (natural) languages, 
in particular how to program computers to process and analyze large amounts of natural language data.
"""
word_tok = nltk.word_tokenize(sentence)
print(word_tok)

nltk.pos_tag(word_tok)

# 명사와 형용사만 표시한다.
sent_nnjj = [word for word, pos in nltk.pos_tag(word_tok) if pos == 'NN' or pos == 'JJ']
sent_nnjj

# bigram
bigram = [(a, b) for a, b in nltk.bigrams(sent_nnjj)]
bigram

# trigram
trigram = [(a, b, c) for a, b, c in nltk.trigrams(sent_nnjj)]
trigram

# n-gram
ngram = [(a, b, c, d) for a, b, c, d in nltk.ngrams(sent_nnjj, 4)]
ngram

# 명사 앞에 오는 품사를 확인한다.
bigram_pos = [(a, b) for a, b in nltk.bigrams(nltk.pos_tag(word_tok))]
noun_preceders = [a[1] for a, b in bigram_pos if b[1] == 'NN']

fdist = nltk.FreqDist(noun_preceders)
[(tag, cnt) for (tag, cnt) in fdist.most_common()]

bigram

# 사전 (vocabulary) 생성
word2idx = {}
n_idx = 0
for sent in all_tokens:
    for word in sent:
        if word.lower() not in word2idx:
            word2idx[word.lower()] = n_idx
            n_idx += 1
idx2word = {v:k for k, v in word2idx.items()}

word2idx

# text를 사전의 index로 표현
text_idx = []
for sent in all_tokens:
    sent_idx = []
    for word in sent:
        sent_idx.append(word2idx[word.lower()])
    text_idx.append(sent_idx)
print(text_idx[0])

# text_idx를 다시 단어로 표시
text = []
for sent_idx in text_idx:
    sent = []
    for word_idx in sent_idx:
        sent.append(idx2word[word_idx])
    text.append(sent)
print(text[0])

