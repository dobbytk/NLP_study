# Regular Expression 연습
import nltk
import re
nltk.download('words')
nltk.download('nps_chat')
nltk.download('treebank')

# 영어 단어 목록
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
len(wordlist)

# ed로 끝나는 단어
v = [w for w in wordlist if re.search('ed$', w)]
v[:20]

v = [w for w in wordlist if re.search('^..j..t..$', w)]
print(v[:20])

# 1. 전화기 문자판 (textonyms) - 완료
v = [w for w in wordlist if re.search('^[ghi][mno][jkl][def]', w) and len(w) == 4]
print(v[:20])
# ['gold', 'golf', 'hold', 'hole']

# 2. NPS Chat corpus - 완료
chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
v = [w for w in chat_words if re.search('^m+i+n+e+$', w)]
print(v[:20])
# ['miiiiiiiiiiiiinnnnnnnnnnneeeeeeeeee', 'miiiiiinnnnnnnnnneeeeeeee', 'mine', 'mmmmmmmmiiiiiiiiinnnnnnnnneeeeeeee']

# 3 - 완료
v = [w for w in chat_words if re.search('^[ha][ha]*$', w)]
print(v[:20])
# ['a', 'aaaaaaaaaaaaaaaaa', 'aaahhhh', 'ah', 'ahah', 'ahahah', 'ahh', 'ahhahahaha', 'ahhh', 'ahhhh', 'ahhhhhh', 
# 'ahhhhhhhhhhhhhh', 'h', 'ha', 'haaa', 'hah', 'haha', 'hahaaa', 'hahah', 'hahaha']

# 4. Penn Treebank corpus - 완료
wsj = set(nltk.corpus.treebank.words())
v = [w for w in wsj if re.search('\d+[.]\d+', w)]
print(v[:30])
# ['7.55', '14.99', '9.9', '319.75', '3.42', '7.422', '251.2', '361.8', '1.92', '6.84', '2.62', '3.2', '446.62', 
# '2.8', '1.125', '8.575', '4.7', '737.5', '16.9', '84.29', '8.55', '7.60', '85.1', '1.19', '38.3', '4.3', '8.14', 
# '9.82', '8.60', '1.2']

# 5. - 완료
v = [w for w in wsj if re.search('[A-Z]\$$', w)]
print(v[:20])
# ['C$', 'US$']

# 6 - 완료
v = [w for w in wsj if re.search('^[0-9][0-9][0-9][0-9]$', w)] 
print(v[:20])
# ['1901', '1987', '1988', '2000', '1992', '1979', '1965', '1787', 
# '1969', '1967', '1970', '1956', '1990', '1983', '1977', '1994', '1993', '1996', '1986', '1980']

# 7 - 완료
v = [w for w in wsj if re.search('^\d+-[a-z]*$', w)]
print(v[:20])
# ['14-hour', '27-year', '100-megabyte', '300-day', '36-day', '237-seat', '10-year', '10-lap', '42-year', 
# '10-day', '52-week', '69-point', '12-point', '500-stock', '15-day', '240-page', '30-minute', '150-point', 
# '520-lawyer', '30-share']

# 8 - 완료
v = [w for w in wsj if re.search('^[a-z]*-[a-z]-[a-z]*$', w)]
print(v[:20])
# ['million-a-year', 'tete-a-tete', 'cents-a-unit']

# 9 - 완료
v = [w for w in wsj if re.search('ed$|ing$', w)]
print(v[:20])
# ['deteriorating', 'Minneapolis-based', 'bounced', 'substance-abusing', 'Baking', 'swapped', 'showing', 'confused', 
# 'Asked', 'fined', 'overcrowding', 'accepting', 'murdered', 'overleveraged', 'Traded', 'included', 'characterized', 
# 'Everything', 'compelling', 'clicked']