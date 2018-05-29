from bs4 import BeautifulSoup
import requests
import pandas as pd
import pickle
from unidecode import unidecode
from nltk.tokenize import word_tokenize
#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from train_predict import train_predict



titles = []
summaries = []

skips = 'Ââ-'

letters = 'ABCDEFGHIJKLMNOPQRSTUVWY'
for l in letters:
    #summaries = []
    #titles = []
    print (l)
    soup_page = BeautifulSoup(requests.get("http://laws.justice.gc.ca/eng/acts/"+l+".html").text,"html.parser")
    aaa = [a for a in soup_page.find_all('a') if hasattr(a,'class')]
    links = [a['href'].split('/')[0] for a in soup_page.find_all('a') if  a.get_attribute_list('class')[0]=="TocTitle"]
    #links = [a for a in soup_page.find_all('a') if a.has_attribute
    for u in links:
        with open("html/"+u+".html", "r", encoding='UTF8') as fd:
            soup = BeautifulSoup(fd.read(), 'html.parser')
        #url = "http://laws.justice.gc.ca/eng/acts/"+u+"/FullText.html"
        #r  = requests.get(url)

        #data = r.text

        #soup = BeautifulSoup(data,'html.parser')
        small_titles = soup.find_all('h6')

        titl = []
        summary = []

        for ts in small_titles:
             if ts.next_sibling is not None:
                 if ts.contents[len(ts.contents)-1].string is not None:
                     titl.append(ts.contents[len(ts.contents)-1].string)
                     summary.append(' '.join(ts.next_sibling.find_all(text=True)))
                 else:
                     print ('flag2')
             else:
                 print('flag1')
                 
        #titl = [next_t(t) for t in small_titles]
       
        
        #summary = [' '.join(next_sib(s)) for s in small_titles]
        
        for sk in skips:
            titl = [t.replace(sk,' ') for t in titl]
            summary = [s.replace(sk,' ') for s in summary]
        #summary = [s.next_sibling.text.strip() for s in small_titles]
        #summary = [s.next_sibling.text for s in small_titles]
        for t in titl:
            if 'afteracquired' in unidecode(' '.join(word_tokenize(t))):
                kk = t
                print(t)
        if (len(titl)>0):
            titles.append(titl)
            summaries.append(summary)

#works
title_set = [t for tx in titles for t in tx]
summary_set = [s for sx in summaries for s in sx]
with open("data/actsT2.txt", 'w') as the_file:
    for t in title_set:            
        the_file.write(unidecode(' '.join(word_tokenize(t)))+'\n')
with open("data/actsS2.txt", 'w') as the_file:
    for s in summary_set:    
        the_file.write(unidecode(' '.join(word_tokenize(s)))+'\n')
