from bs4 import BeautifulSoup
import requests

import urllib 


titles = []
summaries = []


letters = 'ABCDEFGHIJKLMNOPQRSTUVWY'
for l in letters:
    #summaries = []
    #titles = []
    print (l)
    soup_page = BeautifulSoup(requests.get("http://laws.justice.gc.ca/eng/acts"+l+".html").text,"html.parser")
    aaa = [a for a in soup_page.find_all('a') if hasattr(a,'class')]
    links = [a['href'].split('/')[0] for a in soup_page.find_all('a') if  a.get_attribute_list('class')[0]=="TocTitle"]
    #links = [a for a in soup_page.find_all('a') if a.has_attribute
    for u in links:
        url = "http://laws.justice.gc.ca/eng/regulations/"+u+"/FullText.html"
        urllib.request.urlretrieve(url,"html/"+u+".html" )
