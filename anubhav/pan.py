import urllib3, pandoc

url = 'http://mcdaniel.blogs.rice.edu/?p=158'
response = urllib3.urlopen(url).read()

# The first line creates a pandoc object. The second assigns the html response to that object
doc = pandoc.Document()
doc.html = response

f = open('wendell-phillips.txt','w').write(doc.markdown)
