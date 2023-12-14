from myfunc import *

predFile=''
pub0File=''
pubFile=''
pred=ReadCSV(predFile)
pub0=ReadCSV(pub0File)
pub=ReadCSV(pubFile)
p0=set([i['label'] for i in pub0)
p=set([i['label'] for i in pub)
q=p0-p
for i 
