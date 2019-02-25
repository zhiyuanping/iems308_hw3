import numpy as np  
import matplotlib.pyplot as plt 
import glob 
import pandas as pd  
import re
import regex
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 

##load all emails
df = []
path1 = '/Users/daniel/Desktop/School_Work/IE/308/hw3/2013/*.txt'
files2013 = glob.glob(path1)
# iterate over the list getting each file 
for file in files2013:
   # open the file and then call .read() to get the text 
   with open(file, encoding="utf8", errors='ignore') as f:
      contents = f.read()
      df.append(contents)
path2 = '/Users/daniel/Desktop/School_Work/IE/308/hw3/2014/*.txt'
files2014 = glob.glob(path2)
# iterate over the list getting each file 
for file in files2014:
   # open the file and then call .read() to get the text 
   with open(file, encoding="utf8", errors='ignore') as f:
      contents = f.read()
      df.append(contents)

ceo_file = '/Users/daniel/Desktop/School_Work/IE/308/hw3/all/ceo.csv'
companies_file = '/Users/daniel/Desktop/School_Work/IE/308/hw3/all/companies.csv'
percentage_file = '/Users/daniel/Desktop/School_Work/IE/308/hw3/all/percentage.csv'
ceo = pd.read_csv(ceo_file, encoding = "ISO-8859-1", error_bad_lines=False,header = None)
com = pd.read_csv(companies_file, encoding = "ISO-8859-1", error_bad_lines=False,header = None)
per = pd.read_csv(percentage_file, encoding = "ISO-8859-1", error_bad_lines=False,header = None)

#sentence token
sent = []
stop_words = set(stopwords.words('english')) 
for i in range(len(df)):
    print(i)
    ss = sent_tokenize(df[i])
    for s in ss:
        ws = word_tokenize(s)
        #remove stop words and build the tokens
        for w in ws:
            if w not in stop_words:
                sent.append([s,w])

##ceo names
train_ceo =  pd.DataFrame(columns=["index","word","contain_ceo"])
counter1 = 0
ceo.fillna('', inplace=True)
for i in range(len(sent)):
    for j in range(len(ceo)):
        name = ceo.iloc[j,0]+" "+ceo.iloc[j,1]
        if sent[i][0].find(name)>=0:
            contain_ceo = sent[i][0].find("ceo")
            if contain_ceo == -1:
                contain_ceo = 0
            else:
                contain_ceo = 1
            train_ceo.loc[counter1] = [i,name,contain_ceo]
            counter1 = counter1+1
            
for i in range(len(train_ceo)):
    if sent[train_ceo.loc[i,"index"]][0].find("CEO")!=-1:
        train_ceo.loc[i,"contain_ceo"]=1
    st = train_ceo.loc[i,"word"]
    if st[-1] in ['a','e','i']:
        train_ceo.loc[i,"last_aei"] = 1
    else:
        train_ceo.loc[i,"last_aei"] = 0
    train_ceo.loc[i,"in_sample"] = 1
    train_ceo.loc[i,"res"] = 1

names = train_ceo.loc[:,"word"].values.tolist()
counter1 = len(train_ceo)
for i in range(len(sent)):
    print(i)
    n1 = re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+",sent[i][0])
    contain_ceo = sent[i][0].find("CEO")
    if contain_ceo == -1:
        contain_ceo = 0
    else:
        contain_ceo = 1
    if (len(n1)>0):
        for n in n1:
            l = len(n)
            sp = n.find(' ')
            if sp == -1:
                sp = 0
            else:
                sp = 1
            if n not in names:
                if n[-1] in ['a','e','i']:
                    train_ceo.loc[counter1] = [0,i,n,contain_ceo,1,0,0,l,sp]
                else:
                    train_ceo.loc[counter1] = [0,i,n,contain_ceo,0,0,0,l,sp]
                counter1 = counter1+1
        

for i in range(len(train_ceo)):
    train_ceo.loc[i,"length"]=len(train_ceo.loc[i,"word"])
    st = train_ceo.loc[i,"word"]
    sp = st.find(' ')
    if sp == -1:
        sp = 0
    else:
        sp = 1
    train_ceo.loc[i,"space"]=sp

for i in range(len(train_ceo)):
    if (sent[train_ceo.loc[i,"index"]][0].startswith(train_ceo.loc[i,"word"]) == True):
        train_ceo.loc[i,"start"] = 1
    else:
        train_ceo.loc[i,"start"] = 0

for i in range(len(train_ceo)):
    counts = re.findall(r"[A-Z][a-z]+",sent[train_ceo.loc[i,"index"]][0])
    train_ceo.loc[i,"num"] = len(counts)

path='/Users/daniel/Desktop/School_Work/IE/308/hw3/'
filename = "train_ceo1.csv"
train_ceo.to_csv(path+filename)
  
###use dataset processed
train_ceo = pd.read_csv("/Users/daniel/Desktop/School_Work/IE/308/hw3/train_ceo.csv")

data = train_ceo.loc[:,["contain_ceo","last_aei","length","space","start"]]
target = train_ceo.loc[:,"res"]
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(data, target).predict(data)
acc = np.sum(target == y_pred)/len(target)
baseline = np.sum(train_ceo.loc[:,"res"])/len(train_ceo)
print("For CEO names, the baseline is {} and the accuracy is {}".format(baseline, acc))

##company names
train_com =  pd.DataFrame(columns=["index","word","contain_1","contain_2", "contain_3","contain_4","contain_5","length","num_space","res"])
counter2 = 0
com.fillna('', inplace=True)
for i in range(len(sent)):
    print(i)
    for j in range(len(com)):
        name = com.iloc[j][0]
        if sent[i][0].find(name)>=0:
            contain_1 = sent[i][0].find("company")
            contain_2 = sent[i][0].find("Inc")
            contain_3 = sent[i][0].find("Ltd")
            contain_4 = sent[i][0].find("Co")
            contain_5 = sent[i][0].find("Group")
            
            if contain_1 == -1:
                contain_1 = 0
            else:
                contain_1 = 1
            if contain_2 == -1:
                contain_2 = 0
            else:
                contain_2 = 1
            if contain_3 == -1:
                contain_3 = 0
            else:
                contain_3 = 1
            if contain_4 == -1:
                contain_4 = 0
            else:
                contain_4 = 1
            if contain_5 == -1:
                contain_5 = 0
            else:
                contain_5 = 1
            l = len(name)
            a = re.findall(r" ",name)
            num = len(a)
            train_com.loc[counter2] = [i,name,contain_1,contain_2,contain_3,contain_4,contain_5,l,num,1]
            counter2 = counter2+1

names = train_com.loc[:,"word"].values.tolist()
nn = len(train_com)
counter2 = len(train_com)
for i in range(len(sent)):
    print(i)
    n1 = re.findall(r"([A-Z][\w-]*(\s+[A-Z][\w-]*)+)",sent[i][0])
    if (len(n1)>0):
        for n in n1[0]:
            if n not in names:
                contain_1 = sent[i][0].find("company")
                contain_2 = sent[i][0].find("Inc")
                contain_3 = sent[i][0].find("Ltd")
                contain_4 = sent[i][0].find("Co")
                contain_5 = sent[i][0].find("Group")
                if contain_1 == -1:
                    contain_1 = 0
                else:
                    contain_1 = 1
                if contain_2 == -1:
                    contain_2 = 0
                else:
                    contain_2 = 1
                if contain_3 == -1:
                    contain_3 = 0
                else:
                    contain_3 = 1
                if contain_4 == -1:
                    contain_4 = 0
                else:
                    contain_4 = 1
                if contain_5 == -1:
                    contain_5 = 0
                else:
                    contain_5 = 1
                    
                l = len(n)
                a = re.findall(r" ",n)
                num = len(a)
                train_com.loc[counter2] = [i,n,contain_1,contain_2,contain_3,contain_4,contain_5,l,num,0]
                counter2 = counter2+1
                
for i in range(len(train_com)):
    if (sent[train_com.loc[i,"index"]][0].startswith(train_com.loc[i,"word"]) == True):
        train_com.loc[i,"start"] = 1
    else:
        train_com.loc[i,"start"] = 0
    if (sent[train_com.loc[i,"index"]][0].endswith(train_com.loc[i,"word"]) == True):
        train_com.loc[i,"end"] = 1
    else:
        train_com.loc[i,"end"] = 0
        
filename = "train_com.csv"
train_com.to_csv(filename)
    
#use processed data
train_com = pd.read_csv("train_com.csv")
data = train_com.loc[:,["contain_1","contain_2", "contain_3","contain_4","contain_5","length","num_space","start","end"]]
target = train_com.loc[:,"res"]
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(data, target).predict(data)
baseline = np.sum(train_com.loc[:,"res"])/len(train_com)
acc1 = np.sum(target == y_pred)/len(target)
print("For company names, the baseline is {} and the accuracy is {}".format(baseline, acc))

#regex for percentage extraction
p_num = []
p_mix = []
p_word = []
p_special = []
for i in range(len(df)):
    r1 = re.findall(r" ?[(]?[-]?\d+\.?\d+%[)]?",df[i])
    for rr in r1:
        p_num.append(rr)
        
    r2 = re.findall(r" [-]?\d+\.?\d+[- ]percent",df[i])
    for rr in r2:
        p_mix.append(rr)
    
    r3 = regex.findall(r"(?:(?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight)|zero|(?:s(?:even|ix)|t(?:hir|wen)|f(?:if|or)|eigh|nine)ty(?:[ -](?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight))?|(?:(?:(?:s(?:even|ix)|f(?:our|if)|nine)te|e(?:ighte|lev))en|t(?:(?:hirte)?en|welve))|(?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight))(?: point(?: (?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight)|zero)+)?[- ]percent",df[i])
    for rr in r3:
        p_word.append(rr)
        
    # n-n
    s1 = re.findall(r"\d+\.?\d\-\d+\.?\d+[- ]percent",df[i])
    s2 = re.findall(r"\d+\.?\d\-\d+\.?\d+\%",df[i])
    s3 = re.findall(r"\d+\.?\d\sto\s\d+\.?\d+[- ]percent",df[i])
    s4 = re.findall(r"\d+\.?\d\sto\s\d+\.?\d+\%",df[i])
    
    # percentage points
    s5 = re.findall(r" [-]?\d+\.?\d+[- ]percentage points",df[i])
    s6 = regex.findall(r"(?:(?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight)|zero|(?:s(?:even|ix)|t(?:hir|wen)|f(?:if|or)|eigh|nine)ty(?:[ -](?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight))?|(?:(?:(?:s(?:even|ix)|f(?:our|if)|nine)te|e(?:ighte|lev))en|t(?:(?:hirte)?en|welve))|(?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight))(?: point(?: (?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight)|zero)+)?[- ]percentage points",df[i])
    
    #fraction
    s7 = re.findall(r" [(]?[-]?\d+[\-\+]?\d+\/\d+%[)]?",df[i])
    s8 = re.findall(r" [(]?[\-\+]?\d+\/\d+%[)]?",df[i])
    for rr in s1,s2,s3,s4,s5,s6:
        p_special.append(rr)
        

p_s = []
for pp in p_special:
    if len(pp)>=0:
        for ppp in pp:
            p_s.append(ppp)

percentages = np.concatenate([p_num, p_mix, p_word, p_s])
perc= pd.DataFrame(percentages)

path =  "/Users/daniel/Desktop/School_Work/IE/308/hw3/"
filename = "output_per.csv"
perc.to_csv(path+filename)

names = per.values.tolist()

samples = []
for n in names:
    samples.append(n[0])
    
for i in range(len(perc)):
    if perc.loc[i,0] in samples:
        perc.loc[i,"res"] = 1
    else:
        perc.loc[i,"res"] = 0

acc = np.sum(perc.loc[:,"res"])/len(perc)
print("For percentages, the accuracy is {}".format(acc))

#use processed
train_per = pd.read_csv("train_per.csv")

data = train_com.loc[:,["contain_1","contain_2", "contain_3","contain_4","contain_5","length","num_space"]]
target = train_com.loc[:,"res"]
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(data, target).predict(data)
baseline = np.sum(train_com.loc[:,"res"])/len(train_com)
acc = np.sum(target == y_pred)/len(target)
print("the baseline is {} and the accuracy is {}".format(baseline, acc))

##produce outputs

out_ceo = pd.DataFrame(columns=["index","word","contain_ceo","last_aei","length","space","start"])
counter1 = 0
for i in range(len(sent)):
    print(i)
    n1 = re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+",sent[i][0])
    f = re.findall(r"CEO",sent[i][0])
    if len(f)>0:
        contain_ceo = 1
    else:
        contain_ceo = 0
    if (len(n1)>0):
        for n in n1:
            l = len(n)
            sp = n.find(' ')
            if sp == -1:
                sp = 0
            else:
                sp = 1
            if n[-1] in ['a','e','i']:
                last = 1
            else:
                last = 0
            if (sent[i][0].startswith(n) == True):
                start = 1
            else:
                start = 0
            out_ceo.loc[counter1] = [i,n,contain_ceo,last,l,sp,start]
            counter1 = counter1+1
            
filename = "out_ceo1.csv"
out_ceo.to_csv(filename)


out_com = pd.DataFrame(columns=["index","word","contain_1","contain_2", "contain_3","contain_4","contain_5","length","num_space","start","end"])
counter2 = 0
for i in range(len(sent)):
    print(i)
    n1 = re.findall(r"([A-Z][\w-]*(\s+[A-Z][\w-]*)+)",sent[i][0])
    if (len(n1)>0):
        for n in n1[0]:
            contain_1 = sent[i][0].find("company")
            contain_2 = sent[i][0].find("Inc")
            contain_3 = sent[i][0].find("Ltd")
            contain_4 = sent[i][0].find("Co")
            contain_5 = sent[i][0].find("Group")
            if contain_1 == -1:
                contain_1 = 0
            else:
                contain_1 = 1
            if contain_2 == -1:
                contain_2 = 0
            else:
                contain_2 = 1
            if contain_3 == -1:
                contain_3 = 0
            else:
                contain_3 = 1
            if contain_4 == -1:
                contain_4 = 0
            else:
                contain_4 = 1
            if contain_5 == -1:
                contain_5 = 0
            else:
                contain_5 = 1
                
            l = len(n)
            a = re.findall(r" ",n)
            num = len(a)
            
            if (sent[i][0].startswith(n) == True):
                start = 1
            else:
                start = 0
            if (sent[i][0].endswith(n) == True):
                end = 1
            else:
                end = 0
            
            out_com.loc[counter2] = [i,n,contain_1,contain_2,contain_3,contain_4,contain_5,l,num,start,end]
            counter2 = counter2+1
            
filename = "out_com.csv"
out_com.to_csv(filename)




##naive bayes predict and select
from sklearn.metrics import precision_recall_fscore_support
train_ceo = pd.read_csv("/Users/daniel/Desktop/School_Work/IE/308/hw3/train_ceo.csv")
n1 = int(np.sum(train_ceo.loc[:,"res"]))
n2 = int(len(train_ceo)-n1)
train_ceo.dropna()
if n1<n2:
    size = range(2*n1)
else:
    size = range(n1-n2,n1+n2)
data_ceo = train_ceo.loc[size,["contain_ceo","last_aei","length","space","start"]]
target_ceo = train_ceo.loc[size,"res"]
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model_ceo_nb = gnb.fit(data_ceo, target_ceo)
y_pred1 = model_ceo_nb.predict(data_ceo)
acc1 = np.sum(target_ceo == y_pred1)/len(target_ceo)
a = precision_recall_fscore_support(target_ceo, y_pred1, average='binary')
print("Naive Bayes with 1 to 1 postive and negative samples: For ceo names, the precision is {}, the recall is {}, and the f1 score is {}.".format(a[0],a[1],a[2]))

train_com = pd.read_csv("/Users/daniel/Desktop/School_Work/IE/308/hw3/train_com.csv")
train_com.dropna()
n1 = int(np.sum(train_com.loc[:,"res"]))
n2 = int(len(train_com)-n1)
if n1<n2:
    size = range(2*n1)
else:
    size = range(n1-n2,n1+n2)
data_com = train_com.loc[size,["contain_1","contain_2", "contain_3","contain_4","contain_5","length","num_space","start","end"]]
target_com = train_com.loc[size,"res"]

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model_com_nb = gnb.fit(data_com, target_com)
y_pred2 = model_com_nb.predict(data_com)
acc2 = np.sum(target_com == y_pred2)/len(target_com)
a = precision_recall_fscore_support(target_com, y_pred2, average='binary')
print("Naive Bayes with 1 to 1 postive and negative samples: For company names, the precision is {}, the recall is {}, and the f1 score is {}.".format(a[0],a[1],a[2]))


out_ceo = pd.read_csv("/Users/daniel/Desktop/School_Work/IE/308/hw3/out_ceo.csv")
out_com = pd.read_csv("/Users/daniel/Desktop/School_Work/IE/308/hw3/out_com.csv")

data_ceo_r = out_ceo.loc[:,["contain_ceo","last_aei","length","space","start"]]
data_com_r = out_com.loc[:,["contain_1","contain_2", "contain_3","contain_4","contain_5","length","num_space","start","end"]]
res_ceo_nb = model_ceo_nb.predict(data_ceo_r)
res_com_nb = model_com_nb.predict(data_com_r)

output_ceo_nb =  out_ceo.loc[res_ceo_nb==1,["index","word"]]
output_com_nb =  out_com.loc[res_com_nb==1,["index","word"]]

path='/Users/daniel/Desktop/School_Work/IE/308/hw3/'
filename = "output_ceo_nb.csv"
#output_ceo_nb.to_csv(path+filename)
filename = "output_com_nb.csv"
#output_com_nb.to_csv(path+filename)

##logistic regression
from sklearn.linear_model import LogisticRegression
model_ceo_lr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(data_ceo, target_ceo)
y_pred3 = model_ceo_lr.predict(data_ceo)
acc3 = np.sum(target_ceo == y_pred3)/len(target_ceo)
a = precision_recall_fscore_support(target_ceo, y_pred3, average='binary')
print("Logistic Regression with 1 to 1 postive and negative samples: For ceo names, the precision is {}, the recall is {}, and the f1 score is {}.".format(a[0],a[1],a[2]))


model_com_lr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(data_com, target_com)
y_pred4 = model_com_lr.predict(data_com)
acc4 = np.sum(target_com == y_pred4)/len(target_com)
a = precision_recall_fscore_support(target_com, y_pred4, average='binary')
print("Logistic Regression with 1 to 1 postive and negative samples: For company names, the precision is {}, the recall is {}, and the f1 score is {}.".format(a[0],a[1],a[2]))


res_ceo_lr = model_ceo_lr.predict(data_ceo_r)
res_com_lr = model_com_lr.predict(data_com_r)

output_ceo_lr =  out_ceo.loc[res_ceo_lr==1,["index","word"]]
output_com_lr =  out_com.loc[res_com_lr==1,["index","word"]]

path='/Users/daniel/Desktop/School_Work/IE/308/hw3/'
filename = "output_ceo_lr.csv"
#output_ceo_lr.to_csv(path+filename)
filename = "output_com_lr.csv"
#output_com_lr.to_csv(path+filename)
