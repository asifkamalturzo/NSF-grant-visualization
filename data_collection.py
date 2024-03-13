import requests
import os
import csv 
from zipfile import ZipFile
import xml.etree.ElementTree as ET


def file_download():
    path = "datasets"
    isExist = os.path.exists(path)
    if not isExist:
       os.makedirs(path)
       print("The new directory is created!")

    for i in range(2000,2022):
        URL = "https://www.nsf.gov/awardsearch/download?DownloadFileName={0}&All=true".format(i)
        # Download the data behind the URL
        response = requests.get(URL)
        # Open the response into a new file 
        file_name = "datasets/{0}.zip".format(i)
        
        open(file_name, "wb").write(response.content)
        print("File download complete for year {0}".format(i))


def extract_zip():
    with os.scandir('datasets/') as entries:
        for entry in entries:
            if(".zip" in entry.name):
                with ZipFile('datasets/'+entry.name, 'r') as zipObj:
                    strn = entry.name.split(".")
                    zipObj.extractall('datasets/'+strn[0])


def parseXML(xmlfile,headers = []):
  
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    file_name="dataset.csv"
    exists = os.path.exists(file_name)
    row_header=[]
    row_value=[]
    if exists:
        row_header_2=[None] * len(headers)
    
    for i in range(len(root[0])):
        if root[0][i].text is not None:
            if not root[0][i].text.strip():
                for j in range(len(root[0][i])):
                    if root[0][i][j].text is not None:
                        if not root[0][i][j].text.strip():
                            for k in range(len(root[0][i][j])):
                                #print(str(root[0][i][j][k].tag)+ " : "+str(root[0][i][j][k].text))
                                if not exists:
                                    row_value.append(str(root[0][i][j][k].text))
                                    row_header.append(str(root[0][i][j].tag)+"_"+str(root[0][i][j][k].tag))
                                else:
                                    s_match=str(root[0][i][j].tag)+"_"+str(root[0][i][j][k].tag)
                                    ind=0
                                    try:
                                        ind=headers.index(s_match)
                                    except ValueError:
                                        ind=-1
                                    if ind!=-1:
                                        row_header_2[ind]=str(root[0][i][j][k].text)
                        else:
                            #print(str(root[0][i][j].tag)+ " : "+str(root[0][i][j].text))
                            if not exists:
                                row_value.append(str(root[0][i][j].text))
                                row_header.append(str(root[0][i].tag)+"_"+str(root[0][i][j].tag))
                            else:
                                s_match=str(root[0][i].tag)+"_"+str(root[0][i][j].tag)
                                ind=0
                                try:
                                    ind=headers.index(s_match)
                                except ValueError:
                                    ind=-1
                                if ind!=-1:
                                    row_header_2[ind]=str(root[0][i][j].text)
            else:
                #print(str(root[0][i].tag)+ " : "+str(root[0][i].text))
                if not exists:
                    row_value.append(str(root[0][i].text))
                    row_header.append(str(root[0][i].tag))
                else:
                    s_match=str(root[0][i].tag)
                    ind=0
                    try:
                        ind=headers.index(s_match)
                    except ValueError:
                        ind=-1
                    if ind!=-1:
                        row_header_2[ind]=str(root[0][i].text)
                
    if not exists:
        with open(file_name, 'w', newline='') as f: 
            write = csv.writer(f)
            write.writerow(row_header)
     
    with open(file_name, 'a', newline='') as f: 
        write = csv.writer(f)
        if not exists:
            write.writerow(row_value)
        else:
            write.writerow(row_header_2)
        #print("writing complete for file: "+xmlfile)
        
    if len(headers)==0:
        return row_header
    else:
        return None
    
    
def create_dataset():
    rootdir = 'datasets/'
    file_name="dataset.csv"
    flag=0
    headers=[]
    exists = os.path.exists(file_name)
    if exists:
        os.remove(file_name)
    list=sorted(os.listdir(rootdir),reverse=True)
    for file in list:
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            #print(d)
            file_list = os.listdir(d)
            for file in file_list:
                try:
                    if flag==0:
                        headers=parseXML(d+"/"+file)
                        flag=1
                    else:
                        parseXML(d+"/"+file,headers)
                except:
                    continue

#file_download()
#extract_zip()
create_dataset()

