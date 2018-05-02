import os 

f = open("/data-sdc/hao01.wang/IJBA/IJBA.txt", "w")
g = os.walk("/data-sdc/hao01.wang/IJBA/dataset/")

for path,d,filelist in g:
   for filename in filelist:
      f.write(os.path.join(path, filename)+'\n')

f.close()
