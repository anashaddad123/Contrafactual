
import subprocess
import os
wavescp = 'wav.scp'
xvectors = 'xvectors.txt'
list_decision = []
dict_id = {}
with open(wavescp) as f:
    lines = f.readlines()
f.close()
for line in lines :
    matchers ='id'
    spk_id = line.split(" ")[0]
    liste_path = line.split(" ")[1].split('/')
    matching = [s for s in liste_path if matchers in s ]
    dict_id[spk_id] = matching[0]


with open(xvectors) as f:
    lines = f.readlines()
for i in range(0,(len(lines)-1)):
    fichier_id = open("id_filedemo.txt", "w+")
    if i+1 != (len(lines)-1):
     for j in range(i+1,(len(lines)-1)):
       ide = lines[i].split(" ")[0]
       ide = ide + str(' ')
       ide1 = lines[j].split(" ")[0]
       ide3 = ide + ide1
       with open('id_filedemo.txt', 'w') as f:
             f.write(ide3)
             f.close()
       with open('xvectors_demo.txt', 'w') as f:
             f.write(lines[i]+lines[-1])
             f.close()    
       result = subprocess.check_output('python3 ~/testtp/pepperUnicorn/pepperUnicorn.py --m test --test ~/testtp/id_filedemo.txt --xvectors ~/testtp/xvectors_demo.txt --saved_model ~/testtp/pepperUnicorn/model_100000_95.pth',shell=True)
       result = result.decode("utf-8") 
       score = result.split('\n')[0]
       decision = result.split('\n')[1]
       if dict_id[ide.strip()] == dict_id[ide1.strip()]:
            list_decision.append([ide,ide1,score,decision,1])
       else :
            list_decision.append([ide,ide1,score,decision,-1])
    else:

        ide = lines[i].split(" ")[0]
        ide = ide + str(' ')
        ide1 = lines[-1].split(" ")[0]
        ide3 = ide + ide1
        fichier_id.write(ide)
        with open('id_filedemo.txt', 'w') as f:
             f.write(ide3)
             f.close()
        with open('xvectors_demo.txt', 'w') as f:
             f.write(lines[i]+lines[-1])
             f.close()    
        result = subprocess.check_output('python3 ~/testtp/pepperUnicorn/pepperUnicorn.py --m test --test ~/testtp/id_filedemo.txt --xvectors ~/testtp/xvectors_demo.txt --saved_model ~/testtp/pepperUnicorn/model_100000_95.pth',shell=True)
        result = result.decode("utf-8") 
        score = result.split('\n')[0]
        decision = result.split('\n')[1]
        if dict_id[ide.strip()] == dict_id[ide1.strip()]:
            list_decision.append([ide,ide1,score,decision,1])
        else :
            list_decision.append([ide,ide1,score,decision,-1])
print(list_decision)




