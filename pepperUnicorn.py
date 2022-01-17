import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import argparse



#***************************
# chargement data
#***************************
parser = argparse.ArgumentParser(description='plot embeddings  .')
parser.add_argument("--dev", default='', type=str, help="dev dataset")
parser.add_argument("--train", default='', type=str, help="train dataset")
parser.add_argument("--test", default='', type=str, help="test dataset")

parser.add_argument("--m", choices=["train","test"],required=True, type=str, help="test dataset")

parser.add_argument("--xvectors", default='', type=str, help="xvectors dict")

parser.add_argument("--saved_model", default='', type=str, help="test dataset")

parser.add_argument("--num_epochs", default='10', type=int, help="number of epochs")
parser.add_argument("--hidden_size", default='512', type=int, help="size of hidden layer")
parser.add_argument("--saved_model_path", default='', type=str, help=" path to model folder ")

args = parser.parse_args()

#******************************
# LOADING dataset
#******************************

def loadXvectorsDict(pathDico):

    file = open(pathDico, "r")
    lines = file.readlines()
    file.close()

    xvectors = {}
    for line in tqdm(lines):
        line = line.rstrip()
        idVector = line.split(' ')[0]

        xvector = line.split(' [ ')[1].split(' ]')[0].split(' ')
        xvector = [float(s) for s in xvector]
        xvector = torch.as_tensor(xvector)
        
        xvectors[idVector] = xvector

    # end for

    return xvectors

# end def


def loadData(dataset,xvectors):

    file = open(dataset, "r")
    lines = file.readlines()
    file.close()

    dataset = []
    nbSame = 0
    nbPasSame=0
    for line in tqdm(lines):
    
        data = [] # batch de 1

        line = line.rstrip()
        
        utt1 = line.split(' ')[0]
        utt2 = line.split(' ')[1]
    
        xvector1 = xvectors[utt1]
        xvector2 = xvectors[utt2]

        spk1 = utt1.split('-')[0].rstrip()
        spk2 = utt2.split('-')[0].rstrip()
        

        if(spk1 == spk2):
            res = 1.0 # 1 == same speaker
        else:
            res = 0.0 # 0 == not the same

        nbSame += res
        nbPasSame += 1.0-res

        data.append(xvector1)
        data.append(xvector2)
        data.append(res)
    
        dataset.append(data)        
    # end for
    return dataset
# end for

#******************************
# topologie network
#******************************

class Net(nn.Module):
    def __init__(self,inSize,outSize,hiddenSize):
        super(Net, self).__init__()
        # une couche cachee
        self.fc1 = nn.Linear(inSize,hiddenSize)
        self.fc2 = nn.Linear(hiddenSize,128)
        self.fc3 = nn.Linear(128,outSize)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sig(x)
        return x

#**********************
# check si GPU dispo
#**********************
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # fonctionnement mono-gpu pour l'instant
else:
    device = torch.device("cpu")


# NETWORK INSTANCE
input_size = 512 # taille des deux embeds concat
output_size = 1 # une unité
net = Net(input_size,output_size,args.hidden_size).to(device)

# iterate on all batch of an epoch
def makeTrainStep(network,dataSet):

    running_loss = 0.0
    network.train()
    threshold = torch.tensor([0.5]).to(device)

    for batch in tqdm(dataSet):
        
        inputs = torch.cat((batch[0],batch[1]),0)
        labels = torch.as_tensor([float(batch[2])])
        

        # pour gestion device gpu / cpu
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = network(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
         
        optimizer.step()

        running_loss += loss.item() 
    # end for epoch
    
    return running_loss
# end def 

def makeEval(network,dataset):

    threshold = torch.tensor([0.5]).to(device)
    network.eval()

    nbGood = 0

    for batch in tqdm(dataset):
        inputs = torch.cat((batch[0],batch[1]),0)
        labels = torch.as_tensor([float(batch[2])])

        # pour gestion device gpu / cpu
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = network(inputs)
        results = (outputs>threshold).float()*1

        result = results.cpu().data.numpy()[0]
        label = labels.cpu().data.numpy()[0]

        if(result == label):
            nbGood +=1

    # end for

    acc = float(nbGood)/float(len(dataset))
    return acc, outputs.item(), results.item() 

#*************
# TRAIN
#*************

 
if(args.m == "train"):
    # *****************************************
    # loss function + optimizer + LR scheduler
    # *****************************************
    learning_rate=0.0001
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    # *****************************************

    print("LOAD XVECTORS")
    xvectors = loadXvectorsDict(args.xvectors)

    print("LOAD TRAIN")
    data_train = loadData(args.train,xvectors)

    print("LOAD DEV")
    data_dev   = loadData(args.dev,xvectors)

    best_acc = 0.0
    tab_dev_loss = []
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        print("train epoch : " + str(epoch+1))    
        # **************************
        # TRAINING OVER BATCHES
        # **************************
        epoch_loss = makeTrainStep(net,data_train)
        print("epoch : " + str(epoch+1) + " finished - loss = "+str(epoch_loss))
        # **************************

        # ***************
        # VALIDATION
        # ***************
        print("\nVALIDATION : " + str(epoch+1))
        dev_acc,_,_ = makeEval(net,data_dev)
        print("DEV accuracy :", dev_acc)
        # ******************
        # LEARNING RATE STEP
        # ******************
        lr = optimizer.param_groups[0]["lr"]
        print("learning rates : ", lr)
        scheduler.step() 
        # *****************

        # *****************
        # SAVE BEST MODEL
        # *****************
        if(dev_acc > best_acc):
            print('find better model (dev acc)')
            path= args.saved_model_path+"_"+str(epoch+1)+".pth"
            print("save as : " + path)
            torch.save(net.state_dict(), path)
            best_acc = dev_acc
        # *****************

    # end for epoch
    print('Finished Training')

elif(args.m == "test"):

    # ***************
    # TEST ONLY
    # ***************

    # ******************
    # MODEL LOADING
    # *****************
    net = Net(input_size,output_size,args.hidden_size).to(device)
    net.load_state_dict(torch.load(args.saved_model,map_location='cpu'))

    xvectors = loadXvectorsDict(args.xvectors)

    data_test = loadData(args.test,xvectors)

    test_acc, confidence, decision = makeEval(net,data_test)

    print(confidence)
    print(decision)

