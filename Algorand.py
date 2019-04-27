import heapq
import random
import math
import ecdsa
import numpy as np
import hashlib
from scipy.stats import binom

seqID = 0

class GlobalState:
    def __init__(self):
        self.seqID = 0
        self.time = 0
        self.numNodes = 5
        self.blockDelay = []
        self.nonBlockDelay = []
        self.pubKeyDB = {}
        self.genesisMsg = "We are building the best Algorand Discrete Event Simulator"
        self.totalStake = 0.0
        self.t_proposer = 5
        self.t_step = 5
        self.t_final = 5
        self.lambda_proposal = 3

    def setBlockDelay(self, mean, std):
        s = np.random.normal(mean, std, gs.numNodes * gs.numNodes)
        index=0
        for i in range(gs.numNodes):
            tmpList = [0.0] * gs.numNodes
            for j in range(i+1):
                if(i==j):
                    tmpList[i] = 0
                else:
                    tmpList[j] = round(max(0, s[index])/1000, 3) # in msec, so dividing by 1000 to make it in sec
                    index+=1
            self.blockDelay.append(tmpList)
        for i in range(gs.numNodes):
            for j in range(i+1):
                self.blockDelay[j][i] = self.blockDelay[i][j]

        
    def setNonBlockDelay(self, mean, std):
        s = np.random.normal(mean, std, gs.numNodes * gs.numNodes)
        index=0
        for i in range(gs.numNodes):
            tmpList = [0.0] * gs.numNodes
            for j in range(i+1):
                if(i==j):
                    tmpList[i] = 0
                else:
                    tmpList[j] = round(max(0, s[index])/1000, 3) # in msec, so dividing by 1000 to make it in sec
                    index+=1
            self.nonBlockDelay.append(tmpList)
        for i in range(gs.numNodes):
            for j in range(i+1):
                self.nonBlockDelay[j][i] = self.nonBlockDelay[i][j]
    
    def assignInitialStake(self, nodes):
        tmpStake = np.random.uniform(1, 50.1, len(nodes))
        index=0
        for node in nodes:
            #print(tmpStake[index])
            node.stake = math.floor(tmpStake[index])
            index +=1
            gs.totalStake +=node.stake
        self.stake = math.floor(random.uniform(1, 50.1))
        #print("stake = ",self.stake)
        return

    def storePublicKeys(self, nodes):
        for node in nodes:
            self.pubKeyDB[node.id] = node.pubKey

    def validateSignature(self, event):
        return True


    def incrementSeqID(self):
        self.seqID = self.seqID + 1
    
    def addTime(self, val):
        self.time = self.time + val


class Node:
    
    def __init__(self, identity):
        self.id = identity
        self.prKey = None
        self.pubKey = None
        self.stake = 0.0
        self.bin_result = []
        self.state = "init"
        return

    def setNeighbors(self, nbr):
        self.neighbors = nbr

    def generateKeyPair(self):
        self.prKey = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1) #Bitcoin eleptic curve=ecdsa.SECP256k1
        self.pubKey = self.prKey.get_verifying_key()
        #signature = self.prKey.sign(b"message")
        #try:
        #    node.pubKey.verify(signature, b"message")
        #    print("good signature")
        #except BadSignatureError:
        #    print("BAD SIGNATURE")
        #open("private.pem","wb").write(sk.to_pem())
        #open("public.pem","wb").write(vk.to_pem())

    

    def processEvent(self):
        return

    def send(self, event):
        print("Node "+str(event.getSrc()) + " send to Node " + str(event.getDest()) + " at time = " + str(event.getTimeStart()))
        if(event.msg.checkIfNodeVisited(self.id) == True):
            print("Silently discarding")
            return # silently discarding
        event.msg.addNodeToVisited(self.id)
        newEvent = Event()
        newEvent.timeStart = event.getTimeEnd()
        newEvent.src = event.getSrc()
        newEvent.timeEnd = newEvent.timeStart
        newEvent.dest = event.getDest()
        newEvent.action = "recv"
        newEvent.msg = event.msg
        #curNode = newEvent.getSrc()
        #for nbr in node[curNode].neighbors:
        #print("Node = "+str(curNode)+" , neighbor = "+str(nbr))
        heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
        gs.incrementSeqID()
        return
    
    def recv(self, event):
        print("Node "+str(event.getDest()) + " recvd from Node " + str(event.getSrc()) + " at time = " + str(event.getTimeStart()))
        if (gs.validateSignature(event) == False):
            print("Invalid signature, abort\n")
            return
        curNode = event.getDest()
        #print(curNode, self.id)
        if(node[curNode].state == "waiting_for_proposal"):
            #add_to_input_buffer
            print()
        else:
            try:
                for nbr in node[curNode].neighbors:
                    newEvent = Event()
                    newEvent.timeStart = event.getTimeEnd()
                    newEvent.src = event.getDest()
                    newEvent.timeEnd = round(newEvent.timeStart + gs.blockDelay[curNode][nbr], 3)
                    newEvent.dest = nbr
                    newEvent.msg = event.msg
                    newEvent.action = "send"
                    heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
                    gs.incrementSeqID()
            except:
                print("EXCEPTION")
            return

    def PRG(self, seed, role):
        #Solution 1
        #print(random.getrandbits(256))
        #Solution 2
        #value=np.random.randint(0,256)
        #local_message="abcdads"
        #msg=local_message+str(local_message)
        m=hashlib.sha512((str(seed)+role).encode('utf-8'))
        digest=int(m.hexdigest(),16)& ((1<<256)-1)#256 bit pseudo random
        return digest # returns 256 b integer

    def nCr(self, n, r):
        return math.factorial(n)/(math.factorial(r) * math.factorial(n-r))

    def binomial_sum(self, j, w, p):
        #print("CHECK IF THIS IS WORKING CORRECTLY")
        k=0
        binoSum = 0.0
        while(k<j and j<=w):
            binoSum += self.nCr(w, k) * pow(p, k) * pow((1-p), (w-k))
            k += 1
        #print(j,w,p, binoSum)
        return binoSum

        '''if j == 0:
            self.bin_result.append(binom.rvs(size=0,n=n,p=p))
            return self.bin_result[0]
        if j < len(self.bin_result):
            return self.bin_result[j]
        self.bin_result.append(self.bin_result[-1]+binom.rvs(size=j,n=n,p=p) )
        return self.bin_result[-1]'''

    def calc_hashlen(self, my_hash):
        bits = 0
        while my_hash != 0:
            my_hash = my_hash >> 1
            bits += 1
        return bits
        #return (len(my_hash))

    def sortition(self, secret_key, seed, threshold, role, w, W):
        pi = self.PRG(str(seed), role)  # seed = <hash of prev rnd || rnd num || step num >
        #print(pi)
        signature = secret_key.sign(str(pi).encode('utf-8')) # my_hash has the signature
        my_hash = signature.hex()
        p = threshold/W
        j = 0
        my_hash = int(my_hash, 16)
        hashlen = self.calc_hashlen(my_hash)
        hash_2hashLen = my_hash/pow(2,hashlen)
        l_limit = self.binomial_sum(j,w,p)
        u_limit = self.binomial_sum(j+1,w,p)
        #print(hash_2hashLen, l_limit, u_limit, j)
        while (hash_2hashLen<l_limit or hash_2hashLen>=u_limit) and j<=w:
            j += 1
            l_limit = self.binomial_sum(j,w,p)
            u_limit = self.binomial_sum(j+1,w,p)
            #print(l_limit, u_limit, j)
        return my_hash,pi,j

    def computePriorityForSubUser(self, my_hash, subUserIndex):
        hashObj = hashlib.sha256()
        hashObj.update((str(my_hash)+str(subUserIndex)).encode("utf-8"))
        return hashObj.hexdigest()

    def waitForProposals(self, gs):
        self.state = "waiting_for_proposal"
        #add event to wait for gs.lambda_proposal time
        return

    def gossip(self, gs, my_hash, pi, subUser, priority):
        payload = str(my_hash)+str(subUser)+str(priority) # also concatenate round no. here
        for nbr in self.neighbors:
            msg = Message(payload)
            print(self.id, "Gossiping to ", nbr)
            event = Event()
            event.msg = msg
            event.src = self.id
            event.dest = nbr
            event.action = "send"
            event.timeStart = gs.time
            event.timeEnd = event.timeStart + gs.nonBlockDelay[self.id][nbr]
            heapq.heappush(eventQ, (gs.time, gs.seqID, event))
            gs.incrementSeqID()
        #self.send(payload) # send payload to neighbors. How many neighbors ?
        self.waitForProposals(gs)
        
    def proposeBlock(self):
        #broadcast block ot be proposed, delay = node.blockDelay
        return


class Event:
    def __init__(self, start=0, delay=0, src=0, dest=0, action="none", msg="none"):
        self.timeStart = start
        self.src = src
        self.dest = dest
        self.timeEnd = self.timeStart + delay
        self.action = action
        self.msg = msg

    def getAction(self):
        return self.action
    def getSrc(self):
        return self.src
    def getDest(self):
        return self.dest
    def getTimeStart(self):
        return self.timeStart
    
    def getTimeEnd(self):
        return self.timeEnd

class Message:
    def __init__(self, msg):
        self.msg = msg
        self.nodesVisited=set()
    
    def addNodeToVisited(self, node):
        self.nodesVisited.add(node)

    def checkIfNodeVisited(self, node):
        if node in self.nodesVisited:
            return True
        return False        

class Block:
    def __init__(self, msg):
        return


def start(gs, nodes):
    #print(len(nodes))
    for node in nodes:
        print(node.id, node.stake)
        threshold = gs.t_proposer
        role = "proposer"
        w = node.stake
        W = gs.totalStake
        hashVal, pi, subUser = node.sortition(node.prKey, "random_seed", threshold, role, w, W)
        if(subUser>0): # node selected as block proposer
            highestPriority = -1
            subUserWithHighestPriority = -1
            for j in range(0, subUser):
                priority = node.computePriorityForSubUser(hashVal, j)
                priority = int(priority, 16)
                if(highestPriority<0 or highestPriority>priority):
                    highestPriority = priority
                    subUserWithHighestPriority = j
            node.gossip(gs, hashVal, pi, subUserWithHighestPriority, highestPriority)                
        print("Node = ", node.id, "stake = ", node.stake, "subusers = ", subUser)
    print("SORTITION PART DONE, NOW BLOCK PROPOSAL PART")
    #SORTITION PART DONE, NOW BLOCK PROPOSAL PART
    while True:
        try:
            ele = heapq.heappop(eventQ)
            curTime = round(ele[0], 3)
            gs.time = curTime
            print("current time = " + str(gs.time))
            curEvent = ele[2]
            if curEvent.getAction() == "send":
                nodes[curEvent.src].send(curEvent)
            elif curEvent.getAction() == "recv":
                nodes[curEvent.dest].recv(curEvent)
            elif curEvent.getAction() == "proposeBlock":
                nodes[curEvent.src].proposeBlock(curEvent)
        except IndexError as e:
            print("No events remaining. ", e)
            break
    return



if __name__ == "__main__":
    gs = GlobalState()
    node = []
    eventQ = []
    gs.time = 0
    gs.setBlockDelay(200, 400)
    gs.setNonBlockDelay(30, 64)

    for i in range(gs.numNodes):
        node.append(Node(i))
        numNeighbors = math.floor(random.uniform(2, 4.1))
        nbrs = []
        while(numNeighbors>0):
            newNbr = random.randint(1, gs.numNodes) - 1 
            if(newNbr in nbrs or newNbr == i):
                continue
            nbrs.append(newNbr)
            numNeighbors = numNeighbors - 1
        node[i].generateKeyPair()
        node[i].setNeighbors(nbrs)
        #print(i, ":", node[i].neighbors)
    gs.assignInitialStake(node)
    gs.storePublicKeys(node)
    msg = Message("a")
    event = Event(gs.time, gs.blockDelay[0][1], 0, 1, "send", msg)

    #heapq.heappush(eventQ, (gs.time, gs.seqID, event))
    #gs.incrementSeqID()
    #heapq.heappush(eventQ, (gs.time, gs.seqID, Event(gs.time, gs.blockDelay[0][2], 0, 2, "send", "a")))
    #gs.incrementSeqID()
    start(gs, node)