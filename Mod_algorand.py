import heapq
import random
import math
import ecdsa
import numpy as np
import hashlib
from scipy.stats import binom
import sys
from threading import Lock, Thread
import time

seqID = 0

class GlobalState:
    def __init__(self):
        self.globalState = "init"
        self.seqID = 0
        self.time = 0
        self.numNodes = int(sys.argv[1])
        self.blockDelay = []
        self.nonBlockDelay = []
        self.pubKeyDB = {}
        self.genesisMsg = "We are building the best Algorand Discrete Event Simulator"
        self.totalStake = 0.0
        self.t_proposer = 5
        self.t_step = 5
        self.t_final = 5
        self.lambda_proposal = 3
        self.lambda_block = 30
        self.lambda_step = 3
        # adding blockchain
        self.blockchain = []
        self.blockchain.append(Block(self.genesisMsg, 0))
        self.lastBlockIndex = 0
        self.pubKey_list = []
        self.roundNum = 0
        self.T = 2/3#Changed to 1/5 from 2/3 Kami
        self.lock = Lock()
        self.blockInserted = False
        self.failStopParameter = 10
        self.byzParameter = 10
        self.algoStartTime = 0
        self.algoEndTime = 0

    def terminate(self):
        exit(0)

    def startBAstar(self, nodes):
        for node in nodes:
            if (node.id % self.failStopParameter == 0 and sys.argv[3] == "fs"):
                #print("id = ", node.id, "I am adversary, won't participate in consensus")
                continue
            node.BAstar(self)

    def setBlockDelay(self, mean, std):
        s = np.random.normal(mean, std, gs.numNodes * gs.numNodes)
        index=0
        for i in range(gs.numNodes):
            tmpList = [0.0] * gs.numNodes
            for j in range(i+1):
                if(i==j or (i%self.byzParameter==0 and j%self.byzParameter==0)):
                    tmpList[i] = 0
                else:
                    tmpList[j] = round(max(0, s[index])/1000, 3) # in msec, so dividing by 1000 to make it in sec
                    index+=1
            self.blockDelay.append(tmpList)
        for i in range(gs.numNodes):
            for j in range(i+1):
                self.blockDelay[j][i] = self.blockDelay[i][j]

    def cleanup(self, nodes, firstTime=False):
        if firstTime == False:
            return
        for node in nodes:
            node.recv_buffer.clear()
        return

    def setNonBlockDelay(self, mean, std):
        s = np.random.normal(mean, std, gs.numNodes * gs.numNodes)
        index=0
        for i in range(gs.numNodes):
            tmpList = [0.0] * gs.numNodes
            for j in range(i+1):
                if(i==j or (i%self.byzParameter==0 and j%self.byzParameter==0)):
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
            node.stake = math.floor(tmpStake[index])
            index +=1
            gs.totalStake +=node.stake
        self.stake = math.floor(random.uniform(1, 50.1))
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
        self.recv_buffer = [] # to recieve incoming messages, can be anything, including a block
        self.state = "" # to simulate various states
        self.step = 0
        self.priority_payload = []
        self.vote_dictionary = dict()
        self.maximum_key = -1
        self.MAXSTEPS = 10
        self.new_max_key = None
        self.byz = False
        self.flag = -1
        return

    def setNeighbors(self, nbr):
        self.neighbors = nbr

    def generateKeyPair(self,gs):
        self.prKey = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1) #Bitcoin eleptic curve=ecdsa.SECP256k1
        self.pubKey = self.prKey.get_verifying_key()
        gs.pubKey_list.append(self.pubKey)
        #signature = self.prKey.sign(b"message")
        #try:
        #    node.pubKey.verify(signature, b"message")
        #    #print("\ngood signature")
        #except BadSignatureError:
        #    #print("\nBAD SIGNATURE")
        #open("private.pem","wb").write(sk.to_pem())
        #open("public.pem","wb").write(vk.to_pem())

    

    def processEvent(self):
        return

    def send(self, gs, event):
        if(event.msg.checkIfNodeVisited(self.id) == True):
            ##print("Silently discarding")
            return # silently discarding
        if (gs.time > event.timeout):
            # stale message, discarding
            return
        ##print("Node "+str(event.getSrc()) + " send to Node " + str(event.getDest()) + " at time = " + str(event.getTimeStart()))
        event.msg.addNodeToVisited(self.id)
        newEvent = Event()
        newEvent.timeStart = event.getTimeEnd()
        newEvent.src = event.getSrc()
        newEvent.timeEnd = newEvent.timeStart
        newEvent.dest = event.getDest()
        #if(self.state == "gossiping_vote"):
        #    newEvent.action = "relay"
        #else:
        newEvent.timeout = event.timeout
        newEvent.action = "recv"
        newEvent.msg = event.msg
        heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
        gs.incrementSeqID()
        return



    def recv(self, gs, event):
        if (gs.validateSignature(event) == False):
            #print("Invalid signature, abort\n")
            return
        if(gs.time > event.timeout):
            #stale message, discarding
            return
        ##print("Node "+str(event.getDest()) + " recvd from Node " + str(event.getSrc()) + " at time = " + str(event.getTimeStart()))
        curNode = event.getDest()
        #if(self.state == "wait_for_proposal"):
        self.recv_buffer.append(event)
        try:
            for nbr in node[curNode].neighbors:
                newEvent = Event()
                newEvent.timeStart = event.getTimeEnd()
                newEvent.src = event.getDest()
                newEvent.timeEnd = round(newEvent.timeStart + gs.blockDelay[curNode][nbr], 3)
                newEvent.dest = nbr
                newEvent.timeout = event.timeout
                newEvent.msg = event.msg
                newEvent.action = "send"
                heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
                gs.incrementSeqID()
        except:
            print("EXCEPTION")
        return

    
    def PRG(self, seed):
    #def
        #Solution 1
        #Solution 2
        #value=np.random.randint(0,256)
        #local_message="abcdads"
        #msg=local_message+str(local_message)
        m=hashlib.sha256((str(seed)).encode('utf-8'))
        digest=int(m.hexdigest(),16)& ((1<<256)-1)#256 bit pseudo random
        return digest # returns 256 b integer

    def nCr(self, n, r):
        return math.factorial(n)/(math.factorial(r) * math.factorial(n-r))

    def binomial_sum(self, j, w, p):
        k=0
        binoSum = 0.0
        while(k<j and j<=w):
            binoSum += self.nCr(w, k) * pow(p, k) * pow((1-p), (w-k))
            k += 1
        ##print(j,w,p, binoSum)
        return binoSum

    def calc_hashlen(self, my_hash):
        bits = 0
        while my_hash != 0:
            my_hash = my_hash >> 1
            bits += 1
        return bits
        #return (len(my_hash))

    def sortition(self, secret_key, seed, threshold, w, W):
        pi = (str(self.PRG(str(seed)))).encode('utf-8')  # seed = <hash of prev rnd || rnd num || step num >
        signature = secret_key.sign(pi) # my_hash has the signature
        my_hash = signature.hex()
        p = threshold/W
        j = 0
        my_hash = int(my_hash, 16)
        hashlen = self.calc_hashlen(my_hash)
        hash_2hashLen = my_hash/pow(2,hashlen)
        l_limit = self.binomial_sum(j,w,p)
        u_limit = self.binomial_sum(j+1,w,p)
        while (hash_2hashLen<l_limit or hash_2hashLen>=u_limit) and j<=w:
            j += 1
            l_limit = self.binomial_sum(j,w,p)
            u_limit = self.binomial_sum(j+1,w,p)
            #print(l_limit, u_limit, j)
        my_hash=signature.hex()
        return my_hash,pi,j

    def computePriorityForSubUser(self, my_hash, subUserIndex):
        hashObj = hashlib.sha256()
        hashObj.update((str(my_hash)+str(subUserIndex)).encode("utf-8"))
        return hashObj.hexdigest()

    def waitForPriorityProposals(self, gs):
        #add event to wait for gs.lambda_proposal time
        self.state = "wait_for_proposal"

        newEvent = Event()
        newEvent.timeStart = gs.time + gs.lambda_proposal
        newEvent.src = self.id
        newEvent.timeEnd = newEvent.timeStart
        newEvent.dest = self.id
        newEvent.action = "processRecvBuffer"
        newEvent.msg = Message("start processing recv_buffer")

        heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
        gs.incrementSeqID()
        return

    def gossip(self, gs, payload_list, typeOfGossip=None):
        payload = ""
        if typeOfGossip == "priority":   #len(payload_list) == 5: # going to send priorities
            roundno = payload_list[0]
            my_hash = payload_list[1]
            subUser = payload_list[2]
            priority = payload_list[3]
            id_pubKey = payload_list[4]
            payload = str(roundno)+"-"+str(my_hash)+"-"+str(subUser)+"-"+str(priority)+"-"+str(id_pubKey) # also concatenate round no. here
            # "-" will help to split the parameters easily
        #self.send(payload) # send payload to neighbors. How many neighbors ?
            msg = Message(payload)
            # event = Event(gs.time, gs.nonBlockDelay[self.id][self.id], gs.time+gs.lambda_proposal, self.id, self.id, "send", msg)
            event = Event(gs.time, gs.nonBlockDelay[self.id][self.id], self.id, self.id, "send", msg)
            self.recv_buffer.append(event)
            self.waitForPriorityProposals(gs)
            for neighbor in self.neighbors:
                msg = Message(payload)
                # event = Event(gs.time, gs.nonBlockDelay[self.id][neighbor], gs.time+gs.lambda_proposal, self.id, neighbor, "send", msg)
                event = Event(gs.time, gs.nonBlockDelay[self.id][neighbor], self.id, neighbor, "send", msg)
                heapq.heappush(eventQ, (event.getTimeStart(), gs.seqID, event))
                gs.incrementSeqID()
        elif typeOfGossip == "voting":
            prevBlockHash = payload_list[0]
            curBlockHash = payload_list[1]
            roundno = payload_list[2]
            step = payload_list[3]
            subUser = payload_list[4]
            vrf_output = payload_list[5]
            payload = str(prevBlockHash)+"-"+str(curBlockHash)+"-"+str(roundno)+"-"+str(step)+"-"+str(subUser)+"-"+str(vrf_output)
            self.state = "gossiping_vote"
            for neighbor in self.neighbors:
                msg = Message(payload)
                event = Event(gs.time, gs.nonBlockDelay[self.id][neighbor], self.id, neighbor, "send", msg)
                heapq.heappush(eventQ, (event.getTimeStart(), gs.seqID, event))
                gs.incrementSeqID()

        elif  typeOfGossip == "blockProposal": #len(payload_list) == 7: 
            # block proposal
            prevBlockHash = payload_list[0]
            rand256 = payload_list[1]
            roundno = payload_list[2]
            my_hash = payload_list[3]
            subUser = payload_list[4]
            priority = payload_list[5]
            id_pubKey = payload_list[6]
            payload = str(prevBlockHash)+"-"+str(rand256)+"-"+str(roundno)+"-"+str(my_hash)+"-"+str(subUser)+"-"+str(priority)+"-"+str(id_pubKey)
            # self.waitForBlockPriorityProposals(gs)
            self.state = "wait_for_proposal"
            for neighbor in self.neighbors:
                if(sys.argv[3]=="byz" and self.byz==True and neighbor%gs.byzParameter==0):
                    continue
                msg = Message(payload)
                event = Event(gs.time, gs.nonBlockDelay[self.id][neighbor], self.id, neighbor, "send", msg)
                heapq.heappush(eventQ, (event.getTimeStart(), gs.seqID, event))
                gs.incrementSeqID()
            if(sys.argv[3]=="byz" and self.byz==True and neighbor%gs.byzParameter==0):
                idx=0
                while(idx<=gs.numNodes):
                    if(self.id == idx):
                        continue
                    msg = Message(payload)
                    event = Event(gs.time, gs.nonBlockDelay[self.id][idx], self.id, idx, "send", msg)
                    heapq.heappush(eventQ, (event.getTimeStart(), gs.seqID, event))
                    gs.incrementSeqID()
                    idx +=gs.byzParameter

    def processRecvBuffer(self):
        highestPriority = -1.0
        subUserWithHighestPriority = -1
        userWithHighestPriority = -1
        for i in range(len(self.recv_buffer)):
            msg = self.recv_buffer[i].getMessage().split("-")
            if len(msg) == 5:
                priority = float(msg[-2]) # msg = str(roundno)+"-"+str(my_hash)+"-"+str(subUser)+"-"+str(priority)+str(id)
                if(priority > highestPriority):
                    highestPriority = priority
                    subUserWithHighestPriority = int(msg[2])
                    userWithHighestPriority = int(msg[-1])
        cleanupEvent = Event()
        cleanupEvent.timeStart = gs.time
        cleanupEvent.timeEnd = cleanupEvent.timeStart
        cleanupEvent.action = "cleanup"
        heapq.heappush(eventQ, (cleanupEvent.getTimeStart(), gs.seqID, cleanupEvent))
        gs.incrementSeqID()
        #print("According to Node ",self.id," ",userWithHighestPriority, " is the highest priority user.")
        if(userWithHighestPriority == self.id):
            #print("I, Node ",self.id, ", am going to propose the next block, subuser = ", subUserWithHighestPriority)
            newEvent = Event()
            newEvent.timeStart = gs.time
            newEvent.src = self.id
            newEvent.timeEnd = newEvent.timeStart
            newEvent.dest = self.id
            newEvent.action = "proposeBlock"
            newEvent.msg = Message("proposeBlock")
            heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
            gs.incrementSeqID()
        return

    def proposeBlock(self, gs):
        if (self.id % gs.failStopParameter == 0 and sys.argv[3] == "fs"):
            print("id = ", self.id, ", I am fs adversary, won't propose block")
            return
        gs.globalState = "blockProposal"
        #broadcast block to be proposed, delay = node.blockDelay
        #print("\nNode ",self.id, " reporting for proposing a new block")
        message = "Block created by Node "+str(self.id)+" at round "+str(len(gs.blockchain)-1)
        # using PRG code here. Can make it modular later.
        prevBlockHash = hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest()
        block = Block(message, prevBlockHash)
        payload_list = []
        payload_list.append(prevBlockHash)
        payload_list.append(random.getrandbits(256))
        for item in self.priority_payload:
            payload_list.append(item)
        self.gossip(gs, payload_list, "blockProposal")
        if(self.byz==True and sys.argv[3] == "byz"):
            print("id = ", self.id, ", I am byzantine, proposing extra block")
            message = "Another block created by Node "+str(self.id)+" at round "+str(len(gs.blockChain)-1)
            block = Block(message, prevBlockHash)
            payload_list = []
            payload_list.append(prevBlockHash)
            payload_list.append(random.getrandbits(256))
            for item in self.priority_payload:
                payload_list.append(item)
            self.gossip(gs, payload_list, "blockProposal")
        return

    def checkBlockProposal(self,gs):
        # called after committe members timeout
        # now they'll check whether any block proposal was received
        # if not, they'll commit vote for an empty block
        #need public key of sender+hash of prev block+roundnumber+"0"
        #if validate success flag=1 and put data in buffer
        #else do nothing
        #at end if flag=0 then create empty block and vote on it

        # Have made a global list of public keys and can access the same using index
        #print("\n:) len(recv_buffer) = "+str(len(self.recv_buffer)))
        local_buffer = None
        local_priority = 2**257
        for i in range(len(self.recv_buffer)):
            msg = self.recv_buffer[i].getMessage().split("-")
            if len(msg) == 7:
                priority = float(msg[-2])
                pubKey = gs.pubKey_list[int(msg[-1])]
                prevBlockHash = hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest()
                roundno=len(gs.blockchain) - 1
                #print("Round->",roundno)
                seed=prevBlockHash+str(roundno)+str(0)
                try:
                    pubKey.verify(bytes.fromhex(msg[3]),str(self.PRG(seed)).encode('utf-8'))
                    if local_buffer == None or priority<local_priority:
                        local_buffer = []
                        prevBlockHash = hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest() # prevHash
                        sha_256 = msg[1] # sha256
                        curBlockHash = hashlib.sha256((prevBlockHash+sha_256).encode('utf-8')).hexdigest()
                        roundno = roundno
                        step = self.step
                        subUserIndex = msg[4]
                        vrf_output = msg[3]

                        local_buffer.append(prevBlockHash)
                        local_buffer.append(curBlockHash)
                        local_buffer.append(roundno)
                        local_buffer.append(step)
                        local_buffer.append(subUserIndex)
                        local_buffer.append(vrf_output)

                        local_priority = priority
                except:
                    pass
                    #print("\nValidation failure")
        if local_buffer == None:
            # empty block
            prevBlockHash = hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest()
            sha_256 = "Empty" # sha256
            curBlockHash = hashlib.sha256((prevBlockHash+sha_256).encode('utf-8')).hexdigest()
            roundno = len(gs.blockchain) - 1
            step = self.step
            subUserIndex = "Empty"

            roundno=len(gs.blockchain) - 1
            hash_prev=hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest()
            seed=hash_prev+str(roundno)+str(self.step) # DIPTYAROOP : CHECK IF NODE.STEP IS CORRECT
            vrf_output = self.PRG(seed)
            local_buffer=[]
            local_buffer.append(prevBlockHash)
            local_buffer.append(curBlockHash)
            local_buffer.append(roundno)
            local_buffer.append(step)
            local_buffer.append(subUserIndex)
            local_buffer.append(vrf_output)
        #print("::LOCAL BUFFER::")
        #for i in local_buffer:
            #print(i)
        return local_buffer  

        

    def countVotes(self, gs):

        self.vote_dictionary = dict()
        #print("PRINTING VALUES")
        for i in range(len(self.recv_buffer)):
            value = self.recv_buffer[i].getMessage()
            # prevBlockHash = msg[0]
            # curBlockHash = msg[1]
            # roundno = msg[2]
            # step = msg[3]
            # subUser = msg[4]
            # vrf_output = msg[5]

            # value = str(prevBlockHash) + "-" + str(curBlockHash) + "-" + str(roundno) + "-" + str(step) + "-" + str(subUser) + "-" + str(vrf_output)
            if value in self.vote_dictionary:
                self.vote_dictionary[value] = self.vote_dictionary[value] + 1
            elif value:
                self.vote_dictionary[value] = 1
        if(len(self.recv_buffer)>0):
            maximum_key = max(self.vote_dictionary, key=self.vote_dictionary.get)
            # if self.isCommittee == True:
            if self.vote_dictionary[maximum_key] > (gs.T * gs.t_step):
                # store this maximu_key
                self.maximum_key = maximum_key
            else:
                # timeout
                self.maximum_key = -1
        else:
            self.maximum_key = -1        
        return self.maximum_key

    def committeeElection(self, gs):
        threshold = gs.t_step
        w = self.stake
        W = gs.totalStake
        roundno=len(gs.blockchain) - 1
        hash_prev=hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest() # Keep structure later
        seed=hash_prev+str(roundno)+str(self.step)
        hashVal, pi, j = self.sortition(self.prKey, seed, threshold, w, W)
        self.step+=1
        if j > 0:
            return True
        else:
            return False

    def Reduction(self, gs):
        #print("\nReduction called")
        if self.committeeElection(gs):#First committee election takes place step=1
            #print("I", self.id, ", am a committee member")
            #33sec delay
            newEvent = Event()
            newEvent.timeStart = gs.time + gs.lambda_block + gs.lambda_step
            newEvent.src = self.id
            newEvent.timeEnd = newEvent.timeStart
            newEvent.dest = self.id
            newEvent.action = "reduction1"
            newEvent.msg = Message("reduction1")
            heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
            gs.incrementSeqID()
            # local_buffer=self.checkBlockProposal(gs)#Committee receives all the values and stores the one with minimum priority
            # self.committeeVote(local_buffer)#Committee members vote
        # 3 sec DELAY
        
        

    def reduction1(self, gs):
        local_buffer=self.checkBlockProposal(gs)#Committee receives all the values and stores the one with minimum priority
        #print("\nLocal buffer in reduction1: ",local_buffer)
        self.committeeVote(local_buffer, gs)
        newEvent = Event()
        newEvent.timeStart = gs.time + gs.lambda_step
        newEvent.src = self.id
        newEvent.timeEnd = newEvent.timeStart
        newEvent.dest = self.id
        newEvent.action = "reduction2"
        newEvent.msg = Message("reduction2")
        heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
        gs.incrementSeqID()

    def reduction2(self, gs):
        #print("\nReduction2")
        max_key=self.countVotes(gs)#Call CountVotes get block in temp_block variable
        # #print("MAXIMUM KEY=",max_key)
        if self.committeeElection(gs):#Second committee election takes place step=2
            if(max_key!=-1):
                self.committeeVote(self.maximum_key.split("-"), gs)#if committee member has temp_block==-1 Vote for empty block
            else:#else committee member votes for current block
                local_buffer=self.getEmptyString()
                self.committeeVote(local_buffer.split("-"), gs)
        newEvent = Event()
        newEvent.timeStart = gs.time + gs.lambda_step
        newEvent.src = self.id
        newEvent.timeEnd = newEvent.timeStart
        newEvent.dest = self.id
        newEvent.action = "reduction3"
        newEvent.msg = Message("reduction3")
        heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
        gs.incrementSeqID()

    def reduction3(self, gs):
        max_key=self.countVotes(gs)#Call CountVotes get block in temp_block2 variable
        if(max_key==-1):#if temp_block2==-1 return empty block
            max_key=self.getEmptyString()
        else:#else return block received
            pass
        self.maximum_key = max_key
        newEvent = Event()
        newEvent.timeStart = gs.time
        newEvent.src = self.id
        newEvent.timeEnd = newEvent.timeStart
        newEvent.dest = self.id
        newEvent.action = "BinaryBAstar"
        newEvent.msg = Message("BinaryBAstar")
        heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
        gs.incrementSeqID()

        #self.BinaryBAstar()
        # return max_key

    def getEmptyString(self):
        prevBlockHash = hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest()
        sha_256 = "Empty" # sha256
        curBlockHash = hashlib.sha256((prevBlockHash+sha_256).encode('utf-8')).hexdigest()
        roundno = len(gs.blockchain) - 1
        step = self.step
        subUserIndex = "Empty"
        prevBlockHash = hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest()
        seed=prevBlockHash+str(roundno)+str(0) # DIPTYAROOP : CHECK IF NODE.STEP IS CORRECT
        vrf_output = self.PRG(seed)
        return str(prevBlockHash) + "-" + str(curBlockHash) + "-" + str(roundno) + "-" + str(step) + "-" + str(subUserIndex) + "-" + str(vrf_output)

    def stringtoBuffer(self,string):
        local_buffer=[]
        for i in string.split('-'):
            local_buffer.append(i)
        return local_buffer

    def BAstar(self, gs):
        self.Reduction(gs)#First call Reduction
        #self.BinaryBAstar()#Then call BinaryBAstar
        # fin=self.countVotes()
        # if(self.new_max_key!=fin):
        #     self.new_max_key = self.getEmptyString()
            #Check previous hash block if it matches enter empty in blockchain. Else ignore
        #self.insertBlock(self.new_max_key)

    def insertBlock(self, new_max_key, gs):
        gs.lock.acquire()
        roundno=str(len(gs.blockchain)-1)
        prevBlockHash = hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest()
        #print("Previous Block Hash ",prevBlockHash)
        #print("What I got ",new_max_key.split("-")[0])
        if prevBlockHash == new_max_key.split("-")[0]:
            block = Block(str(new_max_key),prevBlockHash)
            gs.blockchain.append(block)
            gs.lastBlockIndex += 1
            #print("Printing Blockchain:")
            #print("######################")
            #for i in gs.blockchain:
            #print(str(len(gs.blockchain)-1),",",block,",",new_max_key,",",prevBlockHash)
            #print("######################")
            gs.globalState = "blockAdded"
        gs.lock.release()


    def BinaryBAstar(self, gs):
        r=self.maximum_key
        empty=self.getEmptyString()
        # while self.step<self.MAXSTEPS:

        #     self.step += 1
        #     #PART 1
        if self.committeeElection(gs):
            #local_buffer=self.checkBlockProposal(gs)#Committee receives all the values and stores the one with minimum priority
            if self.maximum_key == -1: # DIPTYAROOP
                self.maximum_key = self.getEmptyString()
            self.committeeVote(self.maximum_key.split("-"), gs)#Committee members vote
            newEvent = Event()
            newEvent.timeStart = gs.time + gs.lambda_step
            newEvent.src = self.id
            newEvent.timeEnd = newEvent.timeStart
            newEvent.dest = self.id
            newEvent.action = "BinaryBAstar1"
            newEvent.msg = Message("BinaryBAstar1")
            heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
            gs.incrementSeqID()


    def BinaryBAstar1(self, gs):
        old=self.maximum_key
        self.countVotes(gs)#Call CountVotes get block in temp_block variable
        if self.maximum_key==-1:
            self.maximum_key=old
        elif self.maximum_key!=self.getEmptyString():
            for i in range(3):
                if self.committeeElection(gs):
                    self.committeeVote(self.maximum_key.split("-"), gs) #Committee members vote

            if self.step == 7:
                if self.committeeElection(gs):
                    self.committeeVote(self.maximum_key.split("-"), gs) #Committee members vote
            self.new_max_key = self.maximum_key
            newEvent = Event()
            newEvent.timeStart = gs.time + gs.lambda_step
            newEvent.src = self.id
            newEvent.timeEnd = newEvent.timeStart
            newEvent.dest = self.id
            newEvent.action = "BAstarFinal"
            newEvent.msg = Message("BAstarFinal")
            heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
            gs.incrementSeqID()
            return
        if self.committeeElection(gs):
            self.committeeVote(self.maximum_key.split("-"), gs)
        newEvent = Event()
        newEvent.timeStart = gs.time + gs.lambda_step
        newEvent.src = self.id
        newEvent.timeEnd = newEvent.timeStart
        newEvent.dest = self.id
        newEvent.action = "BinaryBAstar2"
        newEvent.msg = Message("BinaryBAstar2")
        heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
        gs.incrementSeqID()

    def BinaryBAstar2(self, gs):
        self.countVotes(gs)#Call CountVotes get block in temp_block variable
        if self.maximum_key==-1:
            self.maximum_key=self.getEmptyString()
        elif self.maximum_key==self.getEmptyString():
            for i in range(3):
                if self.committeeElection(gs):
                    self.committeeVote(self.maximum_key.split("-"), gs)
            self.new_max_key = self.maximum_key
            newEvent = Event()
            newEvent.timeStart = gs.time + gs.lambda_step
            newEvent.src = self.id
            newEvent.timeEnd = newEvent.timeStart
            newEvent.dest = self.id
            newEvent.action = "BAstarFinal"
            newEvent.msg = Message("BAstarFinal")
            heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
            gs.incrementSeqID()
            return
        if self.committeeElection(gs):
            self.committeeVote(self.maximum_key.split("-"), gs)
        newEvent = Event()
        newEvent.timeStart = gs.time + gs.lambda_step
        newEvent.src = self.id
        newEvent.timeEnd = newEvent.timeStart
        newEvent.dest = self.id
        newEvent.action = "BinaryBAstar3"
        newEvent.msg = Message("BinaryBAstar3")
        heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
        gs.incrementSeqID()

    def BinaryBAstar3(self, gs):
        r=self.countVotes(gs)#Call CountVotes get block in temp_block variable
        if r==-1:
            if random.choice([0,1])==0:
                pass
            else:
                self.maximum_key=self.getEmptyString()

        if self.step<=self.MAXSTEPS:
            newEvent = Event()
            newEvent.timeStart = gs.time
            newEvent.src = self.id
            newEvent.timeEnd = newEvent.timeStart
            newEvent.dest = self.id
            newEvent.action = "BinaryBAstar"
            newEvent.msg = Message("BinaryBAstar")
            heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
            gs.incrementSeqID()
        else:
            #print("ERROR ENCOUNTERED")
            pass

    def BAstarFinal(self, gs):
        #old=self.maximum_key
        fin=self.countVotes(gs)
        if(self.new_max_key!=fin):
            self.new_max_key = self.getEmptyString()
            # Check previous hash block if it matches enter empty in blockchain. Else ignore
        self.insertBlock(self.new_max_key, gs)

    def committeeVote(self,local_buffer, gs):
        #print("\nInside committeeVote",local_buffer)
        self.gossip(gs, local_buffer, "voting")
        # newEvent = Event()
        # newEvent.timeStart = gs.time + gs.lambda_step
        # newEvent.src = self.id
        # newEvent.timeEnd = newEvent.timeStart
        # newEvent.dest = self.id
        # newEvent.action = "countVotes"
        # newEvent.msg = Message("countVotes") 
        # heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
        # gs.incrementSeqID()


class Event:
    def __init__(self, start=0, delay=0, src=0, dest=0, action="none", msg="none", timeout=99999):
        self.timeStart = start
        self.src = src
        self.dest = dest
        self.timeEnd = self.timeStart + delay
        self.action = action
        self.msg = msg
        self.timeout = timeout

    def getAction(self):
        return self.action
    def getSrc(self):
        return self.src
    def getDest(self):
        return self.dest
    def getTimeStart(self):
        return self.timeStart

    def getMessage(self):
        return self.msg.getMessage()
    
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
    def getMessage(self):
        return str(self.msg)      

class Block:
    def __init__(self, msg, prevBlockHash):
        self.msg = msg
        self.prevBlockHash = prevBlockHash
        return


def start(gs, nodes):
    gs.algoStartTime = time.time()
    for node in nodes:
        threshold = gs.t_proposer
        role = "proposer"
        w = node.stake
        W = gs.totalStake
        node.step = 0
        roundno=len(gs.blockchain) - 1
        hash_prev=hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest()
        seed=hash_prev+str(roundno)+str(0)
        node.step=1
        #yan
        hashVal, pi, subUser = node.sortition(node.prKey, seed, threshold, w, W)
        
        if(subUser>0): # node selected as block proposer
            #print("\nNode = ", node.id, "stake = ", node.stake, "subusers = ", subUser)
            highestPriority = -1
            subUserWithHighestPriority = -1
            for j in range(0, subUser):
                priority = node.computePriorityForSubUser(hashVal, j)
                priority = int(priority, 16)
                if(highestPriority<0 or highestPriority>priority):
                    highestPriority = priority
                    subUserWithHighestPriority = j
            payload_list = []
            payload_list.append(roundno)
            payload_list.append(hashVal)
            payload_list.append(subUserWithHighestPriority)
            payload_list.append(highestPriority)
            payload_list.append(node.id)
            node.priority_payload = payload_list
            node.gossip(gs, payload_list, "priority")                
    #SORTITION PART DONE, NOW BLOCK PROPOSAL PART


def bootstrap(gs, nodes):
    cleanupFirstTime = True
    Blocks=1
    for qwerty in range(Blocks):
        start(gs, nodes)
        
        while True:
            try:
                ele = heapq.heappop(eventQ)
                curTime = round(ele[0], 3)
                gs.time = curTime
                #print("\ncurrent time = " + str(gs.time))
                curEvent = ele[2]
                if curEvent.getAction() == "send":
                    nodes[curEvent.src].send(gs,curEvent)
                elif curEvent.getAction() == "recv":
                    nodes[curEvent.dest].recv(gs,curEvent)
                elif curEvent.getAction() == "processRecvBuffer":
                    nodes[curEvent.src].processRecvBuffer()
                elif curEvent.getAction() == "proposeBlock":
                    nodes[curEvent.src].proposeBlock(gs)
                elif curEvent.getAction() == "checkBlockProposal":
                    nodes[curEvent.src].checkBlockProposal(gs)
                elif curEvent.getAction() == "reduction1":
                    nodes[curEvent.src].reduction1(gs)
                elif curEvent.getAction() == "reduction2":
                    nodes[curEvent.src].reduction2(gs)
                elif curEvent.getAction() == "reduction3":
                    nodes[curEvent.src].reduction3(gs)
                elif curEvent.getAction() == "BinaryBAstar":
                    nodes[curEvent.src].BinaryBAstar(gs)
                elif curEvent.getAction() == "BinaryBAstar1":
                    nodes[curEvent.src].BinaryBAstar1(gs)
                elif curEvent.getAction() == "BinaryBAstar2":
                    nodes[curEvent.src].BinaryBAstar2(gs)
                elif curEvent.getAction() == "BinaryBAstar3":
                    nodes[curEvent.src].BinaryBAstar3(gs)
                elif curEvent.getAction() == "BAstarFinal":
                    nodes[curEvent.src].BAstarFinal(gs)
                elif curEvent.getAction() == "cleanup":
                    gs.cleanup(nodes, cleanupFirstTime)
                    cleanupFirstTime = False
            except IndexError as e:
                #print("No events remaining. ", e)
                #print(gs.globalState, len(gs.blockchain))
                if(gs.globalState == "blockProposal"):
                    #gs.globalState =""
                    gs.startBAstar(nodes)
                elif(gs.globalState == "blockAdded"):
                    cleanupFirstTime = True
                    lengthOfBlockChain = len(gs.blockchain)
                    if(lengthOfBlockChain > int(sys.argv[2])):
                        gs.algoEndTime = time.time()
                        print (sys.argv[1],",", sys.argv[2],",", gs.algoStartTime,",", gs.algoEndTime,",", gs.algoEndTime-gs.algoStartTime)
                        gs.terminate()
                    gs.cleanup(nodes, cleanupFirstTime)
                    cleanupFirstTime = False
                    start(gs, nodes)
                else:
                    gs.terminate()
    

if __name__ == "__main__":
    if(len(sys.argv) < 4):
        print("Usage: python3 Mod_Algorand.py <#users> <#blocks> <normal/fs/byz>")
        exit(0)
    gs = GlobalState()
    node = []
    eventQ = []
    gs.time = 0
    gs.setBlockDelay(200, 400)
    gs.setNonBlockDelay(30, 64)

    for i in range(gs.numNodes):
        node.append(Node(i))
        if(sys.argv[3]=="byz" and i%gs.byzParameter==0):
            node[i].byz = True
        numNeighbors = math.floor(random.uniform(3, 8.1))
        nbrs = []
        while(numNeighbors>0):
            newNbr = random.randint(1, gs.numNodes) - 1 
            if(newNbr in nbrs or newNbr == i):
                continue
            nbrs.append(newNbr)
            numNeighbors = numNeighbors - 1
        node[i].generateKeyPair(gs)
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
    bootstrap(gs, node)

