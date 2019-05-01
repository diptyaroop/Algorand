import heapq
import random
import math
import ecdsa
import numpy as np
import hashlib
import time
from scipy.stats import binom

seqID = 0

class GlobalState:
    def __init__(self):
        self.INF = 99999
        self.seqID = 0
        self.time = 0
        self.numNodes = 10
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
        # adding blockchain
        self.blockchain = []
        self.blockchain.append(Block(self.genesisMsg, 0))
        self.lastBlockIndex = 0
        self.pubKey_list = []
        self.roundNum = 0

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
        self.recv_buffer = [] # to recieve incoming messages, can be anything, including a block
        self.state = "" # to simulate various states
        self.step = 0
        self.priority_payload = []
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
        #    print("good signature")
        #except BadSignatureError:
        #    print("BAD SIGNATURE")
        #open("private.pem","wb").write(sk.to_pem())
        #open("public.pem","wb").write(vk.to_pem())

    

    def processEvent(self):
        return

    def send(self, gs, event):
        if(event.msg.checkIfNodeVisited(self.id) == True):
            #print("Silently discarding")
            return # silently discarding
        if (gs.time > event.timeout):
            # stale message, discarding
            return
        print("Node "+str(event.getSrc()) + " send to Node " + str(event.getDest()) + " at time = " + str(event.getTimeStart()))
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
        #curNode = newEvent.getSrc()
        #for nbr in node[curNode].neighbors:
        #print("Node = "+str(curNode)+" , neighbor = "+str(nbr))
        heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
        gs.incrementSeqID()
        return
    
    def recv(self, gs, event):
        if (gs.validateSignature(event) == False):
            print("Invalid signature, abort\n")
            return
        if(gs.time > event.timeout):
            #stale message, discarding
            return
        print("Node "+str(event.getDest()) + " recvd from Node " + str(event.getSrc()) + " at time = " + str(event.getTimeStart()))
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
        #else:
        #    print("Node ",self.id," will now start processing recv_buffer.")
            # PROBLEM
            # SOMEHOW ONLY ONE NODE IS EXECUTING THIS ENTIRE THING
        return

    def PRG(self, seed):
    #def
        #Solution 1
        #print(random.getrandbits(256))
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

    def sortition(self, secret_key, seed, threshold, w, W):
        pi = (str(self.PRG(str(seed)))).encode('utf-8')  # seed = <hash of prev rnd || rnd num || step num >
        #print(pi)
        signature = secret_key.sign(pi) # my_hash has the signature
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
 
    def waitForBlockPriorityProposals(self, gs): # DIPTYAROOP : I don't know if this function is supposed to be used.
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

    def waitForVotes(self, gs):
        self.state = "wait_for_votes"
        newEvent = Event()
        newEvent.timeStart = gs.time + gs.lambda_proposal
        newEvent.src = self.id
        newEvent.timeEnd = newEvent.timeStart
        newEvent.dest = self.id
        newEvent.action = "countVotes"
        newEvent.msg = Message("start counting votes")
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
            node_id = payload_list[4]
            payload = str(roundno)+"-"+str(my_hash)+"-"+str(subUser)+"-"+str(priority)+"-"+str(node_id) # also concatenate round no. here
            # "-" will help to split the parameters easily
            msg = Message(payload)
            event = Event(gs.time, gs.nonBlockDelay[self.id][self.id], gs.time+gs.lambda_proposal, self.id, self.id, "send", msg)
            self.recv_buffer.append(event)
            self.waitForPriorityProposals(gs)
            for neighbor in self.neighbors:
                msg = Message(payload)
                event = Event(gs.time, gs.nonBlockDelay[self.id][neighbor], gs.time+gs.lambda_proposal, self.id, neighbor, "send", msg)
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
                event = Event(gs.time, gs.nonBlockDelay[self.id][neighbor], gs.time+gs.lambda_proposal, self.id, neighbor, "send", msg)
                heapq.heappush(eventQ, (event.getTimeStart(), gs.seqID, event))
                gs.incrementSeqID()
        elif  typeOfGossip == "blockProposal": #len(payload_list) == 7: 
            print("typeOfGossip = blockProposal, id = ", self.id)
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
                msg = Message(payload)
                timeout = gs.time+gs.lambda_block+gs.lambda_proposal
                event = Event(gs.time, gs.blockDelay[self.id][neighbor], timeout , self.id, neighbor, "send", msg)
                print("sending to ", neighbor, "at ", event.getTimeStart(), "timenow = ", gs.time)
                heapq.heappush(eventQ, (event.getTimeStart(), gs.seqID, event))
                gs.incrementSeqID()


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
        print("According to Node ",self.id," ",userWithHighestPriority, " is the highest priority user.")
        if(userWithHighestPriority == self.id):
            print("I, Node ",self.id, ", am going to propose the next block, subuser = ", subUserWithHighestPriority)
            newEvent = Event()
            newEvent.timeStart = gs.time
            newEvent.src = self.id
            newEvent.timeEnd = newEvent.timeStart
            newEvent.dest = self.id
            newEvent.action = "proposeBlock"
            newEvent.msg = Message("proposeBlock")
            heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
            gs.incrementSeqID()
        self.checkIfCommitteeMember(gs)
        return

    def checkIfCommitteeMember(self, gs):
        threshold = gs.t_step
        w = self.stake
        W = gs.totalStake
        roundno=len(gs.blockchain) - 1
        hash_prev=hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest() # Keep structure later
        seed=hash_prev+str(roundno)+str(1)
        hashVal, pi, j = self.sortition(self.prKey, seed, threshold, w, W)

        if j > 0 : # I am a committee member
            self.state = "wait_for_proposal"
            newEvent = Event()
            newEvent.timeStart = gs.time + gs.lambda_proposal + gs.lambda_block
            newEvent.src = self.id
            newEvent.timeEnd = newEvent.timeStart
            newEvent.dest = self.id
            newEvent.action = "checkBlockProposal"
            newEvent.msg = Message("checkBlockProposal") 
            heapq.heappush(eventQ, (newEvent.getTimeStart(), gs.seqID, newEvent))
            gs.incrementSeqID()
        return

    def proposeBlock(self, gs):
        #broadcast block to be proposed, delay = node.blockDelay
        print("Node ",self.id, " reporting for proposing a new block")
        print("neighbors : ", self.neighbors)
        message = "Block created by Node "+str(self.id)+" at round "+ str(gs.roundNum) # need to update round
        # using PRG code here. Can make it modular later.
        prevBlockHash = hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8'))

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
        print("I, ", self.id, " am a committee member")
        print("len(recv_buffer) = "+str(len(self.recv_buffer)))
        local_buffer = None
        local_priority = 2**257

        #Vote for highest priority block
        for i in range(len(self.recv_buffer)):
            msg = self.recv_buffer[i].getMessage().split("-")
            if len(msg) == 7:
                priority = float(msg[-2])
                pubKey = gs.pubKey_list[int(msg[-1])]
                prevBlockHash = hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest()
                roundno=len(gs.blockchain) - 1
                seed=prevBlockHash+str(roundno)+str(0)
                try:
                    pubKey.verify(bytes.fromhex(msg[3]),str(self.PRG(seed)).encode('utf-8'))
                    #print("Save me God.")
                    if local_buffer == None or priority<local_priority:
                        local_buffer = []
                        prevBlockHash = prevBlockHash # prevHash
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
                    print("Validation failure")
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
            local_buffer = []
            local_buffer.append(prevBlockHash)
            local_buffer.append(curBlockHash)
            local_buffer.append(roundno)
            local_buffer.append(step)
            local_buffer.append(subUserIndex)
            local_buffer.append(vrf_output)  
        
        vote =""
        for item in local_buffer:
            vote = vote + "," + str(item)
        self.gossip(gs, local_buffer, "voting")
        
        return

    
    #incomingMsgs is a bufer for storage of incoming messages?
    def countVotes(self, gs):#(self, ctx, round, step,T,tao ,lamda):
        return
    	#counts={}# // hash table, new keys mapped to 0
    	#voters={}
    	#msgs=incomingMsgs[rounds][step]
    	#for m in msgs:
		    #votes,value,sorthash=ProcessMsg(ctx,tao ,m)
    		#if pk in voters or votes ==0:
			    #continue
		    #voters = voters | {pk}
		    #counts[value] + = votes
		    #if counts[value] > T * tao:#// if we got enough votes, then output this value
    			#return value
	#return "timeout"
    


class Event:
    def __init__(self, start=0, delay=0, timeout=99999, src=0, dest=0, action="none", msg="none"):
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
    cleanupFirstTime = True
    #print(len(nodes))
    for node in nodes:
        #print(node.id, node.stake)
        threshold = gs.t_proposer
        role = "proposer"
        w = node.stake
        W = gs.totalStake
        #Deba
        node.step = 0
        roundno=len(gs.blockchain) - 1
        hash_prev=hashlib.sha256((gs.blockchain[gs.lastBlockIndex].msg).encode('utf-8')).hexdigest()
        seed=hash_prev+str(roundno)+str(0)
        #yan
        hashVal, pi, subUser = node.sortition(node.prKey, seed, threshold, w, W)
        if(subUser>0): # node selected as block proposer
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
            #print("Debayan hashval",hashVal)
            payload_list.append(subUserWithHighestPriority)
            payload_list.append(highestPriority)
            payload_list.append(node.id)
            node.priority_payload = payload_list
            node.gossip(gs, payload_list, "priority")
        else:
            node.checkIfCommitteeMember(gs)                
        print("Node = ", node.id, "stake = ", node.stake, "subusers = ", subUser)
        #print("SORTITION PART DONE, NOW BLOCK PROPOSAL PART")
    #SORTITION PART DONE, NOW BLOCK PROPOSAL PART
    curTime = 0
    while True:
        try:
            ele = heapq.heappop(eventQ)
            prevTime = curTime
            curTime = round(ele[0], 3)
            gs.time = curTime
            if(curTime - prevTime >= gs.lambda_block):
                for node in nodes:
                    node.waitForVotes(gs)
            curEvent = ele[2]
            print("current time = " + str(gs.time))
            if curEvent.getAction() == "send":
                nodes[curEvent.src].send(gs, curEvent)
            elif curEvent.getAction() == "recv":
                nodes[curEvent.dest].recv(gs, curEvent)
            elif curEvent.getAction() == "processRecvBuffer":
                nodes[curEvent.src].processRecvBuffer()
            elif curEvent.getAction() == "proposeBlock":
                nodes[curEvent.src].proposeBlock(gs)
            elif curEvent.getAction() == "checkBlockProposal":
                nodes[curEvent.src].checkBlockProposal(gs)
            elif curEvent.getAction() == "countVotes":
                nodes[curEvent.src].countVotes(gs)
            elif curEvent.getAction() == "cleanup":
                gs.cleanup(nodes, cleanupFirstTime)
                cleanupFirstTime = False

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
        node[i].generateKeyPair(gs)
        node[i].setNeighbors(nbrs)
        #print(i, ":", node[i].neighbors)
    gs.assignInitialStake(node)
    gs.storePublicKeys(node)
    
    #msg = Message("a")
    #event = Event(gs.time, gs.blockDelay[0][1], 0, 1, "send", msg)
    #heapq.heappush(eventQ, (gs.time, gs.seqID, event))
    #gs.incrementSeqID()
    #heapq.heappush(eventQ, (gs.time, gs.seqID, Event(gs.time, gs.blockDelay[0][2], 0, 2, "send", "a")))
    #gs.incrementSeqID()
    start(gs, node)

