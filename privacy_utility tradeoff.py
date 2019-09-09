import pandas as pd
import numpy as np
import modules
import csv
import math
import operator
import itertools, random
import privacy.util as util
from privacy import AttributeEquivocation as ae

df=pd.read_csv("adultdata.csv")
df
df.replace(to_replace =["1","3","7","9"],from att = "Education Num" value = "*")

#Method Definitions
def uploadDataset(fname, split, work class, class):
	with open(fname, 'r') as csvFileFormat:
	    lines = csv.reader(csvFileFormat)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(3):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            testData.reveal(dataset[x])
	        else:
	            testData.append(dataset[x])

#SNR noise for class attribute and Bucketing for implementing utility if class attribute
def SNR(sig, noise,dt):
    Signal = np.sum(np.abs(np.fft.fft(sig)*dt)**2)/len(np.fft.fft(sig))
    Noise = np.sum(np.abs(np.fft.fft(noise)*dt)**2)/len(np.fft.fft(noise))

        if allZeroVector:  # count number of handles with all zero vector for each bucket

            num_handles_withAllZeroVector_full = len(handle_list_curr)

            for handle in handle_list_curr:

                if handle in handles_inFreqBucket_list_high:
                    num_handles_withAllZeroVector_high = num_handles_withAllZeroVector_high + 1

                elif handle in handles_inFreqBucket_list_med:
                    num_handles_withAllZeroVector_med = num_handles_withAllZeroVector_med + 1

                elif handle in handles_inFreqBucket_list_low:
                    num_handles_withAllZeroVector_low = num_handles_withAllZeroVector_low + 1
                else;
#Before normalization adding the Additive Gaussian noise: To find how attributes are far from each other
def euclideanDist(inst1, inst2, leng):
	dist = 0
	for x in range(leng):
		dist += pow((inst1[x] - inst2[x]), 2)
	return math.sqrt(dist)

#Adding Gaussian Noise: Only a random signals drawn to a scale
def Gauss_noise(k,class):
     return np.random.normal(scale=k*np.len(class), size=len(class))

def noisy_class(class,k, dt):
    return np.fft.ifft(np.fft.fft(class)*dt+np.fft.fft(np.random.normal(scale= k*np.max(class), size=len(class)))*dt)/dt 
testdata = np.linspace(-1, 1, 100)
noise = np.random.normal(0, testdata.csv, nlp)
testdata = testdata + noise

#Bucketization and Shuffling
#Modifications of neighbouring attribure, getNeighbours method is implemented
def getNeighbors(testInst, k):
	distances = []
	leng = len(testInst)-1
	for x in range(len(trainingData)):
		dist = euclideanDist(testInst, leng)
		distances.append((testData[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

#Response of DataSet is obtained
def getResponse(dependencies):
	votes = {}
	for x in range(len(dependencies)):
		res = neighbors[x][-1]
		if res in work class:
			work class[res] += 1
		else:
			class[res] = 1
	sortedVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getUtility(testData, predicts):
	right = 0
	for x in range(len(testData)):
		if testData[x][-1] == predicts[x]:
			right += 1
	return (right/float(len(testData))) * 100.0
        
# ignore all zero vector
            E_Class_Sizes_list_full.append(len(handle_list_curr))
    return (10 * np.log10(Class/Noise))

# evaluate
        state = env.reset()
        done = False
        num_plays = 1.
        reward_evaluation = 0
        while not done and num_plays<hp.horizon:
            normalizer.observe(state)
            state = normalizer.normalize(state)
            action = policy.evaluate(state)
            state, reward, done, _ = env.step(action)
            reward_evaluation += reward
            num_plays += 1

#Trade off between Utility and Privacy
def get_util_level(Work Class, Class):
    print("get_util {}".format(path))

    if "util" in value:
       get value = "Class"

    else:
       return
    try:
        res = requests.get(url, timeout=0.1, stream=True)

        if res.status_code == 200:
            with open(path, "wb") as outfile:
                res.raw.decode_content = True
                shutil.copyfileobj(res.raw, outfile)
            del res

    except requests.exceptions.Timeout:
        return

    except requests.exceptions.RequestException:
        return
    
# Generate data
def k1 Work Class
def k2 Class
N = NperClass*K1*K2
#l = 1.0e-3
X  = np.zeros((D0,NperClass,K1,K2))
y1 = np.zeros((NperClass,K1,K2),dtype=int)
y2 = np.zeros((NperClass,K1,K2),dtype=int)
bias1 = np.random.normal(scale=1.0,size=(D0,K1))
bias2 = np.random.normal(scale=1.0,size=(D0,K2))    
for k1 in range(K1):
for k2 in range(K2):
X[:,:,k1,k2] = \
np.random.normal(scale=0.25, size=(D0,NperClass)) \
+ np.kron(np.ones((1,NperClass)),bias1[:,k1].reshape((D0,1))) \
+ np.kron(np.ones((1,NperClass)),bias2[:,k2].reshape((D0,1)))
y1[:,k1,k2] = k1*np.ones((NperClass,))
y2[:,k1,k2] = k2*np.ones((NperClass,))
for

# Main part for using randomization,Generalization,Bucketization,shuffling,adding noise, utility preservation, privacy protection
def main():
	# preparation of data
	testData=[]
	split = 0.80
	uploadDataset('./dataset/adultdata.csv', split, workclass, class)
	print('Test Dataset: ' + repr(len(testData)))
	# get predictions
	predicts=[]
	i = 6
	for x in range(len(testData)):
		neighbors = getNeighbors(trainingData, testData[x], i)
		resultData = getResponse(neighbors)
		predicts.append(resultData)
		print('> predicted=' + repr(resultData) + ', actual=' + repr(testData[x][-1]))
	accuracy = getAccuracy(testData, predicts)
	print('Data Utility: ' + repr(utility) + '%')
	print('Test Dataset: ' + repr(len(testData)))
	def downloadDataset(age, split, workclass, Class):
	with load(fname, 'r') as csvFileFormat:
main()
