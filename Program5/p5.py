print("\nNaive Bayes Classifier for concept learning problem")
import csv
import random
import math
import operator
def safe_div(x,y):
  if y == 0:
     return 0
  return x/y
def loadCsv(filename):
  lines=csv.reader(open(filename))
  dataset=list(lines)
  for i in range(len(dataset)):
     dataset[i]=[float(x)for x in dataset[i]]
  return dataset
def splitDataset(dataset,splitRatio):
  trainSize =int(len(dataset)*splitRatio)
  trainSet=[]
  copy =list(dataset)
  i=0
  while len(trainSet)<trainSize:
    trainSet.append(copy.pop(i))
  return[trainSet,copy]
def separateByClass(dataset):
  separated={}
  for i in range(len(dataset)):
      vector=dataset[i]
      if(vector[-1]not in separated):
          separated[vector[-1]]=[]
      separated[vector[-1]].append(vector)
  return separated
def mean(numbers):
    return safe_div(sum(numbers),float(len(numbers)))
def stdev(numbers):
    avg=mean(numbers)
    variance = safe_div(sum([pow(x-avg,2) for x in numbers]),float(len(numbers)-1))
    return math.sqrt(variance)
def summarize(dataset):
  summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
  print(summaries)
  del summaries[-1]
  return summaries
def summarizeByClass(dataset):
  separated = separateByClass(dataset)
  summaries = {}
  for classValue, instances in separated.items(): 
     summaries[classValue] = summarize(instances)
  return summaries
def calculateProbability(x, mean, stdev):
  exponent = math.exp(-safe_div(math.pow(x-mean,2),(2*math.pow(stdev,2))))
  final = safe_div(1,(math.sqrt(2*math.pi) * stdev)) * exponent
  return final
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
       probabilities[classValue] = 1
       for i in range(len(classSummaries)):
         mean, stdev = classSummaries[i]
         x = inputVector[i]
         probabilities[classValue] *= calculateProbability(x,mean,stdev)
    return probabilities
def predict(summaries, inputVector):
  probabilities = calculateClassProbabilities(summaries, inputVector)
  bestLabel, bestProb = None,-1
  for classValue, probability in probabilities.items():
    if bestLabel is None or probability > bestProb:
       bestProb = probability
       bestLabel = classValue
    return bestLabel
def getPredictions(summaries, testSet):
  predictions = []
  for i in range(len(testSet)):
     result = predict(summaries, testSet[i])
     predictions.append(result)
  return predictions
def getAccuracy(testSet, predictions):
  correct = 0
  for i in range(len(testSet)):
     if testSet[i][-1] == predictions[i]:
        correct += 1
        accuracy = safe_div(correct,float(len(testSet))) * 100.0
  return accuracy
def main():
  filename = 'pima-indians-diabetes.data.csv'
  splitRatio = 0.75
  dataset = loadCsv(filename)
  trainingSet, testSet = splitDataset(dataset, splitRatio)
  print('Split {0} rows into'.format(len(dataset)))
  print('Number of Training data: ' + (repr(len(trainingSet))))
  print('Number of Test Data: ' + (repr(len(testSet))))
  print("\nThe Training set are:")
  for x in trainingSet:
     print(x)
  print("\nThe Test data set are:")
  for x in testSet:
     print(x)
  summaries = summarizeByClass(trainingSet)
  predictions = getPredictions(summaries, testSet)
  actual = []
  for i in range(len(testSet)):
     vector = testSet[i]
     actual.append(vector[-1])
  print('Actual values: {0}%'.format(actual))
  print('Predictions: {0}%'.format(predictions))
  accuracy = getAccuracy(testSet, predictions)
  print('Accuracy: {0}%'.format(accuracy))
main()