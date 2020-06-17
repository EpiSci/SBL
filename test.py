import budd 
from pomdp import *
import datetime
import collins

#Writes a numpy matrix to an xls file. Returns the last row the matrix was written on. Currently supports only 3D numpy matrices.
def writeNumpyMatrixToFile(sheet, matrix, row=0,col=0):
    dimensions = matrix.shape
    rowCount = row
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            
            for k in range(dimensions[2]):
                sheet.write(rowCount,col+k, matrix[i][j][k])
            rowCount = rowCount + 1
        rowCount = rowCount + 1 #Provide an extra space between submatrices
    return rowCount

#Writes a numpy matrix to a csv file. Currently supports only 3D numpy matrices.
def writeNumpyMatrixToCSV(c, matrix):
    dimensions = matrix.shape
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            c.writerow(matrix[i][j][:])
        c.writerow([]) #Provide an extra space between submatrices


#Uses the Test 1 parameters outlined in the SBLTests.docx file with column updates (Collins' method)
def test1_v1(filename,env):
    gainThresh = 0 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 50000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    insertRandActions = True
    explore = 0.05 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    collins.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename)

#Uses the Test 1 parameters outlined in the SBLTests.docx file without column updates (Our method)
def test1_v2(filename,env):
    gainSplitThresh = 0.05 #Threshold of gain to determine if the model should stop learning
    surpriseThresh = 0 #0; used for one-step extension gain splitting
    entropyThresh = 0.55
    numActionsPerExperiment = 50000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    explore = 0.05
    percentTimeofBudd = 0.9
    budd.approximateSPOMDPLearning(env=env, gainThresh=gainSplitThresh, numActions=numActionsPerExperiment, explore=explore, surpriseThresh=surpriseThresh, splitWithEntropy=True, entropyThresh=entropyThresh,writeToFile=True, earlyTermination=False, budd=True, percentTimeofBudd=percentTimeofBudd, filename=filename)

#Uses the Test 2 parameters outlined in the SBLTests.docx file with random actions (no agent control)
def test2_v1(filename,env):
    gainThresh = 0.01 #Threshold of gain to determine if the model should stop learning
    surpriseThresh = 0 #0; used for one-step extension gain splitting
    entropyThresh = 0.4
    numActionsPerExperiment = 50000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    explore = 0.05
    percentTimeofBudd = 0.9
    budd.approximateSPOMDPLearning(env, gainThresh, numActionsPerExperiment, explore, surpriseThresh, splitWithEntropy=True, entropyThresh=entropyThresh, writeToFile=True, earlyTermination=True, budd=True, percentTimeofBudd=percentTimeofBudd, have_control = False, conservativeness_factor=0, confidence_factor=5, filename=filename)

#Uses the Test 2 parameters outlined in the SBLTests.docx file with agent control
def test2_v2(filename,env):
    gainThresh = 0.01 #Threshold of gain to determine if the model should stop learning
    surpriseThresh = 0 #0; used for one-step extension gain splitting
    entropyThresh = 0.4
    numActionsPerExperiment = 50000 #Note: for larger environments (e.g. Example5), this should be larger (e.g. 200,000)
    explore = 0.05
    percentTimeofBudd = 0.9
    budd.approximateSPOMDPLearning(env, gainThresh, numActionsPerExperiment, explore, surpriseThresh, splitWithEntropy=True, entropyThresh=entropyThresh, writeToFile=True, earlyTermination=True, budd=True, percentTimeofBudd=percentTimeofBudd, have_control = True, conservativeness_factor=0, confidence_factor=50, filename=filename)

    
if __name__ == "__main__":
    testNum = 1
    versionNum = 2
    envNum = 2
    numSubTests = 5
    testString = "test"+str(testNum)+"_v"+str(versionNum)
    envString = "Example"+str(envNum)

    date = datetime.datetime.today()

    # test1_v("hi2.csv",Example2())
    for subTest in range(5):
        filename = "Testing Data/Test" + str(testNum) + "_v" + str(versionNum) + "_env" + str(envNum) + "_" + str(date.month) + "_" + str(date.day) +  "_" + str(date.hour) + "_" + str(date.minute) + "_" + str(subTest) + ".csv"
        print(filename)
        env = locals()[envString]()
        locals()[testString](filename,env)
