from pomdp import *
import datetime
import spomdp

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
    gainThresh = 0.01 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 25000
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    useBudd = False
    revisedSplitting = False
    haveControl = False
    confidenceFactor = None
    localization_threshold = 0.75
    spomdp.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor, localization_threshold)

#Uses the Test 1 parameters outlined in the SBLTests.docx file with column updates (Collins' method) with Budd enabled
def test1_v3(filename,env):
    gainThresh = 0.01 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 25000
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    useBudd = True
    revisedSplitting = False
    haveControl = False
    confidenceFactor = None
    localization_threshold = 0.75
    spomdp.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor, localization_threshold)

#Uses the Test 2 parameters outlined in the SBLTests.docx file with random actions (no agent control)
def test2_v1(filename,env):
    haveControl = False
    confidenceFactor = 250
    gainThresh = 0.01 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 75000
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    useBudd = True
    revisedSplitting = False
    localization_threshold = 0.75
    spomdp.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor, localization_threshold)

#Uses the Test 2 parameters outlined in the SBLTests.docx file with agent control
def test2_v2(filename,env):
    haveControl = True
    confidenceFactor = 250
    gainThresh = 0.01 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 75000
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    useBudd = True
    revisedSplitting = False
    localization_threshold = 0.75
    spomdp.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor, localization_threshold)

#Uses the Test 2 parameters outlined in the SBLTests.docx file with agent control
def test2_v3(filename,env):
    useBudd = False
    haveControl = False
    confidenceFactor = 250
    gainThresh = 0.01 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 75000
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    revisedSplitting = False
    localization_threshold = 0.75
    spomdp.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor, localization_threshold)


#Uses the Test 3 parameters outlined in the SBLTests.docx file with Collins' method of SDE splitting
def test3_v1(filename,env):
    gainThresh = 0.05 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 25000
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    useBudd = False
    revisedSplitting = False
    haveControl = False
    confidenceFactor = None
    localization_threshold = 0.75
    spomdp.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor, localization_threshold)


#Uses the Test 3 parameters outlined in the SBLTests.docx file with improved SDE splitting
def test3_v3(filename,env):
    gainThresh = 0.05 #Threshold of gain to determine if the model should split (equivalent to surpriseThresh in budd.py)
    numActionsPerExperiment = 25000
    insertRandActions = False
    explore = 0.5 #Note: Since Collins' pseudocode does not insert random actions between SDEs, the default value for this is 0.5 (as suggested in the dissertation) if insertRandActions is not enabled. Otherwise use 0.05
    patience = 0
    useBudd = True
    revisedSplitting = True
    haveControl = False
    confidenceFactor = None
    localization_threshold = 0.75
    spomdp.psblLearning(env, numActionsPerExperiment, explore,patience,gainThresh, insertRandActions, True, filename, useBudd, revisedSplitting, haveControl, confidenceFactor, localization_threshold)
    
if __name__ == "__main__":

    '''----------BEGIN USER DEFINED TESTING PARAMETERS----------'''
    '''
    TestNum: Which test type to run. Use only the test number (e.g. For Test 3, use 3)
    Test 1: Test the transition posteriors update
    Test 2: Test the agent navigation algorithm
    Test 3: Test the invalid SDE splitting
    '''
    testNum = 2

    '''
    versionNum: Which test version to run. Use only the version number (e.g. For version 3, use 3)
    For Test 1: 1 corresponds to frequency-dependent transition posteriors update equation, 3 corresponds to our proposed frequency-independent transition posteriors update equation
    For Test 2: 1 corresponds to frequency-independent transition posteriors update equation without control, 2 corresponds to frequency-independent transition posteriors update equation with control, 3 corresponds to frequency-dependent transition posteriors update equation without control
    For Test 3: 1 corresponds to previous SDE generation algorithms, 3 corresponds to our proposed SDE generation algorithm with "safety checks"
    '''
    versionNum = 2

    '''
    envNum: The testing environment to test on
    1: ae-Shape fully built
    2: ae-Shape with initial observations
    3: ae-Litle Prince with initial observations
    32: ae-Little Prince fully built
    4: ae-1D Maze with initial observations
    42: ae-1D Maze fully built
    6: ae-Balance Beam fully built
    7: ae-Balance Beam with initial observations
    '''
    envNum = 2

    '''
    numSubTests: The number of tests to run consecutively
    '''
    numSubTests = 10

    '''----------END USER DEFINED TESTING PARAMETERS----------'''
    testString = "test"+str(testNum)+"_v"+str(versionNum)
    envString = "Example"+str(envNum)
    date = datetime.datetime.today()
    for subTest in range(numSubTests):
        filename = "Testing Data/Test" + str(testNum) + "_v" + str(versionNum) + "_env" + str(envNum) + "_" + str(date.month) + "_" + str(date.day) +  "_" + str(date.hour) + "_" + str(date.minute) + "_" + str(subTest) + ".csv"
        print(filename)
        env = locals()[envString]()
        locals()[testString](filename,env)
