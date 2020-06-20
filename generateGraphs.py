import csv
import glob
import numpy as np
from matplotlib import pyplot as plt


def generateGraphTest1():
    # use list and not numpy array since we don't know how many iterations were done
    v1Data = []
    v2Data = []
    model_splits = []
    
    for versionNum in range(1, 1+2):
        files = glob.glob("./Testing Data/Test1_v" + str(versionNum) + "/*.csv")
        data = []
        if versionNum == 1:
            data = v1Data
        else:
            data = v2Data
        for filename in files:
            with open(filename, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                iteration_num = 0
                model_num = 0
                offset_amount = 0
                trialData = []
                for row in csv_reader:
                    if row['0'] == 'Model Num ' + str(model_num+1):
                        if iteration_num + offset_amount not in model_splits:
                            model_splits.append(iteration_num + offset_amount)
                        model_num = model_num + 1
                        offset_amount = offset_amount + iteration_num + 1  # add the number of iterations from the last model + 1 (since we start counting at zero)
                    elif row['0'] == 'Iteration: ':
                        iteration_num = float(row['1'])
                    elif row['0'] == 'Error:':
                        trialData.append([iteration_num + offset_amount, float(row['1'])])
                data.append(trialData)

    v1Data = np.array(v1Data)
    v2Data = np.array(v2Data)
    v1Data_average = np.mean(v1Data, axis=0)
    v2Data_average = np.mean(v2Data, axis=0)

    plt.scatter(v1Data_average[:,0], v1Data_average[:,1], label="Collins")
    plt.scatter(v2Data_average[:,0], v2Data_average[:,1], label="BUDD")
    for num in range(len(model_splits)):
        split = model_splits[num]
        if num == 0:
            plt.axvline(x=split, color='gray', label="Model Split")
        else:
            plt.axvline(x=split, color='gray')
    plt.xlabel("Number of Actions Taken")
    plt.ylabel("Error")
    plt.title("Model Error vs. Number of Actions Taken")
    plt.legend()

    axes = plt.gca()
    axes.set_ylim([0,1])  # make it so that the y axis starts at zero and goes to 1

    plt.show()


if __name__ == "__main__":
    testNum = 1
    versionNum = 2
    envNum = 2
    numSubTests = 5
    
    generateGraphTest1()

