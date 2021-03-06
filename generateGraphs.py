import csv
import glob
import numpy as np
import re
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from subprocess import check_call
from tempfile import NamedTemporaryFile
from pomdp import *
import codecs
import matplotlib.ticker as mtick


def generateGraphTest1(useFirstPoint,genAbsoluteError):
    environments = []
    files = glob.glob("./Testing Data/Test1_v" + str(1) + "/*.csv")

    for file in files:
        env = re.search("env\d+", file).group()
        env_num = env[len("env"):]
        environments.append(env_num)
    
    environments = set(environments)

    figure_num = 1
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # use list and not numpy array since we don't know how many iterations were done
    for env_num in environments:
        print("environment " + str(env_num))
        v1Data = []
        v2Data = []
        model_splits = []
        for versionNum in [1,3]:
            files = glob.glob("./Testing Data/Test1_v" + str(versionNum) + "/Test" + str(1) + "_v" + str(versionNum) + "_env" + str(env_num) + "*.csv")
            if len(files) == 0:
                continue
            data = []
            if versionNum == 1:
                data = v1Data
            else:
                data = v2Data

            validCount = 0
            totalCount = len(files)
            for filename in files:
                # find the returned model num
                finalModelNum = -1
                isValidSplitTrial = False
                with open(filename, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    foundFinal = False
                    for row in csv_reader:
                        if row['0'] == '*':
                            foundFinal = True
                            continue
                        if foundFinal is True and finalModelNum == -1:
                            temp = row['0']
                            finalModelNum = int(temp[len('Model Num '):])
                        if foundFinal is True and row['0'] == 'Absolute Error:':
                            absError = float(row['1'])
                            if absError < 1:
                                validCount = validCount+1
                                isValidSplitTrial = True

                if isValidSplitTrial:
                    with open(filename, mode='r') as csv_file:
                        csv_reader = csv.DictReader(csv_file)
                        iteration_num = 0
                        model_num = 0
                        offset_amount = 0
                        trialData = []
                        for row in csv_reader:
                            if model_num > finalModelNum:
                                break
                            if row['0'] == '*':
                                break
                            elif row['0'] == 'Model Num ' + str(model_num+1):
                                if iteration_num + offset_amount not in model_splits:
                                    model_splits.append(iteration_num + offset_amount)
                                model_num = model_num + 1
                                offset_amount = offset_amount + iteration_num + 1  # add the number of iterations from the last model + 1 (since we start counting at zero)
                            elif row['0'] == 'Iteration: ':
                                iteration_num = float(row['1'])
                            elif row['0'] == 'Error:' and genAbsoluteError is False:
                                if iteration_num == 0 and useFirstPoint is False:
                                    continue
                                trialData.append([iteration_num + offset_amount, float(row['1'])])
                            elif row['0'] == 'Absolute Error:' and genAbsoluteError:
                                if float(row['1']) < 1:
                                    if iteration_num == 0 and useFirstPoint is False:
                                        continue
                                    trialData.append([iteration_num + offset_amount, float(row['1'])])

                        data.append(trialData)

            print("Percent Trials Correct for Version " + str(versionNum) + " : " + str(validCount/totalCount))
        v1Data = np.array(v1Data)
        if v1Data.size > 0: # Check to make sure at least one trial was successful
            v1Data_average = np.mean(v1Data, axis=0)
            v1Data_stdDev = np.std(v1Data, axis=0)

        v2Data = np.array(v2Data)
        if v2Data.size > 0: # Check to make sure at least one trial was successful
            v2Data_average = np.mean(v2Data, axis=0)
            v2Data_stdDev = np.std(v2Data, axis=0)

        plt.rcParams.update({'font.size': 16})
        
        plt.figure(figure_num)
        figure_num = figure_num + 1

        if v1Data.size > 0: # Check to make sure at least one trial was successful
            plt.errorbar(v1Data_average[:,0], v1Data_average[:,1],fmt='.',yerr=v1Data_stdDev[:,1],ecolor=colors[0],label="Freq. Dep.\nPosterior Update",color=colors[0],markersize=14,capsize=8)
        
        if v2Data.size > 0:
            plt.errorbar(v2Data_average[:,0], v2Data_average[:,1],fmt='.',yerr=v2Data_stdDev[:,1],ecolor=colors[1],label="Freq. Ind.\nPosterior Update",color=colors[1],markersize=14,capsize=8)
        for num in range(len(model_splits)):
            split = model_splits[num]
            if num == 0:
                plt.axvline(x=split, color='gray', label="Model Split")
            else:
                plt.axvline(x=split, color='gray')
        plt.xlabel("Number of Actions Taken")
        yLabel = "Relative Error"
        errorType = "Rel. "
        errorFile = "Relative"
        if (genAbsoluteError):
            yLabel = "Absolute Error"
            errorType = "Abs. "
            errorFile = "Absolute"
        else:
            yLabel = "Relative Error"
            errorType = "Rel. "
            errorFile = "Relative"
        plt.ylabel(yLabel)
        plt.title("Posterior Update Comparison (" + errorType + "Model Error)\nFor " + str(getEnvironmentName(env_num)))
        leg = plt.legend(loc = 'lower left', fontsize=14)
        leg.get_frame().set_edgecolor('black')

        axes = plt.gca()
        axes.set_ylim([0,axes.get_ylim()[1]])
        axes.set_xlim([0,axes.get_xlim()[1]])

        plt.show()
        plt.savefig("Testing Data/Test1Env" + env_num + errorFile +"ErrorGraph.png", bbox_inches='tight')


def generateGraphTest2():
    # use list and not numpy array since we don't know how many iterations were done

    environments = []
    files = glob.glob("./Testing Data/Test2_v" + str(1) + "/*.csv")

    for file in files:
        env = re.search("env\d+", file).group()
        env_num = env[len("env"):]
        environments.append(env_num)
    
    environments = set(environments)

    figure_num = 1
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for env_num in environments:
        v1Data = []
        v2Data = []
        v3Data = []
        model_splits = []
        for versionNum in range(1, 4):
            files = glob.glob("./Testing Data/Test2_v" + str(versionNum) + "/Test" + str(2) + "_v" + str(versionNum) + "_env" + str(env_num) + "*.csv")
            data = []
            if versionNum == 1:
                data = v1Data
            elif versionNum == 2:
                data = v2Data
            else:
                data = v3Data
            for filename in files:
                # find the returned model num
                finalModelNum = -1
                isValidSplitTrial = False
                with open(filename, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    foundFinal = False
                    for row in csv_reader:
                        if row['0'] == '*':
                            foundFinal = True
                            continue
                        if foundFinal is True and finalModelNum == -1:
                            temp = row['0']
                            finalModelNum = int(temp[len('Model Num '):])
                        if foundFinal is True and row['0'] == 'Absolute Error:':
                            absError = float(row['1'])
                            if absError < 1:
                                isValidSplitTrial = True

                if not isValidSplitTrial:
                    continue
                
                with open(filename, mode='r') as csv_file:
                    iteration_num = 0
                    model_num = 0
                    offset_amount = 0
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        if model_num > finalModelNum:
                            break
                        if row['0'] == 'Model Num ' + str(model_num+1):
                            model_splits.append(iteration_num + offset_amount)
                            data.append([model_num, iteration_num])
                            model_num = model_num + 1
                            offset_amount = offset_amount + iteration_num + 1  # add the number of iterations from the last model + 1 (since we start counting at zero)
                        elif row['0'] == 'Iteration: ':
                            iteration_num = float(row['1'])
                    if model_num == finalModelNum:
                        data.append([model_num, iteration_num])


        v1Data = np.array(v1Data)
        v2Data = np.array(v2Data)
        v1Data_average = np.mean(v1Data, axis=0)
        v2Data_average = np.mean(v2Data, axis=0)
        v1Data_stdDev = np.std(v1Data, axis=0)
        v2Data_stdDev = np.std(v2Data, axis=0)

        # import pdb; pdb.set_trace()
        print("-----")
        print(env_num)
        v1_sum = np.sum(v1Data, axis=0)
        v2_sum = np.sum(v2Data, axis=0)
        print(v1_sum)
        print(v2_sum)
        v1_avg = v1_sum / 10
        v2_avg = v2_sum / 10
        print(v1_avg)
        print(v2_avg)
        print(v2_avg[1] / v1_avg[1])



        xData = []
        yData = []
        groupings = []

        if len(v3Data) > 0:
            v3Data = np.array(v3Data)
            v3Data_average = np.mean(v3Data, axis=0)
            v3Data_stdDev = np.std(v3Data, axis=0)
            xData = np.concatenate((v3Data[:,0], v1Data[:,0], v2Data[:,0]))
            yData = np.concatenate((v3Data[:,1], v1Data[:,1], v2Data[:,1]))
            groupings = np.concatenate((np.full(np.shape(v3Data[:,0]), "Collins with Posterior"), (np.full(np.shape(v1Data[:,0]), "Random Policy")), np.full(np.shape(v2Data[:,0]), "Navigation Policy")))
        else:
            xData = np.concatenate((v1Data[:,0], v2Data[:,0]))
            yData = np.concatenate((v1Data[:,1], v2Data[:,1]))
            groupings = np.concatenate(((np.full(np.shape(v1Data[:,0]), "Collins et al. Policy")), np.full(np.shape(v2Data[:,0]), "Proposed Navigation Policy")))


        plt.rcParams.update({'font.size': 16})

        plt.figure(figure_num)
        figure_num = figure_num + 1
        ax = sns.barplot(x=xData, y=yData, hue=groupings, capsize=0.1) 


        plt.xlabel("Model Split Number")
        plt.ylabel("Number of Actions Taken")
        ax.set_title("Number of Actions Taken per Model Split \nFor " + getEnvironmentName(env_num))
        xlabels = ['{:,d}'.format(x) for x in ax.get_xticks()]
        ax.set_xticklabels(xlabels)
        ax.tick_params(axis='x')
        ldg = plt.legend(fontsize=11.5)
        ldg.get_frame().set_edgecolor('black')

        # plt.show()
        plt.savefig("Testing Data/Test2env" + env_num + ".png", bbox_inches='tight')


def generateGraphTest2_2(useFirstPoint,genAbsoluteError, environmentNum):

    parameters = {'font.size': 16}
    plt.rcParams.update(parameters)

    # use list and not numpy array since we don't know how many iterations were done
    v1Data = []
    v2Data = []
    model_splits = []
    
    for versionNum in [1,2]:
        files = glob.glob("./Testing Data/Test2_v" + str(versionNum) + "/Test2_v" + str(versionNum) + "_env" + str(environmentNum) + "*.csv")
        if len(files) == 0:
            continue
        files = [files[0]]
        data = []
        if versionNum == 1:
            data = v1Data
        else:
            data = v2Data

        validCount = 0
        totalCount = 0
        for filename in files:
            totalCount = totalCount + 1
            finalModelNum = -1
            isValidSplitTrial = False
            with open(filename, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                foundFinal = False
                for row in csv_reader:
                    if row['0'] == '*':
                        foundFinal = True
                        continue
                    if foundFinal is True and finalModelNum == -1:
                        temp = row['0']
                        finalModelNum = int(temp[len('Model Num '):])
                    if foundFinal is True and row['0'] == 'Absolute Error:':
                        absError = float(row['1'])
                        if absError < 1:
                            validCount = validCount+1
                            isValidSplitTrial = True

            if isValidSplitTrial:
                with open(filename, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    iteration_num = 0
                    model_num = 0
                    offset_amount = 0
                    trialData = []
                    model_splits_row = []
                    for row in csv_reader:
                        if model_num > finalModelNum:
                            break
                        if row['0'] == '*':
                            break
                        elif row['0'] == 'Model Num ' + str(model_num+1):
                            if iteration_num + offset_amount not in model_splits and model_num+1 <= finalModelNum:
                                model_splits_row.append(iteration_num + offset_amount)
                            model_num = model_num + 1
                            offset_amount = offset_amount + iteration_num + 1  # add the number of iterations from the last model + 1 (since we start counting at zero)
                        elif row['0'] == 'Iteration: ':
                            iteration_num = float(row['1'])
                        elif row['0'] == 'Error:' and genAbsoluteError is False:
                            if iteration_num == 0 and useFirstPoint is False:
                                continue
                            trialData.append([iteration_num + offset_amount, float(row['1'])])
                        elif row['0'] == 'Absolute Error:' and genAbsoluteError:
                            if float(row['1']) < 1:
                                if iteration_num == 0 and useFirstPoint is False:
                                    continue
                                trialData.append([iteration_num + offset_amount, float(row['1'])])

                    data.append(trialData)
                    model_splits.append(model_splits_row)

        print("Percent Trials Correct for Version " + str(versionNum) + " : " + str(validCount/totalCount))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # import pdb; pdb.set_trace()
    v1Data = np.array(v1Data)
    if v1Data.size > 0: # Check to make sure at least one trial was successful
        v1Data_average = np.mean(v1Data, axis=0)
        v1Data_stdDev = np.std(v1Data, axis=0)

    v2Data = np.array(v2Data)
    if v2Data.size > 0: # Check to make sure at least one trial was successful
        v2Data_average = np.mean(v2Data, axis=0)
        v2Data_stdDev = np.std(v2Data, axis=0)
    
    if v1Data.size > 0: # Check to make sure at least one trial was successful
        print(v1Data_average.size)
        plt.scatter(v1Data_average[:,0], v1Data_average[:,1],label="Collins et al. Policy",c=colors[0])
    
    if v2Data.size > 0:
        plt.scatter(v2Data_average[:,0], v2Data_average[:,1],label="Proposed Navigation Policy",c=colors[1])
    for row_num in range(len(model_splits)):
        row = model_splits[row_num]
        color = ''
        grouping = ''
        linestyle = ''
        if row_num == 0:
            color = colors[0]
            grouping = "Collins et al. Policy"
            linestyle = "-"
        else:
            color = colors[1]
            grouping = "Proposed Navigation Policy"
            linestyle = "--"
        for num in range(len(row)):
            split = model_splits[row_num][num]
            if num == 0:
                plt.axvline(x=split, color=color, label= grouping + "\nModel Split", linestyle=linestyle)
            else:
                plt.axvline(x=split, color=color, linestyle=linestyle)


    plt.xlabel("Number of Actions Taken")
    plt.ylabel("Error")
    ax = plt.gca()
    ax.set_title("Model Error vs. Number of Actions Taken \nFor " + getEnvironmentName(str(environmentNum)))
    leg = plt.legend(fontsize=12)
    leg.get_frame().set_edgecolor('black')


    if useFirstPoint:
        ax.set_ylim([0,1])  # make it so that the y axis starts at zero and goes to 1

    plt.savefig("Testing Data/Test2_actions.png", bbox_inches='tight')
    plt.show()

# Generate bar graph for varying alpha values
def generateGraphTest2_3(numTrialsPerAlpha):
    # use list and not numpy array since we don't know how many iterations were done

    environments = []
    files = glob.glob("./Testing Data/Test2_v" + str(2) + "/*.csv")

    for file in files:
        env = re.search("env\d+", file).group()
        env_num = env[len("env"):]
        environments.append(env_num)
    
    environments = set(environments)

    figure_num = 1
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    prevAlphaVal = -1
    holder = []
    for env_num in environments:
        v1Data = []
        v2Data = []
        v3Data = []
        v1DataCorrect = []
        v2DataCorrect = []
        v3DataCorrect = []
        model_splits = []
        for versionNum in range(1, 4):
            files = glob.glob("./Testing Data/Test2_v" + str(versionNum) + "/Test" + str(2) + "_v" + str(versionNum) + "_env" + str(env_num) + "*.csv")
            totalTrials = len(files)
            data = []
            if versionNum == 1:
                data = v1Data
            elif versionNum == 2:
                data = v2Data
            else:
                data = v3Data
            for filename in files:
                # find the returned model num
                finalModelNum = -1
                isValidSplitTrial = False
                with open(filename, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    foundFinal = False
                    for row in csv_reader:
                        if row['0'] == '*':
                            foundFinal = True
                            continue
                        if foundFinal is True and finalModelNum == -1:
                            temp = row['0']
                            finalModelNum = int(temp[len('Model Num '):])
                        if foundFinal is True and row['0'] == 'Absolute Error:':
                            absError = float(row['1'])
                            if absError < 1:
                                isValidSplitTrial = True

                # if not isValidSplitTrial:
                #     continue
                
                
                alpha_val = 0   
                with open(filename, mode='r') as csv_file:
                    row_number = 0
                    offset_amount = 0
                    end_row = float('inf')
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        if row_number == 2: # This is the row where the alpha value is stored
                            alpha_val = float(row['2'])
                            # print(alpha_val)
                        elif row['0'] == '*': 
                            end_row = row_number
                        elif row['0'] == 'Absolute Error:' and row_number > end_row:
                            absErr = float(row['1'])
                            # if absErr <= 1:
                            data.append([alpha_val, absErr])
                            # if absErr < 1:
                            #     dataCorrect[alphaNum] = dataCorrect[alphaNum+1]


                        row_number = row_number + 1
                # if alpha_val != prevAlphaVal and prevAlphaVal != -1:
                #     data.append(np.array(holder))
                #     holder = []
                # holder.append([alpha_val, absErr])
                # prevAlphaVal = alpha_val

                    # if model_num == finalModelNum:
                    #     data.append([model_num, iteration_num])
            # data.append(np.array(holder))


        v1Data = np.array(v1Data)
        v2Data = np.array(v2Data)
        v2Data = v2Data[v2Data[:,0].argsort()] # sort based off of the x values (e.g. the alpha value)
        v2Data = np.flipud(v2Data)

        ratioDividendv2 = v2Data.copy()

        print(v2Data)

        print(v2Data)
        v1Data_average = np.mean(v1Data, axis=0)
        v2Data_average = []
        for alphaTrial in v2Data:
            v2Data_average.append(np.mean(alphaTrial, axis=0))
        v2Data_average = np.array(v2Data_average)
        
        # v2Data_average = np.mean(v2Data, axis=0)
        v1Data_stdDev = np.std(v1Data, axis=0)
        v2Data_stdDev = []
        for alphaTrial in v2Data:
            v2Data_stdDev.append(np.std(alphaTrial, axis=0))
        v2Data_stdDev = np.array(v2Data_stdDev)
        # v2Data_stdDev = np.std(v2Data, axis=0)
        correctLocs2 = v2Data[:,1] < 1
        v2Accuracy = []
        i = 0
        while i < len(v2Data):
            j = 0
            alphaSum = 0
            while j < numTrialsPerAlpha:
                alphaSum = alphaSum + correctLocs2[i+j]
                j = j + 1
            v2Accuracy.append(alphaSum / numTrialsPerAlpha)
            i = i + numTrialsPerAlpha
        print(correctLocs2)
        print(v2Accuracy)
        v2Accuracy = np.array(v2Accuracy)

        v2Data = np.array([row for row in v2Data if row[1] < 1])
        print(v2Data)
        # import pdb; pdb.set_trace()
        print("-----")
        print(env_num)
        # v1_sum = np.sum(v1Data, axis=0)
        # v2_sum = np.sum(v2Data, axis=0)
        # # print(v1_sum)
        # print(v2_sum)
        # v1_avg = v1_sum / 10
        # v2_avg = v2_sum / 10
        # print(v1_avg)
        # print(v2_avg)
        # print(v2Data)
        # print(v2_avg[1] / v1_avg[1])



        xData = []
        yData = []
        groupings = []

        if len(v3Data) > 0 and len(v1Data) > 0:
            correctLocs1 = v1Data[:,1] < 1
            v1Accuracy = []
            i = 0
            while i < len(v1Data):
                j = 0
                alphaSum = 0
                while j < numTrialsPerAlpha:
                    alphaSum = alphaSum + correctLocs1[i+j]
                    j = j + 1
                v1Accuracy.append(alphaSum / numTrialsPerAlpha)
                i = i + numTrialsPerAlpha
            print(v1Accuracy)
            v1Data = np.array(v1Data)
            print(v1Data)

            correctLocs3 = v3Data[:,1] < 1
            v3Accuracy = []
            i = 0
            while i < len(v3Data):
                j = 0
                alphaSum = 0
                while j < numTrialsPerAlpha:
                    alphaSum = alphaSum + correctLocs3[i+j]
                    j = j + 1
                v3Accuracy.append(alphaSum / numTrialsPerAlpha)
                i = i + numTrialsPerAlpha
            print(v3Accuracy)
            v3Data = np.array(v3Data)
            print(v3Data)

            xData = np.concatenate((v3Data[:,0], v1Data[:,0], v2Data[:,0]))
            yData = np.concatenate((v3Data[:,1], v1Data[:,1], v2Data[:,1]))
            yDataAccuracy = np.concatenate((v3Accuracy, v1Accuracy, v2Accuracy))
            groupings = np.concatenate((np.full(np.shape(v3Data[:,0]), "Original Collins et al. Policy"), (np.full(np.shape(v1Data[:,0]), "Collins et al. Policy with Freq. Indep. Updates")), np.full(np.shape(v2Data[:,0]), "Proposed Naviagation Policy")))
        if len(v3Data) > 0:
            correctLocs3 = v3Data[:,1] < 1
            v3Accuracy = []
            i = 0
            while i < len(v3Data):
                j = 0
                alphaSum = 0
                while j < numTrialsPerAlpha:
                    alphaSum = alphaSum + correctLocs3[i+j]
                    j = j + 1
                v3Accuracy.append(alphaSum / numTrialsPerAlpha)
                i = i + numTrialsPerAlpha
            print(v3Accuracy)
            v3Data = np.array(v3Data)
            print(v3Data)

            v3Data = np.array(v3Data)
            print(v3Data)

            xData = np.concatenate((v3Data[:,0], v2Data[:,0]))
            yData = np.concatenate((v3Data[:,1], v2Data[:,1]))
            yDataAccuracy = np.concatenate((v3Accuracy, v2Accuracy))
            groupings = np.concatenate((np.full(np.shape(v3Data[:,0]), "Collins et al. Policy"), np.full(np.shape(v2Data[:,0]), "Proposed Navigation Policy")))
        elif len(v1Data) > 0:
            v1Data = v1Data[v1Data[:,0].argsort()] # sort based off of the x values (e.g. the alpha value)
            v1Data = np.flipud(v1Data)
            ratioDividendv1 = v1Data.copy()


            correctLocs1 = v1Data[:,1] < 1
            v1Accuracy = []
            i = 0
            while i < len(v1Data):
                j = 0
                alphaSum = 0
                while j < numTrialsPerAlpha:
                    alphaSum = alphaSum + correctLocs1[i+j]
                    j = j + 1
                v1Accuracy.append(alphaSum / numTrialsPerAlpha)
                i = i + numTrialsPerAlpha
            print(v1Accuracy)
            v1Accuracy = np.array(v1Accuracy)

            # Parse v1Data and remove any values that have an absolute error of 1
            v1Data = np.array([row for row in v1Data if row[1] < 1])
            v1Data = np.array(v1Data)
            print(v1Data)


            xData = np.concatenate((v1Data[:,0], v2Data[:,0]))
            yData = np.concatenate((v1Data[:,1], v2Data[:,1]))
            yDataAccuracy = np.concatenate((v1Accuracy, v2Accuracy))
            groupings = np.concatenate((np.full(np.shape(v1Data[:,0]), "Collins et al. Policy"), np.full(np.shape(v2Data[:,0]), "Proposed Navigation Policy")))
            groupingsAccuracy = np.concatenate((np.full(np.shape(v1Accuracy), "Collins et al. Policy"), np.full(np.shape(v2Accuracy), "Proposed Navigation Policy")))
        else:
            xData = v2Data[:,0]
            print(xData)
            yData = v2Data[:,1]
            groupings = (np.full(np.shape(v2Data[:,0]), "Proposed Navigation Policy"))

        plt.rcParams.update({'font.size': 16})

        plt.figure(figure_num)
        figure_num = figure_num + 1
        print(xData)
        ax = sns.barplot(x=xData, y=yData, hue=groupings, capsize=0.1) 

        uniqeXVals = np.unique(xData)

        plt.xlabel("Alpha Value")
        plt.ylabel("Error")
        ax.set_title("Model Transition Probability Error Versus Alpha \nFor " + getEnvironmentName(env_num))
        # xlabels = ['{:,d}'.format(x) for x in ax.get_xticks()]
        xlabels = [x for x in uniqeXVals]
        ax.set_xticklabels(xlabels)
        ax.tick_params(axis='x')
        ax.invert_xaxis()
        ldg = plt.legend(fontsize=11.5)
        ldg.get_frame().set_edgecolor('black')

        # plt.show()
        plt.savefig("Testing Data/Test2ErrorAlphaVarying_env" + env_num + ".png", bbox_inches='tight')

        # =========================== Make the histogram plot for accuracy
        plt.rcParams.update({'font.size': 16})
        plt.figure(figure_num)
        figure_num = figure_num + 1
        xDataAccuracy = (np.unique(xData))
        xDataAccuracy = np.append(xDataAccuracy, xDataAccuracy)
        xDataAccuracy = np.flipud(xDataAccuracy)
        print(xDataAccuracy)
        print(yDataAccuracy)
        print(groupingsAccuracy)
        ax = sns.barplot(x=xDataAccuracy, y=yDataAccuracy, hue=groupingsAccuracy, capsize=0.1) 

        uniqeXVals = np.unique(xData)

        plt.xlabel("Alpha Value")
        plt.ylabel("Success Rate")
        ax.set_title("Successful Model Generation Versus Alpha \nFor " + getEnvironmentName(env_num))
        # xlabels = ['{:,d}'.format(x) for x in ax.get_xticks()]
        xlabels = [x for x in uniqeXVals]
        ax.set_xticklabels(xlabels)
        ax.tick_params(axis='x')
        ax.invert_xaxis()
        ldg = plt.legend(fontsize=11.5)
        ldg.get_frame().set_edgecolor('black')

        # plt.show()
        plt.savefig("Testing Data/Test2AccuracyAlphaVarying_env" + env_num + ".png", bbox_inches='tight')

        # =========================== Make the plot for success/error plot
        plt.rcParams.update({'font.size': 16})
        plt.figure(figure_num)
        figure_num = figure_num + 1
        ratioDividend = np.concatenate((ratioDividendv1, ratioDividendv2))
        ratioDividend = ratioDividend[:,1]

        appendedAccuracy = np.array([])
        for i in range(len(yDataAccuracy)):
            appendedAccuracy = np.append(appendedAccuracy, np.repeat(yDataAccuracy[i], yDataAccuracy[i]*10))

        newRatio = []
        for i in range(len(ratioDividend)):
            if ratioDividend[i] < 1:
                newRatio.append(ratioDividend[i])
        ratioDividend = np.array(newRatio)
                
        print("!!!!!!!!!!!")
        print(appendedAccuracy)
        print("##########")
        print(ratioDividend)
        print("^^^^^^^^^^^^")

        yDataRatio = appendedAccuracy/ratioDividend
        # yDataRatio[ratioDividend == 1] = 0 # Don't count the trials that have an absolue error of 1

        print(xDataAccuracy)
        print(yDataRatio)
        print(groupings)
        ax = sns.barplot(x=xData, y=yDataRatio, hue=groupings, capsize=0.1) 

        uniqeXVals = np.unique(xData)

        plt.xlabel("Alpha Value")
        plt.ylabel("(Success Rate) / Error")
        ax.set_title("Normalized Success Rate and Error Versus Alpha \nFor " + getEnvironmentName(env_num))
        # xlabels = ['{:,d}'.format(x) for x in ax.get_xticks()]
        xlabels = [x for x in uniqeXVals]
        ax.set_xticklabels(xlabels)
        ax.tick_params(axis='x')
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(0, 12)
        ax.invert_xaxis()
        ldg = plt.legend(fontsize=11.5)
        ldg.get_frame().set_edgecolor('black')

        # plt.show()
        plt.savefig("Testing Data/Test2SuccessOverErrorAlphaVarying_env" + env_num + ".png", bbox_inches='tight')


def generateGraphTest3(useFirstPoint,genAbsoluteError):
    environments = []
    files = glob.glob("./Testing Data/Test3_v" + str(1) + "/*.csv")

    for file in files:
        env = re.search("env\d+", file).group()
        env_num = env[len("env"):]
        environments.append(env_num)
    
    environments = set(environments)

    figure_num = 1
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for env_num in environments:
        print(env_num)
        # use list and not numpy array since we don't know how many iterations were done
        v1Data = []
        v2Data = []
        model_splits = []

        
        for versionNum in [1,3]:
            files = glob.glob("./Testing Data/Test3_v" + str(versionNum) + "/Test" + str(3) + "_v" + str(versionNum) + "_env" + str(env_num) + "*.csv")
            if len(files) == 0:
                continue
            firstFile = files[0]
            data = []
            if versionNum == 1:
                data = v1Data
            else:
                data = v2Data

            validCount = 0
            totalCount = len(files)
            for filename in files:
                # find the returned model num
                finalModelNum = -1
                isValidSplitTrial = False
                with open(filename, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    foundFinal = False
                    for row in csv_reader:
                        if row['0'] == '*':
                            foundFinal = True
                            continue
                        if foundFinal is True and finalModelNum == -1:
                            temp = row['0']
                            finalModelNum = int(temp[len('Model Num '):])
                        if foundFinal is True and row['0'] == 'Absolute Error:':
                            absError = float(row['1'])
                            if absError < 1:
                                validCount = validCount+1
                                isValidSplitTrial = True

                if isValidSplitTrial:
                    with open(filename, mode='r') as csv_file:
                        csv_reader = csv.DictReader(csv_file)
                        iteration_num = 0
                        model_num = 0
                        offset_amount = 0
                        trialData = []
                        for row in csv_reader:
                            if model_num > finalModelNum:
                                break
                            if row['0'] == '*':
                                break
                            elif row['0'] == 'Model Num ' + str(model_num+1):
                                if iteration_num + offset_amount not in model_splits:
                                    model_splits.append(iteration_num + offset_amount)
                                model_num = model_num + 1
                                offset_amount = offset_amount + iteration_num + 1  # add the number of iterations from the last model + 1 (since we start counting at zero)
                            elif row['0'] == 'Iteration: ':
                                iteration_num = float(row['1'])
                            elif row['0'] == 'Error:' and genAbsoluteError is False:
                                if iteration_num == 0 and useFirstPoint is False:
                                    continue
                                trialData.append([iteration_num + offset_amount, float(row['1'])])
                            elif row['0'] == 'Absolute Error:' and genAbsoluteError:
                                if float(row['1']) < 1:
                                    if iteration_num == 0 and useFirstPoint is False:
                                        continue
                                    trialData.append([iteration_num + offset_amount, float(row['1'])])

                        data.append(trialData)

            print("Percent Trials Correct for Version " + str(versionNum) + " : " + str(validCount/totalCount))
        v1Data = np.array(v1Data)
        if v1Data.size > 0: # Check to make sure at least one trial was successful
            v1Data_average = np.mean(v1Data, axis=0)
            v1Data_stdDev = np.std(v1Data, axis=0)

        v2Data = np.array(v2Data)
        if v2Data.size > 0: # Check to make sure at least one trial was successful
            v2Data_average = np.mean(v2Data, axis=0)
            v2Data_stdDev = np.std(v2Data, axis=0)

        plt.rcParams.update({'font.size': 16})
        
        plt.figure(figure_num)
        figure_num = figure_num + 1
        
        if v1Data.size > 0: # Check to make sure at least one trial was successful
            print(v1Data_average.size)
            plt.errorbar(v1Data_average[:,0], v1Data_average[:,1],fmt='.',yerr=v1Data_stdDev[:,1],ecolor=colors[0],label="Collins et al. \nSDE Generation",color=colors[0],markersize=10,capsize=5)
        
        if v2Data.size > 0:
            plt.errorbar(v2Data_average[:,0], v2Data_average[:,1],fmt='.',yerr=v2Data_stdDev[:,1],ecolor=colors[1],label="Revised \nSDE Generation",color=colors[1],markersize=10,capsize=5)
        for num in range(len(model_splits)):
            split = model_splits[num]
            if num == 0:
                plt.axvline(x=split, color='gray', label="Model Split")
            else:
                plt.axvline(x=split, color='gray')
        plt.xlabel("Number of Actions Taken")
        yLabel = "Relative Error"
        errorType = "Rel. "
        errorFile = "Relative"
        if (genAbsoluteError):
            yLabel = "Absolute Error"
            errorType = "Abs. "
            errorFile = "Absolute"
        else:
            yLabel = "Relative Error"
            errorType = "Rel. "
            errorFile = "Relative"
        plt.ylabel(yLabel)
        plt.title("SDE Splitting Comparison (" + errorType + "Model Error)\nFor " + str(getEnvironmentName(env_num)))
        leg = plt.legend(loc = 'lower left', fontsize=14)
        leg.get_frame().set_edgecolor('black')

        axes = plt.gca()
        axes.set_ylim([0,axes.get_ylim()[1]])
        axes.set_xlim([0,axes.get_xlim()[1]])

        plt.show()
        plt.savefig("Testing Data/Test3Env" + env_num + errorFile +"ErrorGraph.png", bbox_inches='tight')


def getEnvironmentName(env_num):
    if env_num == "2":
        return "αϵ-Shape Environment"
    elif env_num == "3":
        return "αϵ-Little Prince Environment"
    elif env_num == "4":
        return "αϵ-1D Maze Environment"
    elif env_num == "7":
        return "αϵ-Balance Beam Environment"
    return "Environment " + env_num





def getModelGraph(env_num, SDE_Set, A_S, transitionProbs, filename):
        """
        Write graph to a temporary file and invoke `dot`.

        The output file type is automatically detected from the file suffix.

        *`graphviz` needs to be installed, before usage of this method.*
        """
        lines = []
        lines.append("digraph tree {")
        # add the nodes
        for sde_idx in range(len(SDE_Set)):
            sde = SDE_Set[sde_idx]
            line = '    "' + str(sde_idx) + '" ['
            sde_str = "\n("
            for m_a in sde:
                if m_a == "square":
                    sde_str = sde_str + "&#9633;,"
                elif m_a == "diamond":
                    sde_str = sde_str + "&#9674;,"
                else:
                    sde_str = sde_str + m_a + ","
            sde_str = sde_str[:-1] + ')'  # -1 to get rid of comma
            label = str(sde_idx) + sde_str
            line = line + 'label="' + label + '"'
            if sde[0] == "square":
                line = line + ', shape="square"'
            elif sde[0] == "diamond":
                line = line + ', shape="diamond", height=1.7'
            elif sde[0] == "volcano":
                line = line + ', shape="trapezium"'
            elif sde[0] == "rose":
                line = line + ', shape="polygon", sides=7'
            elif sde[0] == "goal":
                line = line + ', shape="square"'
            line = line + ', style=bold'
            line = line + ', fontname="Times-Bold"'
            line = line + '];'
            lines.append(line)

        # do the ranks
        if env_num == 6:
            for sde_idx in range(len(SDE_Set)):
                line = ""
                if sde_idx == 1: # we'll do this one when sde_idx == 2
                    continue
                elif sde_idx == 2:
                    line = '  { rank=min; "1"; "2"; }'
                else:
                    rank = "same"
                    if sde_idx == 3:
                        rank = "source"
                    elif sde_idx == 0:
                        rank = "sink"
                    line = '  { rank=' + rank + '; ' + '"' + str(sde_idx) + '"; }'
                lines.append(line)
        elif env_num == 1:
            for sde_idx in range(len(SDE_Set)):
                line = ""
                if sde_idx == 3: # we'll do this one when sde_idx == 2
                    line = '  { rank=same; "1"; "0"; }'
                elif sde_idx == 2:
                    line = '  { rank=same; "2"; "3"; }'
                else:
                    continue
                lines.append(line)
        # elif env_num == 42:
        #     line = '  { rank=same; "0"; "1"; "2"; "3"; }'
        #     lines.append(line)

        # add the edges
        for m_idx in range(len(SDE_Set)):
            for a_idx in range(len(A_S)):
                row = transitionProbs[a_idx, m_idx, :]
                m_p_idx = np.argmax(row)
                probability = np.max(row)
                line = '    "' + str(m_idx) + '" -> "' + str(m_p_idx) + '" '
                line = line + '[label=" ' + A_S[a_idx] + '\n '+ str(probability) + '"'
                line = line + ', style=bold'
                line = line + ', fontname="Times-Bold"'
                # if abs(m_idx - m_p_idx) == 1:
                #     line = line + ', weight=4'
                # elif abs(m_idx - m_p_idx) == 3:
                #     line = line + ', weight=1'
                # else:
                #     line = line + ', weight=1'
                line = line + '];'

                lines.append(line)

        lines.append("}")

        # now write the file
        with codecs.open("dotfile.dot", "w", "utf-8") as dotfile:
        # with NamedTemporaryFile("wb", delete=False) as dotfile:
            dotfilename = dotfile.name
            for line in lines:
                dotfile.write("%s\n" % line)
                # dotfile.write(("%s\n" % line).encode("utf-8"))
            dotfile.flush()
            cmd = ["dot", dotfilename, "-T", "png", "-o", filename]
            # dot dotfile.dot -T png -o temp.png
            check_call(cmd)
        try:
            remove(dotfilename)
        except Exception:
            msg = 'Could not remove temporary file %s' % dotfilename
        
        img=mpimg.imread(filename)
        imgplot = plt.imshow(img)
        plt.show()


def getPercentAccurate(environmentNum):    
    for versionNum in [1,2]:
        files = glob.glob("./Testing Data/Test2_v" + str(versionNum) + "/Test2_v" + str(versionNum) + "_env" + str(environmentNum) + "*.csv")
        if len(files) == 0:
            continue

        validCount = 0
        totalCount = 0
        for filename in files:
            totalCount = totalCount + 1
            finalModelNum = -1
            isValidSplitTrial = False
            with open(filename, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                foundFinal = False
                for row in csv_reader:
                    if row['0'] == '*':
                        foundFinal = True
                        continue
                    if foundFinal is True and finalModelNum == -1:
                        temp = row['0']
                        finalModelNum = int(temp[len('Model Num '):])
                    if foundFinal is True and row['0'] == 'Absolute Error:':
                        absError = float(row['1'])
                        if absError < 1:
                            validCount = validCount+1
                            isValidSplitTrial = True

        print("Percent Trials Correct for Version " + str(versionNum) + " : " + str(validCount/totalCount))

if __name__ == "__main__":

    # Test 1 (Testing Transition Posterior Update Equation)
    # Relative Error
    # generateGraphTest1(False,False)

    # Test 1 (Testing Transition Posterior Update Equation)
    # Absolute Error
    # generateGraphTest1(False,True)


    # Test 2 (Testing Agent Autonomous Navigation Algorithm)
    # Bar Graphs for Relative Error
    # generateGraphTest2()

    # Test 2 (Testing Agent Autonomous Navigation Algorithm)
    # Scatter Plot for Single Trial Comparison, Shape Environment
    # generateGraphTest2_2(False, False, 2)

    # Test 2 (Testing Agent Autonomous Navigation Algorithm)
    # Bar Graph for Varying Alpha
    generateGraphTest2_3(10)


    # Test 3 (Testing SDE Generation Algorithms)
    # Relative Error
    # generateGraphTest3(False,False)

    # Test 3 (Testing SDE Generation Algorithms)
    # Absolute Error
    # generateGraphTest3(False,True)
