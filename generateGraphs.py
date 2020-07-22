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


def generateGraphTest1(useFirstPoint,genAbsoluteError):
    # use list and not numpy array since we don't know how many iterations were done
    v1Data = []
    v2Data = []
    model_splits = []
    
    for versionNum in [1,3]:
        files = glob.glob("./Testing Data/Test1_v" + str(versionNum) + "/*.csv")
        if len(files) == 0:
            continue
        firstFile = files[0]
        env = re.search("env\d+", firstFile).group()
        env_num = env[len("env"):]
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
    
    # plt.scatter(v1Data_average[:,0], v1Data_average[:,1], label="Collins")
    if v1Data.size > 0: # Check to make sure at least one trial was successful
        print(v1Data_average.size)
        plt.errorbar(v1Data_average[:,0], v1Data_average[:,1],fmt='.',yerr=v1Data_stdDev[:,1],ecolor="#0B00AB",label="Collins",color="blue",markersize=10,capsize=5)
    
    # plt.scatter(v2Data_average[:,0], v2Data_average[:,1], label="BUDD")
    if v2Data.size > 0:
        plt.errorbar(v2Data_average[:,0], v2Data_average[:,1],fmt='.',yerr=v2Data_stdDev[:,1],ecolor="#BD6800",label="BUDD",color="orange",markersize=10,capsize=5)
    for num in range(len(model_splits)):
        split = model_splits[num]
        if num == 0:
            plt.axvline(x=split, color='gray', label="Model Split")
        else:
            plt.axvline(x=split, color='gray')
    plt.xlabel("Number of Actions Taken")
    plt.ylabel("Error")
    plt.title("Model Error vs. Number of Actions Taken For Environment " + env_num)
    plt.legend()

    axes = plt.gca()
    if useFirstPoint:
        axes.set_ylim([0,1])  # make it so that the y axis starts at zero and goes to 1

    plt.show()


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
        v3Data = np.array(v3Data)
        v1Data_average = np.mean(v1Data, axis=0)
        v2Data_average = np.mean(v2Data, axis=0)
        v3Data_average = np.mean(v3Data, axis=0)
        v1Data_stdDev = np.std(v1Data, axis=0)
        v2Data_stdDev = np.std(v2Data, axis=0)
        v3Data_stdDev = np.std(v3Data, axis=0)

        xData = np.concatenate((v3Data[:,0], v1Data[:,0], v2Data[:,0]))
        yData = np.concatenate((v3Data[:,1], v1Data[:,1], v2Data[:,1]))
        groupings = np.concatenate((np.full(np.shape(v3Data[:,0]), "Collins"), (np.full(np.shape(v1Data[:,0]), "BUDD W/out Control")), np.full(np.shape(v2Data[:,0]), "BUDD W/ Control")))
        
        plt.figure(figure_num)
        figure_num = figure_num + 1
        sns.barplot(x=xData, y=yData, hue=groupings, capsize=0.1) 


        plt.xlabel("Model Split Number")
        plt.ylabel("Number of Actions Taken")
        plt.title("Number of Actions Taken per Model Split For Environment " + env_num)
        plt.legend()

    plt.show()


def generateGraphRelativeError(useFirstPoint):
    # use list and not numpy array since we don't know how many iterations were done
    versionNum = 1
    model_splits = []
    files = glob.glob("./Testing Data/relativeError/*.csv")
    firstFile = files[0]
    data = []
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
                    if iteration_num == 0 and useFirstPoint is False:
                            continue
                    trialData.append([iteration_num + offset_amount, float(row['1'])])
            data.append(trialData)

    v1Data = data
    v1Data = np.array(v1Data)
    v1Data_average = np.mean(v1Data, axis=0)
    v1Data_stdDev = np.std(v1Data, axis=0)


    # plt.scatter(v1Data_average[:,0], v1Data_average[:,1])
    plt.errorbar(v1Data_average[:,0], v1Data_average[:,1],fmt='.',yerr=v1Data_stdDev[:,1],ecolor="#0B00AB",color="blue",markersize=10,capsize=5)
    for num in range(len(model_splits)):
        split = model_splits[num]
        if num == 0:
            plt.axvline(x=split, color='gray', label="Model Split")
        else:
            plt.axvline(x=split, color='gray')
    plt.xlabel("Number of Actions Taken")
    plt.ylabel("Relative Error")
    plt.title("Relative Model Error vs. Number of Actions Taken For Environment " + "1")
    plt.legend()

    axes = plt.gca()
    if useFirstPoint:
        axes.set_ylim([0,1])  # make it so that the y axis starts at zero and goes to 1

    plt.show()

def generateGraphTest3(useFirstPoint):
    # use list and not numpy array since we don't know how many iterations were done
    v1Data = []
    v2Data = []
    model_splits = []
    
    for versionNum in range(1, 1+3):
        files = glob.glob("./Testing Data/Test3_v" + str(versionNum) + "/*.csv")
        if len(files) == 0:
            continue
        firstFile = files[0]
        env = re.search("env\d+", firstFile).group()
        env_num = env[len("env"):]
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
                        if iteration_num == 0 and useFirstPoint is False:
                            continue
                        trialData.append([iteration_num + offset_amount, float(row['1'])])
                data.append(trialData)

    v1Data = np.array(v1Data)
    if v1Data.size > 0:
        v1Data_average = np.mean(v1Data, axis=0)
        v1Data_stdDev = np.std(v1Data, axis=0)

    v2Data_average = np.mean(v2Data, axis=0)
    if v2Data.size > 0:
        v2Data = np.array(v2Data)
        v2Data_stdDev = np.std(v2Data, axis=0)

    
    # plt.scatter(v1Data_average[:,0], v1Data_average[:,1], label="Collins")
    # plt.scatter(v2Data_average[:,0], v2Data_average[:,1], label="BUDD")
    if v1Data.size > 0:
        plt.errorbar(v1Data_average[:,0], v1Data_average[:,1],fmt='.',yerr=v1Data_stdDev[:,1],ecolor="#0B00AB",label="Collins",color="blue",markersize=10,capsize=5)

    if v2Data.size > 0:
        plt.errorbar(v2Data_average[:,0], v2Data_average[:,1],fmt='.',yerr=v2Data_stdDev[:,1],ecolor="#BD6800",label="BUDD",color="orange",markersize=10,capsize=5)
    for num in range(len(model_splits)):
        split = model_splits[num]
        if num == 0:
            plt.axvline(x=split, color='gray', label="Model Split")
        else:
            plt.axvline(x=split, color='gray')
    plt.xlabel("Number of Actions Taken")
    plt.ylabel("Error")
    plt.title("Model Error vs. Number of Actions Taken For Environment " + env_num)
    plt.legend()

    axes = plt.gca()
    if useFirstPoint:
        axes.set_ylim([0,1])  # make it so that the y axis starts at zero and goes to 1

    plt.show()


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
        # with codecs.open("dotfile.dot", "w", "utf-8") as dotfile:
        with NamedTemporaryFile("wb", delete=False) as dotfile:
            dotfilename = dotfile.name
            for line in lines:
                # dotfile.write("%s\n" % line)
                dotfile.write(("%s\n" % line).encode("utf-8"))
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

if __name__ == "__main__":
    envNum = 42
    envString = "Example"+str(envNum)
    env = locals()[envString]()
    getModelGraph(envNum, env.SDE_Set, env.A_S, env.get_true_transition_probs(), "env" + str(envNum) + "Graph.png")
    # generateGraphTest1(False,False)
    # generateGraphRelativeError()
