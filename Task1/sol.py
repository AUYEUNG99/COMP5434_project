import csv

f = open("data/Train_Data.csv", "r")
inputData = f.readlines()

def map(inputData):
    mapResult = []
    for line in inputData[1:]:
        data = line.split(",")
        mapResult.append(data[3] + '-' + data[4] + '-' +
                         data[18] + '-' + data[19])
    return mapResult

mapResult = map(inputData)


def reduce(mapResult):
    reuslt = []
    for line in mapResult:
        num = line.split("-")
        totalCost = []
        totalCost.append(round(
            (float(num[0]) * float(num[2]) + float(num[1]) * float(num[3])) * 6.784829586))
        reuslt.append(totalCost)
    with open("Task1/totalCost.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["totalCost"])
        writer.writerows(reuslt)

reduce(mapResult)
