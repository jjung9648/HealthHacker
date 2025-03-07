import pandas
import sklearn.preprocessing as sk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score

# get DataFrame from Sleep_health_and_lifestyle_dataset.csv in /data
def getDataFrame():
    # load csv file(DataFrame)
    df = pandas.read_csv("../data/Sleep_health_and_lifestyle_dataset.csv")

    # print first five data from df
    return df

# set X(input value) = Gender, Age, Occupation, Blood Pressure, Heart Rate, Sleep Disorder, Sleep Duration, Physical Activity Level, Stress Level, BMI Category, Daily Steps
#  y(output value to expect) = Quality of Sleep
def setXandY(df):
    X = df[["Gender", "Age", "Occupation", "Blood Pressure", "Heart Rate", "Sleep Disorder", "Sleep Duration", "Physical Activity Level", "Stress Level", "BMI Category", "Daily Steps"]]
    y = df[["Quality of Sleep"]]
    return [X, y]

# encode labels that are text
def encodeLabels(Xandy:list):
    encoderGender = sk.LabelEncoder()
    encoderOccupation = sk.LabelEncoder()
    encoderSleepDisorder = sk.LabelEncoder()
    encoderBMI = sk.LabelEncoder()
    encoderBlood = sk.LabelEncoder()
    Xandy[0]["Gender"] = encoderGender.fit_transform(Xandy[0]["Gender"])
    Xandy[0]["Occupation"] = encoderOccupation.fit_transform(Xandy[0]["Occupation"])
    Xandy[0]["Sleep Disorder"] = encoderSleepDisorder.fit_transform(Xandy[0]["Sleep Disorder"])
    Xandy[0]["BMI Category"] = encoderBMI.fit_transform(Xandy[0]["BMI Category"])
    Xandy[0]["Blood Pressure"] = encoderBlood.fit_transform(Xandy[0]["Blood Pressure"])
    joblib.dump(encoderGender, "../model/label_encoderGender.pkl")
    joblib.dump(encoderOccupation, "../model/label_encoderOccupation.pkl")
    joblib.dump(encoderSleepDisorder, "../model/label_encoderSleepDisorder.pkl")
    joblib.dump(encoderBMI, "../model/label_encoderBMI.pkl")
    joblib.dump(encoderBlood, "../model/label_encoderBlood.pkl")
    return Xandy

# split data for train(80%) and test(20%)
def splitData(Xandy:list):
    X_train, X_test, y_train, y_test = train_test_split(Xandy[0], Xandy[1], test_size=0.2, random_state=42)
    joblib.dump([X_train, X_test, y_train, y_test], "../data/split_data.pkl")
    return [X_train, X_test, y_train, y_test]

def trainModel(model):
    inputAndOutput = setXandY(getDataFrame())
    encodedIO = encodeLabels(inputAndOutput)
    splitedIO = splitData(encodedIO)
    model.fit(splitedIO[0], splitedIO[2]) # train the Linear Regression model
    print("Training Completed!")


def testModelUser(model,userGender,userAge,userOcupation,userSleepDisorder,userBMICategory,userPhysicalActivityLevel,userBloodPressure,userHeartRate,userSleepDuration,userStressLevel,userDailySteps):
    encoderGender = joblib.load("../model/label_encoderGender.pkl")
    encoderOccupation = joblib.load("../model/label_encoderOccupation.pkl")
    encoderSleepDisorder = joblib.load("../model/label_encoderSleepDisorder.pkl")
    encoderBMI = joblib.load("../model/label_encoderBMI.pkl")
    encoderBlood = joblib.load("../model/label_encoderBlood.pkl")
    userDf = pandas.DataFrame({
        "Gender": [encoderGender.transform([userGender])[0]], 
        "Age": [userAge], 
        "Occupation": [encoderOccupation.transform([userOcupation])[0]], 
        "Blood Pressure": [encoderBlood.transform([userBloodPressure])[0]], 
        "Heart Rate": [userHeartRate], 
        "Sleep Disorder": [encoderSleepDisorder.transform([userSleepDisorder])[0]],
        "Sleep Duration": [userSleepDuration],
        "Physical Activity Level": [userPhysicalActivityLevel], 
        "Stress Level": [userStressLevel], 
        "BMI Category": [encoderBMI.transform([userBMICategory])[0]], 
        "Daily Steps": [userDailySteps]
    })
    user_pred = model.predict(userDf)
    print(user_pred)

def testModel(model):
    X_train, X_test, y_train, y_test = joblib.load("../data/split_data.pkl")
    y_pred = model.predict(X_test)
    # Evaluate the mean absolute error of the model
    mae = mean_absolute_error(y_test, y_pred)
    # Evaluate the R2 score of the model. 1 = perfect, 0 = average, negative = bad
    r2 = r2_score(y_test, y_pred)
    print("mean_absolute_error of the model: {0}\nr2_score of the model(1=perfect,0=average,negative=bad): {1}".format(mae,r2))

if __name__ == "__main__":
    print("This is a prototype of sleep app Linear regression model. Need further implementation of other functionalities.")
    model = None
    # check if model exist
    if os.path.exists("../model/linear_model.pkl") == False:
        model = LinearRegression()
        print("Train the model first.")
    else:
        model = joblib.load("../model/linear_model.pkl") # load model
        print("Existing model loaded.")
    getInput = int(input("1: train the model, 2: test the model, 3: test the model with user input\n"))

    # print((joblib.load("../model/label_encoderSleepDisorder.pkl")).classes_)
    if getInput == 1:
        trainModel(model)
    elif getInput == 2:
        testModel(model)
    else:
        # print("User's Basic info(set it in the database)")
        # userGender = input("Gender(Male, Female): ")
        # userAge = int(input("Age: "))
        # userOcupation = input("Ocupation(ex.Software Engineer, Teacher): ")
        # userSleepDisorder = input("Sleep Disorder(ex. None, Sleep Apnea, Insomnia): ")
        # userWeight = float(input("Weight in kg(ex.70): "))
        # userHeight = float(input("Height in cm(ex.165): "))
        # userBMICategory = ""
        # if userWeight / (userHeight ** 2) < 25:
        #     userBMICategory = "Normal"
        # else:
        #     userBMICategory = "Overweight"
        # userPhysicalActivityLevel = input("Physical Activity Level(30~100): ")

        # print("Get this data every day and average it")
        # userHighBloodPressure = sum([int(input("High Blood Pressure of {0}st day(ex.125): ".format(i+1))) for i in range(7)]) // 7
        # userLowBloodPressure = sum([int(input("Low Blood Pressure of {0}st day(ex.80): ".format(i+1))) for i in range(7)]) // 7
        # userBloodPressure = str(userHighBloodPressure) + "/" + str(userLowBloodPressure)
        # userHeartRate =  sum([int(input("Heart Rate of {0}st day(ex.100): ".format(i+1))) for i in range(7)]) // 7
        # userSleepDuration = sum([int(input("Sleep Duration of {0}st day(ex.7): ".format(i+1))) for i in range(7)]) // 7
        # userStressLevel = sum([int(input("Stress Level of {0}st day(3~8): ".format(i+1))) for i in range(7)]) // 7
        # userDailySteps = sum([int(input("DailySteps of {0}st day(3000~8000): ".format(i+1))) for i in range(7)]) // 7
        # testModelUser(model, userGender,userAge,userOcupation,userSleepDisorder,userBMICategory,userPhysicalActivityLevel,userBloodPressure,userHeartRate,userSleepDuration,userStressLevel,userDailySteps)
        testModelUser(model, "Male",23,"Teacher","Insomnia","Normal",30,"125/80",100,7,4,4000)
    joblib.dump(model, "../model/linear_model.pkl") # save model