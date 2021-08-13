import json

class mainCar:
    def __init__(self,EngineModel,Milage,ModelName):
     self.EngineModel = EngineModel
     self.Milage=Milage
     self.ModelName=ModelName

    def printDetails(a):
        print("Basic Common Model has Milage of {a1} and Model Name is {b} with Engine Model Number as{c}".format(a1=a.Milage,b=a.ModelName,c=a.EngineModel))   

class SubCars(mainCar):
       
    def DisplayAdditionalFeatures(abc):
       Tiers = 4
       Nitors ="No"
       print("This is an High End car which contains the these additional Features of tires {a} and {b} Nitros will be provided with this Model".format(a=Tiers,b=Nitors))


highendcar = SubCars(123,150,"Basic Audi")
highendcar.printDetails()
highendcar.DisplayAdditionalFeatures()
print(dir(highendcar))


# some JSON:
x =  '{ "name":"John", "age":30, "city":"New York"}'

# parse x:
y = json.loads(x)

# the result is a Python dictionary:
print(y["age"])
newList = ["a",21.2,True,"a"]
newTutple = (1,"a",4.5,True,1)
newSets ={2,"at",4.52,True,2}
newDictionary={"brand":1,"tesat":45,"tesat":45}

for x in newList:
 print(x)

for x1 in newTutple:
 print(x1)

for x2 in newSets:
 print(x2)

for x3 in newDictionary:
 print(x3)

#print(newList)
#print(newTutple)
#print(newSets)
#print(newDictionary)



