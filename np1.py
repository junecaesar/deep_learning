import numpy as np
A = np.array([[1,2],[3,4]])
B = np.array([10,20])
result=A*B
print(result)


class Dog:
    def __init__(self,petname,temp):
        self.name=petname;
        self.temprature=temp;

    def status(self):
        print("Dog name is:",self.name)
        print("Dog temprature is:",self.temprature)

    def setTemprature(self,temp):
        self.temprature=temp;

    def bark(self):
        print("Woof!")

print("\nAAA")
lassie=Dog("Lassie",37)
print(lassie.status(),"\nThe new dog is:")

print("\nBBB")
print("\nNow change the temprature")
lassie.setTemprature(50)
print("Now the temprature has been changed to:",lassie.status())