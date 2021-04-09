import pandas as pd
import numpy as np
class Find_S:
    def __init__(self,no_of_attr):
        self.S = np.array([None]*no_of_attr)
    def satisfy(self,X):
        if None in self.S:
            return False
        for i in range(len(self.S)):
            if self.S[i]!='?' and X[i]!=self.S[i]:
                return False
        return True
    def generalise(self,X):
        if None in self.S:
            self.S = X
        else:
            for i in range(len(self.S)):
                if self.S[i]!='?' and X[i]!=self.S[i]:
                    self.S[i] = '?'
        return self.S
    def fit(self,features,target):
        for x,y in zip(features,target):
            if y:
                issatisfiable = self.satisfy(x)
                if not issatisfiable:
                    self.S = self.generalise(x)
        return self.S
    def predict(self,test_X):
        pred_y = []
        for x in test_X:
            if self.satisfy(x):
                pred_y.append(1)
            else:
                pred_y.append(0)
        pred_y = np.array(pred_y)
        return pred_y
    def accuracy(self,test_y,pred_y):
        return (sum(test_y == pred_y)/len(test_y))*100

enjoysport = pd.read_csv('ml/collegestuff/ConceptLearning/common.csv')
enjoysport.EnjoySport = enjoysport.EnjoySport.apply(lambda x:1 if x == "Yes" else 0)
print(enjoysport)
X = enjoysport.iloc[:,:-1].values
y = enjoysport.iloc[:,-1].values
trial = Find_S(X.shape[1])
res = trial.fit(X,y)
print(f'Final hypothesis:{res}')
car = pd.read_csv('ml/collegestuff/ConceptLearning/car2.csv')
print(car)
car.target.value_counts()
pos = ['acc','good','vgood']
car.target = car.target.apply(lambda x:1 if x in pos else 0)
categ = ['4','more']
car.persons = car.persons.apply(lambda x:'4 or more' if x in categ else '2')
categ = ['med','high']
car.safety = car.safety.apply(lambda x:'good' if x in categ else 'bad')
from sklearn.model_selection import train_test_split
train,test = train_test_split(car,test_size=0.25,random_state=0)
x = train.iloc[:,:-1].values
y = train.iloc[:,-1].values
model = Find_S(x.shape[1])
res = model.fit(x,y)
print(f'Final hypothesis for car dataset:{res}')
test_x = test.iloc[:,:-1].values
test_y = test.iloc[:,-1].values
pred_y = model.predict(test_x)
print(f'Test accuracy for car dataset:{model.accuracy(test_y,pred_y)}')

