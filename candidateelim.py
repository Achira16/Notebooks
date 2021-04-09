import pandas as pd
import numpy as np
class candidateElimination:
    def __init__(self,no_of_attr):
        self.S = [[None]*no_of_attr]
        self.G = [['?']*no_of_attr]
    def isConsistent(self,X,checkG=True):
        if checkG:
            to_remove = []
            for g in self.G:
                if None in g:
                    to_remove.append(g)
                else:
                    for i in range(len(g)):
                        if g[i]!='?' and g[i]!=X[i]:
                            to_remove.append(g)
            for g in to_remove:
                try:
                    self.G.remove(g)
                except:
                    pass
        else:
            to_remove = []
            for s in self.S:
                flag = False
                if None in s:
                    pass
                else:
                    for i in range(len(s)):
                        if s[i]!='?' and s[i]!=X[i]:
                            flag = True
                    if not flag:
                        to_remove.append(s)
            for s in to_remove:
                try:
                    self.S.remove(s)
                except:
                    pass
    def checkMoreGeneral(self,s,g):
        if None in g and None not in s:
            return False
        if None in s and None not in g:
            return True
        for i in range(len(s)):
            if g[i]!='?':
                if s[i] == '?':
                    return False
                elif s[i]!=g[i]:
                    return False
        return True
    def satisfy(self,s,x):
        if None in s:
            s = x.tolist()
        else:
            for i in range(len(s)):
                if s[i]!='?' and s[i]!=x[i]:
                        s[i] = '?'
        return s
    def consistent(self,g,x):
        if None in g:
                return True
        flag = False
        for i in range(len(g)):
            if g[i]!='?' and g[i]!=x[i]:
                flag = True
        return flag
    def fit(self,features,target):
        print("Progress...")
        categories = []
        for i in range(features.shape[1]):
            categories.append(np.unique(features[:,i]))
        example = 1
        for x,y in zip(features,target):
            if y:
                self.isConsistent(x)
                for j in range(len(self.S)):
                    self.S[j] = self.satisfy(self.S[j],x)
                s_remove = []
                for s in self.S:
                    for g in self.G:
                         if not self.checkMoreGeneral(s,g):
                             s_remove.append(s)
                for s in s_remove:
                    try:
                        self.S.remove(s)
                    except:
                        pass
            else:
                self.isConsistent(x,False)
                to_remove = []
                for j in range(len(self.G)):
                    flag = self.consistent(self.G[j],x)
                    g = self.G[j]
                    if not flag:
                        results = []
                        to_remove.append(j)
                        for i in range(len(g)):
                            for val in categories[i]:
                                if g[i]=='?':
                                    g_new = g[:i]+[val]+g[i+1:]
                                    check = self.consistent(g_new,x)
                                    if check:
                                        results.append(g_new)
                                else:
                                    g_new = g[:i]+[None]+g[i+1:]
                                    check = self.consistent(g_new,x)
                                    if check:
                                        results.append(g_new)
                        self.G.extend(results)
                for ind in to_remove:
                    try:
                        self.G.remove(self.G[ind])
                    except:
                        pass
                to_remove.clear()
                for s in self.S:
                    for g in self.G:
                         if not self.checkMoreGeneral(s,g):
                             to_remove.append(g)
                for g in to_remove:
                    try:
                        self.G.remove(g)
                    except:
                        pass
            print(f'Example {example}: S: {self.S}, G:{self.G}')
            example+=1
        print('Done..')
        return (self.S,self.G)
data = pd.read_csv('ml/collegestuff/ConceptLearning/common.csv')
data.EnjoySport = data.EnjoySport.apply(lambda x:1 if x=="Yes" else 0)
print(data)
model = candidateElimination(data.shape[1]-1)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
S,G = model.fit(X,y)
print(f'S: {S} , G: {G}')
