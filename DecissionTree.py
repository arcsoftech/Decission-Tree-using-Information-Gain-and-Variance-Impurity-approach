"""
Name:Arihant Chhajed
Purpose: Machine Learning Class Assignment I
Decription: Decission Tree using Information Gain and Variance Impurity Heuristic
"""
import pandas as pd
import numpy as np
import math
import sys
from copy import deepcopy
import random

# Optional
# from tabulate import tabulate

random.seed(123)
L = ""
K= ""
toPrint= ""
node_marker = 0
dataset_training = ""
dataset_validation = ""
dataset_testing = ""
leaf_node_count=""

"""
CommandLine Input
"""
if(sys.argv.__len__() != 7):
    sys.exit("""
        Please provide the argument in the format as stated below:-
        <L> <K> <training-dataSet-path> <validation-dataSet-path> <test-dataSet-path> <to-print>
        L: integer (used in the post-pruning algorithm)
        K: integer (used in the post-pruning algorithm)
        to-print:{yes,no}
        """)
else:
    try:
        L = int(sys.argv[1])
        K= int(sys.argv[2])
        trainDataUrl = sys.argv[3]
        testDataUrl = sys.argv[4]
        validationDataUrl = sys.argv[5]
        toPrint= sys.argv[6]
        dataset_training = pd.read_csv(trainDataUrl) 
        dataset_validation = pd.read_csv(testDataUrl)
        dataset_testing = pd.read_csv(validationDataUrl)
    except Exception as ex:
        if(type(ex).__name__ == "ValueError"):
            print("Please enter integer value for L and K")
        else:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print (message)
print("Please note that I am assuming the last column in data as classification column\nTree building process started...")

def entropy(S):
    """
    Entropy Calculator
    """
    total=S.shape[0]
    classes, classCounts = np.unique(S.values[:,-1], return_counts=True)
    label_counts = dict(zip(classes, classCounts))
    try:
        entropySubElements = [-(label_counts[x]/total)*math.log(label_counts[x]/total,2) for x in classes]
        entropy = sum(entropySubElements)
    except Exception:
        return 0.0
    return entropy
      

def informaionGainByEntropy(S,A):
    """
    Information Gain Heuristic
    """
    entropy_S = entropy(S)
    attributes_values = S[A].unique()
    child_sets= [S[S[A] == x] for x in attributes_values]
    count_main_set=S.shape[0]
    entropy_mul_ratio__child_Set=[(x.shape[0]/count_main_set)*entropy(x) for x in child_sets]
    gain = entropy_S - sum(entropy_mul_ratio__child_Set)
    return gain

def varianceImpurity(S):
    """
    Variance Impurity
    """
    total=S.shape[0]
    classes, classCounts = np.unique(S.values[:,-1], return_counts=True)
    label_counts = dict(zip(classes, classCounts))
    probablity_attr_elements = [(label_counts[x]/total) for x in classes]
    variance_impurity = np.prod(np.array(probablity_attr_elements))
    return variance_impurity    

def informaionGainByVarianceImpurity(S,A):
    """
    Information gain by Varian Impurity Heuristic
    """
    VI_S = varianceImpurity(S)
    attributes_values = S[A].unique()
    child_sets= [S[S[A] == x] for x in attributes_values]
    count_main_set=S.shape[0]
    variance_impurity_mul_ratio__child_Set=[(x.shape[0]/count_main_set)*varianceImpurity(x) for x in child_sets]
    gain = VI_S - sum(variance_impurity_mul_ratio__child_Set)
    return gain
  
def traceTree(node_list,node):
    """
    Scan(Trace) Tree
    """
    if(node.nodeType !="R"):
        bar = [x['level'] for x in node_list if x['node'].id == node.parentId][0]
        node_list.append({"level":bar+1,"node":node})
    else:
        bar= 0
        node_list.append({"level":bar,"node":node})

    for x in node.children:
        traceTree(node_list,x)
    return
def countLeafNode(node):
    """
    Leaf Node Counter
    """
    if(node.children.__len__()==0):
        return 1
    return sum(countLeafNode(x) for x in node.children)
def treeStatistic(node):
    """
    Tree Statistics
    """
    node_list=[]
    traceTree(node_list,node)
    no_leaf_node = countLeafNode(node)
    return no_leaf_node,node_list

def findClass(X,node):
    """
    Find label for new data
    """
    if(node.nodeType == 'L'):
        return node.className
    else:
        for x in node.children:
            if(x.value==X[x.name]):
                return findClass(X,x)        
def calculateAccuracy(S,Dt):
    """
    Accuracy Estimator
    """
    correctlyClassified=0
    X = S.values[:,0:-1]
    columns= S.columns[0:-1]
    Y = S.values[:,-1]
    for i in range(X.shape[0]):
        features=dict(zip(columns,X[i,:]))
        predictedValue=findClass(features,Dt.root)
        if(predictedValue == Y[i]):
            correctlyClassified = correctlyClassified + 1
    return (correctlyClassified/Y.shape[0])*100
def bestAttributefinder(S,H):
    """
    Find best attributes
    """
    maxGain = -1.0
    for x in S.columns[0:-1]:
        currentGain = informaionGainByEntropy(S,x) if H == 0 else informaionGainByVarianceImpurity(S,x)
        if maxGain < currentGain:
            maxGain = currentGain
            bestAttribute = x
    return bestAttribute 
def postPrunning(Dt):
    """
    Post Prunning
    """
    global L,K
    Dt_best=Dt
    try:
        for i in range(L-1):
            _Dt = deepcopy(Dt)
            M = random.randrange(1,K)
            for j in range(M-1):
                L,node_list = treeStatistic(_Dt.root)
                no_non_leaf_node = [node['node'] for node in node_list if node['node'].nodeType =='I']
                N= no_non_leaf_node.__len__()-1
                P = random.randrange(0,N)
                tempNode=no_non_leaf_node[P]
                tempNode.children=[]
                tempNode.nodeType='L'
                label_counts = {v: k for k, v in tempNode.targetClassCount.items()}
                tempNode.className = label_counts[max(list(label_counts.keys()))]
            accuracy_Dt = calculateAccuracy(dataset_validation,_Dt)
            accuracy_Dt_best = calculateAccuracy(dataset_validation,Dt_best)
            if(accuracy_Dt > accuracy_Dt_best):
                Dt_best = _Dt
    except Exception:
        return Dt_best
    return Dt_best
 
"""
    Node Data Structure
"""
class Node():
    def __init__(self):
        global node_marker
        self.children = []
        self.nodeType = None
        self.value = None
        self.className=None
        self.name = None
        self.targetClassCount = None
        node_marker = node_marker + 1
        self.id = node_marker
        self.parentId= -1
    
    def setNodeValue(self,nodeType, name = None, value = None):
        self.nodeType = nodeType
        self.value = value
        self.name = name

    def setParentId(self,val):
        self.parentId = val

    def appendChildren(self,child):
        self.children.append(child)

"""
Tree Data Structure
"""
class DecissionTree():
    def __init__(self,heuristic):
        self.root = Node()
        self.heuristic = heuristic
        self.root.setNodeValue("R", "InitialPointer","#")
    
    def createTree(self, S , branch):
        """
        Tree Generator
        """
        classes, classCounts = np.unique(S.values[:,-1], return_counts=True)
        label_counts = dict(zip(classes, classCounts))
        branch.targetClassCount = label_counts
        total=S.shape[0] 
        for x in classes:
            if label_counts[x] == total:
                branch.className=x
                branch.nodeType='L'
                return 
        if(S.shape[1]==1):
            value = max(classCounts)
            inv_map = {v: k for k, v in label_counts.items()}
            branch.className=inv_map[value]
            branch.nodeType='L'
            return
        else:
            Best_Attribute=bestAttributefinder(S,self.heuristic)
            for x in np.sort(S[Best_Attribute].unique()):
                subtree=Node()
                subtree.setNodeValue('I',Best_Attribute,x)
                subtree.setParentId(branch.id)
                branch.appendChildren(subtree)
                new_set=S[S[Best_Attribute] == x].drop([Best_Attribute],axis=1)
                self.createTree(new_set,subtree)
        
    def printTree(self):
        """
        Prints Tree On Console
        """
        node_list=[]
        traceTree(node_list,self.root)
        for x in node_list:
            i= x["level"]
            while i!=0:
                print("| ",end="")
                i=i-1            
            print("{} = {} : {}".format(x["node"].name,x["node"].value,x["node"].className if x["node"].nodeType == "L" else ""))

# Section to generate tree
entropy_based_tree = DecissionTree(0)
entropy_based_tree.createTree(dataset_training , entropy_based_tree.root )
VI_based_tree = DecissionTree(1)
VI_based_tree.createTree(dataset_training , VI_based_tree.root )
entropy_based_tree_prunned=postPrunning(entropy_based_tree)
VI_based_tree_prunned=postPrunning(VI_based_tree)
Accuracy_InformationGain=calculateAccuracy(dataset_testing,entropy_based_tree)
Accuracy_Variance_Impurity=calculateAccuracy(dataset_testing,VI_based_tree)
print("Tree building process completed.Please check the output file in the root directory of this project")
if(toPrint.lower() == 'yes'):
    print("Tree before prunning using information gain based heuristic")
    entropy_based_tree.printTree()
    print("Accuracy:-",calculateAccuracy(dataset_testing,entropy_based_tree))
    print("Tree before prunning using variance impurity based heuristic")
    VI_based_tree.printTree()
    print("Accuracy:-",calculateAccuracy(dataset_testing,VI_based_tree))
    print("Tree after prunning using information gain based heuristic for L={} and K={}".format(L,K))
    entropy_based_tree_prunned.printTree()
    print("Accuracy:-",calculateAccuracy(dataset_testing,entropy_based_tree_prunned))
    print("Tree after prunning using variance impurity based heuristic L={} and K={}".format(L,K))
    VI_based_tree_prunned.printTree()
    print("Accuracy:-",calculateAccuracy(dataset_testing,VI_based_tree_prunned))

f = open("Output.txt",'w')
print("--------Output--------",file=f)
print("Tree before prunning using information gain based heuristic",file=f)
print("Accuracy:-",Accuracy_InformationGain,file=f)
print("Tree before prunning using variance impurity based heuristic",file=f)
print("Accuracy:-",Accuracy_Variance_Impurity,file=f)
print("Tree after prunning using information gain based heuristic",file=f)
print("Accuracy:-",calculateAccuracy(dataset_testing,entropy_based_tree_prunned),file=f)
print("Tree after prunning using variance impurity based heuristic",file=f)
print("Accuracy:-",calculateAccuracy(dataset_testing,VI_based_tree_prunned),file=f)

print("Please use option 'yes' when executing the program to display the tree on console.")

f.close()


# Optional Section For Report Generation
# print("Generating report....")
# f = open("report.txt",'w')
# print("Prunning Output for different combnation of L and k",file=f)
# combinations=[(random.randrange(100,110),random.randrange(100,200)) for i in range(10)]

# Table = []
# for x in combinations:
#     subTable= []
#     L,K = x
#     subTable.append("({},{})".format(L,K))
#     d1=DecissionTree(0)
#     d2=DecissionTree(1)
#     d1 = DecissionTree(0)
#     d1.createTree(dataset_training , d1.root )
#     d2 = DecissionTree(1)
#     d2.createTree(dataset_training , d2.root )
#     Dt1=postPrunning(d1)
#     Dt2=postPrunning(d2)
#     subTable.append(calculateAccuracy(dataset_testing,Dt1))
#     subTable.append(calculateAccuracy(dataset_testing,Dt2))
#     Table.append(subTable)
# print(tabulate(Table, headers=['(L,K)', 'Information Gain Heuristic(Accuracy)','Variance Impurity Heuristic(Accuracy)'], tablefmt='orgtbl'),file=f)
# f.close()

