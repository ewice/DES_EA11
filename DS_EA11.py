from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import datetime
# AGE: <=30 -> 0 | 31-40 -> 1 | >40 -> 2
# Income: low -> 0 | medium -> 1 | high -> 2
# Student: no -> 0 | yes -> 1
# credit rating: fair -> 0 | excellent -> 1
X = [
    [0, 2, 0, 0],
    [0, 2, 0, 1],
    [1, 2, 0, 0],
    [2, 1, 0, 0],
    [2, 0, 1, 0],
    [2, 0, 1, 1],

    [1, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [2, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 1, 0, 1],
    [1, 2, 1, 0],
    [2, 1, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 1, 0],
    [2, 1, 1, 1],
    [1, 2, 0, 1],
   
]

y = [0,0,1,1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,1,1]
# 0 = kauft nicht | 1 = kauft
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

export_graphviz(
    tree_clf,
    out_file="./tree"+datetime.datetime.now().isoformat()+".dot",
    feature_names= ["age", "income", "student", "credit_rating"],
    class_names=["kauft nicht", "kauft"],
    rounded=True,
    filled=True
 )

 ############################################################
 ##########
 ##########
 ##########  Der Baum fängt genauso an, wie der Baum aus Aufgabe 1. Grund hierfür könnte sein, dass die Klassen nicht richtig codiert wurden. 
 ##########  Da ML Modelle nur mit Zahlen arbeiten können, wurde hier das Label der Klasse (z.B. jünger oder gleich 30) einem Wert zugewiesen (z.B. 0)
 ##########  Dies hätte eventuell mit OneHotEncoder gemacht werden soll. Jedoch überstieg dies meine aktuellen Fähigkeiten.
 ##########
 ##########
############################################################