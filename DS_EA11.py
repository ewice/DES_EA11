from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import datetime

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
    [1, 2, 0, 1]
]

y = [0,0,1,1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,1,1]
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
