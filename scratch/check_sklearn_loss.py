import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_gbdt_loss():
    try:
        HistGradientBoostingClassifier(loss="log_loss", max_iter=1).fit([[0], [1]], [0, 1])
        return "log_loss"
    except Exception:
        return "auto"

loss = get_gbdt_loss()
print(f"sklearn {sklearn.__version__} selected loss: {loss}")

clf = HistGradientBoostingClassifier(loss=loss, max_iter=5)
clf.fit(np.zeros((10, 2)), np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]))
print("HistGBDT fit succeeded!")

lr = LogisticRegression(C=0.5, max_iter=100, solver="lbfgs", random_state=42)
lr.fit(np.zeros((10, 2)), np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]))
print("LogisticRegression fit succeeded!")
