import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split

veriler = pd.read_csv('pulsar_stars.csv')

X= veriler.iloc[:,0:8].values
Y= veriler.iloc[:,8].values

x_denek, x_test,y_denek,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

sınıflandırma = Sequential()

sınıflandırma.add(Dense(4, init = 'uniform', activation = 'relu' , input_dim = 8))
sınıflandırma.add(Dense(1, init = 'uniform', activation = 'sigmoid'))
sınıflandırma.compile(optimizer = Nadam(lr=0.0005), loss =  'binary_crossentropy' , metrics = ['accuracy'] )
history=sınıflandırma.fit(x_denek, y_denek,validation_data=(x_test,y_test), epochs=50)





y_tahmin = sınıflandırma.predict(x_test)
y_tahmin = (y_tahmin > 0.5)

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC eğrisi')
    plt.legend()
    plt.show()
    


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Doğrluk")
plt.ylabel("acu")
plt.xlabel("epoch")
plt.legend(['Train', 'Test'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Hata")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train', 'Test'])
plt.show()


tahmin = sınıflandırma.predict_proba(x_test)
tahmin = y_tahmin[:, 0]
auc = roc_auc_score(y_test, tahmin)
fpr, tpr, thresholds = roc_curve(y_test, tahmin)
plot_roc_curve(fpr, tpr)

fpr, tpr, threshold = metrics.roc_curve(y_test, tahmin)
roc_auc = metrics.auc(fpr, tpr)

print("-------------------------------------------------------------------------------------")
print('AUC = %0.2f' % auc)
print("-------------------------------------------------------------------------------------")
from sklearn.metrics import confusion_matrix
ConfusionMatrix = confusion_matrix(y_test,y_tahmin)
print("Confusion Matrix")
print(ConfusionMatrix)
print("-------------------------------------------------------------------------------------")




