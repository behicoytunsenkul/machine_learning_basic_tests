import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
#BURADA LOGISTIC REGRESYON SINIFLANDIRMASI YAPACAĞIZ
#Denetimli Öğrenme: Öğretmen (etiketli veri) var → Sınıflandırma veya tahmin.
#veri olusturma:
X = np.random.rand(100,2) #100 veri notkası ve 2 özellik
y = (X[:,0] +X[:,1]>1).astype(int) #toplamı 1den büyükse 1 aksi ise 0
print(X)
print(y)
#veriyi eğitim ve test kümelerine ayırma:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Model oluşturma ve eğitimi
model = LogisticRegression()
model.fit(X_train, y_train)

#tahmin ve doğrulup hesaplama
y_pred = model.predict(X_test)
accuracy= accuracy_score(y_test,y_pred)
print("Accuracy (Model Doğruluğu): ", accuracy)