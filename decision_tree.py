#Karar Verme Ağacı: Bir soruya evet veya hayır olarak cevap verecek sürec yapısıdır.
#Örnek: Iris Çiçeği türlerini sınıflandırma
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#iris verisetini yukle
data = load_iris()
X , y = data.data , data.target

#veriyi egitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3,random_state=42)

#Karar ağacı modeli olustur ve eğit
model = DecisionTreeClassifier(max_depth=3,random_state=42)
model.fit(X_train,y_train)

#Tahmin yap ve doğruluğu hesapla
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Model Doğruluğu:{accuracy:.2f}")

#karar ağacını gorsellestir
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names,filled=True)
plt.show()