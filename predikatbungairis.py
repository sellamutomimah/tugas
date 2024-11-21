import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# memuat dataset bunga iris
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

# memprediksi spesialis bunga iris
def predict_species():
    try:
        sepal_length = float(entry_sepal_length.get())
        sepal_width = float(entry_sepal_width.get())
        petal_length = float(entry_petal_length.get())
        petal_width = float(entry_petal_width.get())
        
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        species = target_names[prediction]
        messagebox.showinfo("Prediction Result", f"The predicted species is: {species}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

# GUI
root = tk.Tk()
root.title("predikat spesies bunga")

# membuat label dan kotak input di dalam gui untuk memasukkan panjang dan lebar
tk.Label(root, text="panjang kelopak(cm):").grid(row=0, column=0, padx=10, pady=5)
entry_sepal_length = tk.Entry(root)
entry_sepal_length.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="lebar kelopak(cm):").grid(row=1, column=0, padx=10, pady=5)
entry_sepal_width = tk.Entry(root)
entry_sepal_width.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="panjang mahkota(cm):").grid(row=2, column=0, padx=10, pady=5)
entry_petal_length = tk.Entry(root)
entry_petal_length.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="lebar mahkota(cm):").grid(row=3, column=0, padx=10, pady=5)
entry_petal_width = tk.Entry(root)
entry_petal_width.grid(row=3, column=1, padx=10, pady=5)

# membuat Button / tombol menggunakan tkinter
predict_button = tk.Button(root, text="Predict", command=predict_species)
predict_button.grid(row=4, column=0, columnspan=2, pady=10)

# menjalankan event loop dari gui menggunakan tkinter
root.mainloop()
