import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib
matplotlib.use('TkAgg')  # Ensure Tkinter-compatible backend


# Load dataset
dataset = pd.read_csv('new.csv')

# GUI setup
root = tk.Tk()
root.title("Heart Disease Detection System")
root.configure(bg="#f5f6f5")

# Set window size (responsive)
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

# Load and set background image
try:
    image2 = Image.open('images/14.jpg')
    image2 = image2.resize((w, h), Image.LANCZOS)
    background_image = ImageTk.PhotoImage(image2)
    background_label = tk.Label(root, image=background_image)
    background_label.image = background_image
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
except Exception as e:
    print(f"Image loading error: {str(e)}")
    root.configure(bg="#e6f3fa")

# Main content frame
content_frame = tk.Frame(root, bg="white", bd=5)
content_frame.place(relx=0.5, rely=0.5, anchor="center", width=1000, height=600)
content_frame.config(highlightbackground="#d3d3d3", highlightthickness=1)

# Heading
lbl = tk.Label(
    content_frame,
    text="Heart Disease Detection System",
    font=('Helvetica Neue', 28, 'bold'),
    bg="white",
    fg="#2c3e50"
)
lbl.pack(pady=20)

# Function to display confusion matrix


def display_confusion_matrix(cm, title="Confusion Matrix"):
    try:
        # Create a new top-level window
        cm_win = tk.Toplevel()
        cm_win.title(title)

        # Create a figure and axis using matplotlib
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)

        # Draw the heatmap on the axis
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # Render the figure onto Tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=cm_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        print(f"[ERROR] Failed to render confusion matrix: {e}")
        tk.messagebox.showerror("Display Error", f"Unable to show confusion matrix: {e}")

# Function to display classification report
def display_classification_report(report):
    print("Classification Report:\n", report)

# Function to display accuracy
def display_accuracy(accuracy):
    print("Accuracy: {:.2f}%".format(accuracy * 100))

# SVM Model Training and Evaluation
def Model_Training():
    X = dataset.drop(columns=['target'])
    y = dataset['target']
    X = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=6)
    
    svm_classifier = SVC(kernel='linear', random_state=6)
    svm_classifier.fit(X_train, y_train)
    svm_y_pred = svm_classifier.predict(X_test)
    
    svm_accuracy = accuracy_score(y_test, svm_y_pred)
    svm_classification_report = classification_report(y_test, svm_y_pred)
    svm_confusion_matrix = confusion_matrix(y_test, svm_y_pred)
    
    display_classification_report(svm_classification_report)
    display_confusion_matrix(svm_confusion_matrix, title='SVM Confusion Matrix')
    display_accuracy(svm_accuracy)

# Random Forest Model Training and Evaluation
def RF():
    X = dataset.drop(columns=['target'])
    y = dataset['target']
    X = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=6)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=6)
    rf_classifier.fit(X_train, y_train)
    rf_y_pred = rf_classifier.predict(X_test)
    
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    rf_classification_report = classification_report(y_test, rf_y_pred)
    rf_confusion_matrix = confusion_matrix(y_test, rf_y_pred)
    
    display_classification_report(rf_classification_report)
    display_confusion_matrix(rf_confusion_matrix, title='Random Forest Confusion Matrix')
    display_accuracy(rf_accuracy)

# Decision Tree Model Training and Evaluation
def DST():
    le = LabelEncoder()
    data = dataset.dropna()
    data['target'] = le.fit_transform(data['target'])
    data['thal'] = le.fit_transform(data['thal'])
    data['cp'] = le.fit_transform(data['cp'])
    x = data.drop(['target'], axis=1)
    y = data['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
    
    clf_gini = DecisionTreeClassifier(criterion='entropy', random_state=2)
    clf_gini.fit(x_train, y_train)
    y_pred = clf_gini.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_result = classification_report(y_test, y_pred)
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    
    display_classification_report(classification_report_result)
    display_confusion_matrix(confusion_matrix_result, title='Decision Tree Confusion Matrix')
    display_accuracy(accuracy)

# Naive Bayes Model Training and Evaluation
def NB():
    X = dataset.drop(columns=['target'])
    y = dataset['target']
    X = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)
    
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_result = classification_report(y_test, y_pred)
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    
    display_classification_report(classification_report_result)
    display_confusion_matrix(confusion_matrix_result, title='Naive Bayes Confusion Matrix')
    display_accuracy(accuracy)

# ANN Model Training and Evaluation
def ANN_algo():
    try:
        le = LabelEncoder()
        data = dataset.dropna()
        data['target'] = le.fit_transform(data['target'])
        data['thal'] = le.fit_transform(data['thal'])
        data['cp'] = le.fit_transform(data['cp'])
        x = data.drop(['target'], axis=1)
        y = data['target']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(x_train)
        X_test = sc.transform(x_test)
        
        classifier = Sequential()
        classifier.add(Dense(activation="relu", input_dim=13, units=8, kernel_initializer="uniform"))
        classifier.add(Dense(activation="relu", units=14, kernel_initializer="uniform"))
        classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
        classifier.add(Dropout(0.2))
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        classifier.fit(X_train, y_train, batch_size=8, epochs=100, verbose=1)
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        classification_report_result = classification_report(y_test, y_pred)
        confusion_matrix_result = confusion_matrix(y_test, y_pred)
        
        display_classification_report(classification_report_result)
        display_confusion_matrix(confusion_matrix_result, title='ANN Confusion Matrix')
        display_accuracy(accuracy)
    except Exception as e:
        print(f"Error in ANN model: {str(e)}")
        messagebox.showerror("Error", f"Failed to process ANN model: {str(e)}")

def call_file():
    import Check_Heart
    Check_Heart.Train()

# Button styling class
class HoverButton(tk.Button):
    def __init__(self, master, **kw):
        tk.Button.__init__(self, master=master, **kw)
        self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
    
    def on_enter(self, e):
        self['background'] = '#3498db'
    
    def on_leave(self, e):
        self['background'] = self.defaultBackground

# Button frame
button_frame = tk.Frame(content_frame, bg="white")
button_frame.pack(pady=20)

# Model buttons
models = [
    ("SVM", Model_Training),
    ("Decision Tree", DST),
    ("Random Forest", RF),
    ("Naive Bayes", NB),
    ("ANN", ANN_algo)
]

for i, (text, command) in enumerate(models):
    btn = HoverButton(
        button_frame,
        text=text,
        command=command,
        width=15,
        height=2,
        font=('Helvetica Neue', 12, 'bold'),
        bg="#2ecc71",
        fg="white",
        activebackground="#27ae60",
        relief="flat",
        borderwidth=0
    )
    btn.grid(row=0, column=i, padx=10, pady=10)

# Disease Detection button
detect_btn = HoverButton(
    content_frame,
    text="Disease Detection",
    command=call_file,
    width=20,
    height=2,
    font=('Helvetica Neue', 14, 'bold'),
    bg="#e74c3c",
    fg="white",
    activebackground="#c0392b",
    relief="flat",
    borderwidth=0
)
detect_btn.pack(pady=10)

# Exit button
exit_btn = HoverButton(
    content_frame,
    text="Exit",
    command=root.destroy,
    width=20,
    height=2,
    font=('Helvetica Neue', 14, 'bold'),
    bg="#95a5a6",
    fg="white",
    activebackground="#7f8c8d",
    relief="flat",
    borderwidth=0
)
exit_btn.pack(pady=10)

# Add a footer
footer = tk.Label(
    content_frame,
    text="Heart Disease Detection System",
    font=('Helvetica Neue', 10),
    bg="white",
    fg="#7f8c8d"
)
footer.pack(side="bottom", pady=10)

root.mainloop()