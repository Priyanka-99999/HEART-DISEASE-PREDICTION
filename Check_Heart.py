import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from reportlab.pdfgen import canvas
import os
import sys
import tempfile

# Load dataset
dataset = pd.read_csv('new.csv')

def Train():
    root = tk.Toplevel()
    root.title("Heart Disease Detection")

    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (w, h))

    try:
        image2 = Image.open('images/14.jpg')
        image2 = image2.resize((w, h), Image.LANCZOS)
        background_image = ImageTk.PhotoImage(image2)
        background_label = tk.Label(root, image=background_image)
        background_label.image = background_image
        background_label.place(x=0, y=0)
    except Exception as e:
        root.configure(background='light blue')
        print(f"Image loading error: {str(e)}")

    lbl = tk.Label(root, text="Heart Disease Detection", font=('times', 35, 'bold'), height=1, width=30, bg="#006400", fg="white")
    lbl.place(x=350, y=15)

    X = dataset.drop(columns=['target'])
    y = dataset['target'].apply(lambda x: 1 if x == 1 else 0)
    X = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=6)

    model = RandomForestClassifier(n_estimators=100, random_state=6)
    model.fit(X_train, y_train)

    column_names = X.columns

    # Create a frame to center the field frame
    center_frame = tk.Frame(root, bg="white")
    center_frame.place(relx=0.5, rely=0.4, anchor="center")

    field_frame = tk.Frame(center_frame, bg="white", bd=2, relief="groove")
    field_frame.pack(pady=20, padx=20)

    # Style configuration
    label_font = ('Arial', 12, 'bold')
    entry_font = ('Arial', 12)
    bg_color = "#f0f8ff"  # AliceBlue
    field_bg = "#e6e6fa"   # Lavender
    entry_bg = "white"

    entries = {}
    fields = [
        ("Age", "50"), ("Sex (1=male, 0=female)", "1"), ("Chest Pain Type (0-3)", "0"),
        ("Resting Blood Pressure", "120"), ("Serum Cholesterol", "200"),
        ("Fasting Blood Sugar (1=true, 0=false)", "0"), ("Resting ECG (0-2)", "1"),
        ("Max Heart Rate", "150"), ("Exercise Induced Angina (1=yes, 0=no)", "0"),
        ("ST Depression", "0"), ("Slope of ST Segment (0-2)", "0"),
        ("Number of Major Vessels (0-4)", "0"), ("Thalassemia (0-3)", "0")
    ]

    for i, (label, default) in enumerate(fields):
        row = i % 7
        col = (i // 7) * 2
        
        # Create label with styling
        lbl = tk.Label(field_frame, text=label+":", font=label_font, bg=field_bg, padx=5, pady=5)
        lbl.grid(row=row, column=col, sticky="ew", padx=5, pady=5)
        
        # Create entry with styling
        var = tk.StringVar(value=default)
        entry = tk.Entry(field_frame, textvariable=var, font=entry_font, width=15, 
                         bg=entry_bg, bd=2, relief="sunken")
        entry.grid(row=row, column=col+1, padx=5, pady=5, sticky="ew")
        entries[label] = var

    # Configure grid weights to make columns expandable
    for i in range(4):
        field_frame.columnconfigure(i, weight=1)
    for i in range(7):
        field_frame.rowconfigure(i, weight=1)

    result_var = tk.StringVar()
    result_var.set("Enter values and click Predict")
    result_label = tk.Label(root, textvariable=result_var, font=('times', 20, 'bold'), bg="light yellow", width=50, height=2)
    result_label.place(relx=0.5, rely=0.75, anchor="center")  # Adjusted position

    def open_pdf(filepath):
        try:
            if sys.platform.startswith('darwin'):
                os.system(f'open "{filepath}"')
            elif os.name == 'nt':
                os.startfile(filepath)
            elif os.name == 'posix':
                os.system(f'xdg-open "{filepath}"')
        except Exception as e:
            messagebox.showerror("Error", f"Could not open PDF: {str(e)}")

    def generate_pdf(result_text, user_data):
        try:
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            filename = os.path.join(temp_dir, "heart_disease_report.pdf")
            
            c = canvas.Canvas(filename)
            c.setFont("Helvetica-Bold", 20)
            c.drawString(100, 770, "Heart Disease Prediction Report")

            c.setFont("Helvetica", 14)
            y_position = 730
            for field, value in user_data.items():
                c.drawString(100, y_position, f"{field}: {value}")
                y_position -= 20

            c.setFont("Helvetica-Bold", 16)
            c.drawString(100, y_position - 20, f"Prediction Result: {result_text}")
            c.save()
            
            return filename
        except Exception as e:
            print(f"PDF generation error: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate PDF: {str(e)}")
            return None

    def download_pdf(filename):
        if not filename or not os.path.exists(filename):
            messagebox.showerror("Error", "PDF file not found")
            return
            
        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile="heart_disease_report.pdf"
        )
        
        if save_path:
            try:
                import shutil
                shutil.copy(filename, save_path)
                messagebox.showinfo("Success", f"PDF saved to: {save_path}")
                open_pdf(save_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save PDF: {str(e)}")

    def predict_disease():
        try:
            user_data = {
                'age': float(entries["Age"].get()),
                'sex': float(entries["Sex (1=male, 0=female)"].get()),
                'cp': float(entries["Chest Pain Type (0-3)"].get()),
                'trestbps': float(entries["Resting Blood Pressure"].get()),
                'chol': float(entries["Serum Cholesterol"].get()),
                'fbs': float(entries["Fasting Blood Sugar (1=true, 0=false)"].get()),
                'restecg': float(entries["Resting ECG (0-2)"].get()),
                'thalach': float(entries["Max Heart Rate"].get()),
                'exang': float(entries["Exercise Induced Angina (1=yes, 0=no)"].get()),
                'oldpeak': float(entries["ST Depression"].get()),
                'slope': float(entries["Slope of ST Segment (0-2)"].get()),
                'ca': float(entries["Number of Major Vessels (0-4)"].get()),
                'thal': float(entries["Thalassemia (0-3)"].get())
            }

            input_data = pd.DataFrame([user_data])
            input_data_encoded = pd.get_dummies(input_data, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])

            for col in column_names:
                if col not in input_data_encoded.columns:
                    input_data_encoded[col] = 0

            input_data_encoded = input_data_encoded[column_names]
            input_scaled = scaler.transform(input_data_encoded)
            prediction = model.predict(input_scaled)

            if prediction[0] == 1:
                result = "No Heart Disease Detected"
                result_label.config(bg="green", fg="white")
            else:
                result = "Heart Disease Detected"
                result_label.config(bg="red", fg="white")

            result_var.set(f"Result: {result}")
            
            # Generate PDF and show download button
            filename = generate_pdf(result, user_data)
            if filename:
                download_button = tk.Button(
                    root, 
                    text='Download PDF', 
                    command=lambda: download_pdf(filename),
                    width=15, 
                    height=2,
                    font=('times', 15, 'bold'), 
                    bg='orange', 
                    fg='white'
                )
                download_button.place(relx=0.5, rely=0.85, anchor="center")  # Adjusted position

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            result_var.set("Please enter valid data for all fields")
            result_label.config(bg="yellow")

    # Create a frame for buttons to center them
    button_frame = tk.Frame(root, bg="white")
    button_frame.place(relx=0.5, rely=0.65, anchor="center")  # Moved up from 0.9 to 0.65

    predict_button = tk.Button(button_frame, text="Predict", command=predict_disease, width=15, height=2,
                              font=('times', 15, 'bold'), bg="blue", fg="white")
    predict_button.pack(side="left", padx=20)

    exit_button = tk.Button(button_frame, text="Close", command=root.destroy, width=15, height=2,
                           font=('times', 15, 'bold'), bg="red", fg="white")
    exit_button.pack(side="left", padx=20)

    root.mainloop()