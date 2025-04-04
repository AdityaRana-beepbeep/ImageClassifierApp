import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
import joblib

app = tk.Tk()
app.geometry("700x500")
app.title("Image Classification")

mymodel = joblib.load(r"C:\Users\adity\OneDrive\Desktop\ImageClassifierApp\Fash.joblib")

def classifyimg(imgpath):
    img = Image.open(imgpath).convert("L")
    imgresize = img.resize((28, 28))
    img_array = np.array(imgresize)
    img_flatten = img_array.flatten()
    img_final = img_flatten.reshape(1, -1)
    xyz = mymodel.predict(img_final)
    print("Prediction:", xyz[0])  # Debug line
    return xyz[0]
def uploadimgxyz():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        imgn = ImageTk.PhotoImage(img)
        lblimg.config(image=imgn)
        lblimg.image = imgn
        predict = classifyimg(file_path)
        lblrecit.config(text=f"The image class is: {predict}")

lbl = tk.Label(app, text="Image Classification", font=("Arial", 30))
lbl.pack()

lblimg = tk.Label(app)
lblimg.pack()

lblrecit = tk.Label(app, text=" ", font=("Arial", 20))
lblrecit.pack(pady=10)  # 🠔 This will now appear right under the image

uploadimg = tk.Button(app, text="Upload Image", font=("Arial", 25), command=uploadimgxyz)
uploadimg.pack(pady=20)

app.mainloop()
