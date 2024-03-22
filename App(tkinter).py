from tkinter import *
import tkinter as tk
from Model import *

class App(tk.Tk):
    def __init__(self, model):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # self.overrideredirect(1) # FRAMELESS CANVAS WINDOW
        # self.update() # DISPLAY THE CANVAS
        # Creating elements
        self.canvas = tk.Canvas(self, width=400, height=400, bg = "white", cursor="cross")
        self.canvas.grid(row=0, column=0)#, pady=0, sticky=W, )
        self.canvas.create_rectangle(4, 4, 400, 400)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.btn_classify = tk.Button(self, text = "Recognise", command =self.classify_handwriting) 
        self.btn_classify.grid(row=1, column=1, pady=2, padx=2)
        self.clear_button= tk.Button(self, text = "Clear",command = self.clear_all)
        self.clear_button.grid(row=1, column=0, pady=2)
        self.model = model

    def clear_all(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(4, 4, 400, 400)

    def classify_handwriting(self):
        img = canvas_to_image(self, self.canvas)
        digit, accuracy, prediction = self.model.predict_digit(img)
        self.label.configure(text= ' digit : {} \n accuracy: {:.2f}%'.format(digit, accuracy*100))
    
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

# Creating an instance of the App class and running the Tkinter event loop
if __name__ == "__main__":
    # prepare the model
    model = Model(Model.model_name)
    # model = Model()
    # model = None
    
    app = App(model)
    app.mainloop()