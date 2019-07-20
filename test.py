from tkinter import *
from tkinter import ttk

root = Tk()
root.title("Playing with Scales")
root.config(bg="#f7a173")
mainframe = ttk.Frame(root, padding="24 24 24 24")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

slider = IntVar()

ttk.Scale(mainframe, from_=0, to_=100, length=300,  variable=slider).grid(column=1, row=4, columnspan=5)
ttk.Label(mainframe, textvariable=slider).grid(column=1, row=0, columnspan=5)

for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)


root.mainloop()