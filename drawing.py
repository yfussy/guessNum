from tkinter import *
from PIL import Image, ImageDraw, ImageShow
import os

main_directory = os.path.dirname(os.path.realpath(__file__))
directory = main_directory + '\\temp\\test.png'
my_file_extension = ".png"  # PIL image can be saved as .png .jpg .gif or .bmp file (among others)

width = 168
height = 168
center = height//2
white = (255, 255, 255)
black = (0,0,0)
image1 = None


def paint(event):
    """on event, draw oval onto Canvas (both)"""
    x1, y1 = (event.x + 1), (event.y + 1)
    x2, y2 = (event.x - 1), (event.y - 1)
    cv.create_oval(x1, y1, x2, y2, fill="white", width=0)  # On tkinter Canvas
    draw.line([x1, y1, x2, y2], fill="white",  width=4)  # On PIL Canvas
    # "fill" is the inside color. "Width" is the width of the outline (which is black by default.
    # if we set "width" to 5 the outline would cover over the fill, and we would only see black.

def next():
    """save canvas image as filename"""
    print("'next' button pressed")
    image1.save(directory, 'PNG')
    root.destroy()

def clear ():
    """clear the canvas"""
    # Clear the SEEN canvas
    cv.delete(ALL)
    # Clear the UNSEEN canvas
    """Draws a white rectangle over the entire canvas. """
    w = image1.width
    h = image1.height
    draw.rectangle([0, 0, w, h], fill="white", width=0)  # On PIL Canvas

# def close():
#     root.destroy()

# def show ():
#     print("'show' button pressed")
#     image1.show()

root = Tk()

# Tkinter create a canvas to draw on
# This canvas can be seen.
cv = Canvas(root, width=width, height=height, bg='black')
cv.pack()

# PIL create an empty image and draw object to draw on
# This canvas can NOT be seen. It is in the memory only.
image1 = Image.new("RGB", (width, height), black)
draw = ImageDraw.Draw(image1)



cv.bind("<B1-Motion>", paint)
        
button=Button(text="next", command=next)
button.pack()
button=Button(text="clear", command=clear)
button.pack()
# button=Button(text="show", command=show)
# button.pack()
# button=Button(text="close", command=close)
# button.pack()

root.mainloop()