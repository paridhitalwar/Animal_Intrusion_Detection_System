from tkinter import *  
from PIL import ImageTk,Image
import cv2
from visualizeCv2 import model, display_instances, class_names
import sys

ws = Tk()
ws.option_add('*Font', '19')
ws.title('Menu')
ws.geometry('650x400')
title = Label(ws, text = "Animal intrusion detection system").place(x = 30,y = 20)

run=1
canvas = Canvas(
        ws, 
        width = 500, 
        height = 500
        )    
canvas.place(x = 30, y = 80)  
img = ImageTk.PhotoImage(Image.open('./download.jpg'))  
canvas.create_image(
        10, 
        10, 
        anchor=NW, 
        image=img
        )
canvas.place()
def start():
    args = sys.argv
    if(len(args) <2):
        print("run command: python video_demo.py 0 or video file name")
        sys.exit(0)
    name = args[1]
    if(len(args[1]) == 1):
        name = int(args[1])
        name = int (args[1])
    stream = cv2.VideoCapture(name)
    while run==1:
        ret, frame = stream.read()
        if not ret:
            print("unable to fetch frame")
            break
        results = model.detect([frame], verbose=1)

        r= results[0]
        masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        cv2.imshow("masked_image",masked_image)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    stream.release()
    cv2.destroyWindow("masked_image")    

def stop():
    sys.exit()

b1 = Button(ws, text = "Start",background="lightblue",activeforeground = "white",activebackground = "blue",pady=5,padx=20, command=start).place(x=500,y=100)
b3 = Button(ws, text = "Quit",background="lightblue",activeforeground = "white",activebackground = "blue",pady=5,padx=20, command=stop).place(x=500,y=200)
ws.mainloop() 

