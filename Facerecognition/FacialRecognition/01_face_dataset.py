''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    

'''

import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # ปรับความกว้าง
cam.set(4, 480) # ปรับความสูง

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ป้อนรหัสของใบหน้าด้วยตัวเลขแต่ละคน
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# เริ่มนับตัวอย่างของใบหน้า
count = 0

while(True):

    ret, img = cam.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # บันทึกข้อมูลลงในโฟลเดอร์ชุดข้อมูล
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # กด 'ESC' เพื่อยกเลิก
    if k == 27:
        break
    elif count >= 30: # บันทึกตัวอย่าง30ใบหน้าแล้วหยุด
         break

# ทำการล้างข้อมูล
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


