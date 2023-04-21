"""

Python code to recognize playing cards in players and 
dealers hands using webcam.

    -ENSC482
    -Summer 2021

By: Kyle Granville, Mohit Sharma, Michael Celio

todo:
-code should recognize cards, still need to find a way to append
  to a list for the dealer's and player's cards




"""

import cv2
from cv2 import data
import imutils
import numpy
import glob
import os




def normalize(frame):
    #resize camera view
    frame = imutils.resize(frame,640)
    #convert to greyscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #add gaussian blur to more easily ignore small useless details, keeps edges clear, reduces noise
    blur = cv2.bilateralFilter(gray,10,15,15)
    #only show edges or colour changes
    edges = cv2.Canny(blur,50,150,True) #50,150
    #show edges picture 
    #cv2.imshow("edges", edges)

    #make edges thicker
    kernel = numpy.ones((3,3),numpy.uint8)
    dilate = cv2.dilate(edges, kernel,iterations=1)
    #show thick edges
    #cv2.imshow("dilated", dilate)
    return dilate
    


def get_contours(dilate):
    contours = cv2.findContours(dilate,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return sorted(contours, key=cv2.contourArea,reverse=True)
    
def crop_image(OGimage, rect):
    ((x, y), (w, h), a) = rect

    # Rotate image so rectangle is in same rotation as frame.
    shape = (OGimage.shape[1], OGimage.shape[0])
    #rot matrix
    matrix = cv2.getRotationMatrix2D(center=(x, y), angle=a, scale=1)
    #rotate image to normal frame
    rimage = cv2.warpAffine(src=OGimage, M=matrix, dsize=shape)

    # Crop the image from rotation
    cx = int(x - w / 2)
    cy = int(y - h / 2)
    return rimage[cy:int(cy + h), cx:int(cx + w)]


def resize_image(image):
    approx = numpy.array([[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]],[image.shape[1], 0]], numpy.float32)
    h = numpy.array([[0, 0], [0, 449], [449, 449], [449, 0]], numpy.float32)
    transform = cv2.getPerspectiveTransform(numpy.float32(approx), h)
    return cv2.warpPerspective(image, transform, (450, 450))



def checkWhiteVal(image,dataset,ranks=[]):
    im2 = image
    guess = image
    cardChoice = 0
    ws=32767
    #print(str(len(dataset)))
    for i,im1 in enumerate(dataset):
        #cv2.imshow("im",image)
        im2=cv2.absdiff(image,im1)
        #cv2.imshow("im1",im1)
        #cv2.imshow("image", image)
        #cv2.imshow("im2",im2)
        ws1 = cv2.countNonZero(im2)

        if ws > ws1:
            ws = ws1
            guess=im1
    #        cv2.imshow("im2",im2)
            cardChoice = ranks[i]
                
   # cv2.imshow("guess=" + str(cardChoice),guess)
            
            
            #newCard = input("input card number:   ")
            #if newCard == "q":
            #    pass
            #else:
            #    cv2.imwrite(newCard + ".jpg", image)
    return cardChoice


def process_card_image(image):
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    idk, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh



def process(frame):
    image = frame.copy()
    dilate = normalize(frame)
    contours = get_contours(dilate)

    for c in contours[:12]:
        ((x,y),(w,h),a) = rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = numpy.int0(box)


   #     difx = []
     #   dify = []
    #    if len(currentCardsx)>0:
     #       for cardx in currentCardsx:
     #           difx.append( abs(x-cardx) )
     #       for cardy in currentCardsy:
     #           dify.append( abs(y-cardy) )
        
      #      if min(difx)<25 and min(dify)<25:
     #           continue

        ar = w/float(h) if w>h>0 else h/float(w)
        if 1.35<=ar<=1.6 and w*h<120000 and w*h>7300:
            cv2.drawContours(image, [box], -1, (0, 0, 255), 3)
            cimage = crop_image(image, rect)
            if cimage.shape[0] > 0 and cimage.shape[1] > 0:
                cimage = process_card_image(cimage)
                cimage = resize_image(cimage)
                blur1 = cv2.bilateralFilter(cimage,10,15,15)
                #only show edges or colour changes
                edges1 = cv2.Canny(blur1,50,150,True)
                kernel1 = numpy.ones((3,3),numpy.uint8)
                dilate1 = cv2.dilate(edges1, kernel1,iterations=1)
                idek = resize_image(blur1)

                cv2.imshow("current image",idek)
                print('just type  "q"  to retake image using next frame')
                cname = input("input card suit + value (eg. 4 of hearts = h4):  ")
                if cname == "q":
                    pass
                else:
                    cv2.imwrite(cname + ".jpg",idek)
                
                #cv2.putText(image, prediction, (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .75, 255, 3)
                #cv2.putText(image, str(f'{percentage * 100:.2f}' + '%'), (int(x - w / 2), int(y + 20)),cv2.FONT_HERSHEY_SIMPLEX, .50, 255, 2)
                
                #cv2.imshow("cimage",cimage)
                #cv2.imshow("idk",image)

    cv2.imshow("Image", numpy.hstack((image, cv2.cvtColor(dilate, cv2.COLOR_RGB2BGR))))
    
    

def main():        
    print(" - getting webcam ready - ")
    cap = cv2.VideoCapture(0)
    print(" - webcam on - ")
    print(" -  running  - ")
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame,640)
        #cv2.imshow("Image",frame)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break

        process(frame)

    cap.release()
    cv2.destroyAllWindows()


main()