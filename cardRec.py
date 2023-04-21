"""
       

Python code to recognize playing cards in players and 
dealers hands using webcam.

By: Kyle Granville

note:
-code recognizes cards, need to find a better way to append
  to a list for the dealer's and player's cards

v_8-14-2022
"""

import cv2
from cv2 import data
import imutils
import numpy
import glob
import os
import time
import sys
import socket

numpy.set_printoptions(threshold=sys.maxsize)



pList = []
dList = []
playerCards = []
dealerCards = []
playerSum = 0
dealerSum = 0
printedCards = []

# thresholds for canny and kernel for edges
ctMax=190
ctMin=50
kThicc = (5,5)

# debug variable to show guess image and wait for keypress
# eg. hold space for continuous read
showAndDelay = 0 # 0=off,1=on

#put program into card save-2-file mode
# input 'q' to not save card image, 
#  otherwise input card like:  h3
saveCardsMode = 0 # 0=off,1=on

# set to 0 if using file not socket
# set to 1 if using sockets
socketMode = 0

if socketMode == 1:
    host, port = "127.0.0.1", 25001
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))


def printToFile(dealerHand,playerHand):
    '''Function prints good data from player and dealer
        hands into a file to be read from unity'''
    global sock
    global dList,pList,dealerCards,playerCards,printedCards
    try:
        fp = open("playerHand.txt","a")
        fd = open("dealerHand.txt","a")
    except:
        print("ERROR printing to file, must be currently opened by unity")
        return

    print("now printing cards: ", dealerHand,playerHand)
    if playerHand==[]:
        for card in dealerHand:
            if card != 0 and card not in printedCards:
                fd.write(str(card) + "\n")
                if socketMode ==1:
                    sock.sendall(card.encode("UTF-8"))
                printedCards.append(card)
        
    elif dealerHand ==[]:
        for card in playerHand:
            if card != 0 and card not in printedCards:
                fp.write(str(card) + "\n")
                if socketMode ==1:
                    sock.sendall(card.encode("UTF-8"))
                printedCards.append(card)
    fp.close()
    fd.close()


def normalize(frame):
    '''Function processes webcam image for viewing'''
    #resize camera view
    frame = imutils.resize(frame,640)
    #convert to greyscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #add gaussian blur to more easily ignore small useless details, keeps edges clear, reduces noise
    blur = cv2.bilateralFilter(gray,10,15,15)
    #only show edges or colour changes
    edges = cv2.Canny(blur,ctMin,ctMax,True) #50,150
    #show edges picture 
    #cv2.imshow("edges", edges)

    #make edges thicker
    kernel = numpy.ones(kThicc,numpy.uint8)
    dilate = cv2.dilate(edges, kernel,iterations=1)
    #show thick edges
    #cv2.imshow("dilated", dilate)
    return dilate
    

def getContours(dilate):
    '''Gathers sorted list of contours found in processed image'''
    contours = cv2.findContours(dilate,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return sorted(contours, key=cv2.contourArea,reverse=True)


def cropImage(OGimage, rect):
    '''crops image into size from rect'''
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


def resizeCard(image):
    '''resizes card image into correct size to compare'''
    approx = numpy.array([[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]],[image.shape[1], 0]], numpy.float32)
    h = numpy.array([[0, 0], [0, 449], [449, 449], [449, 0]], numpy.float32)
    transform = cv2.getPerspectiveTransform(numpy.float32(approx), h)
    return cv2.warpPerspective(image, transform, (450, 450))


def getAllCards():
    '''returns full dataset of cards to compare with later
        along with ranks in order of dataset'''
    path = os.path.dirname(os.path.abspath(__file__))
    ims=[]
    cardSuits = ["c","d","h","s"]
    cardRanks = ['a','2','3','4','5','6','7','8','9','10','j','q','k']
    im_name = [()]
    print(" - getting cards dataset - ")
    a=glob.glob(path + "\\images\\")
    
    ranks=[]
    for i in cardRanks:
        for s in cardSuits:
            im = str(s + str(i) + ".jpg")
            image = cv2.imread(im)
            #cv2.imshow("idk",image)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            blur = cv2.bilateralFilter(gray,10,15,15)
            #only show edges or colour changes
            edges = cv2.Canny(blur,ctMin,ctMax,True)
            kernel = numpy.ones(kThicc,numpy.uint8)
            dilate = cv2.dilate(edges, kernel,iterations=1)
            rIm = resizeCard(dilate) #dilate
            
            ims.append(rIm)
            i1=i
            if i in 'j':
                i1=11
            elif i in 'q':
                i1=12
            elif i in 'k':
                i1=13
            elif i in 'a':
                i1=1
            ranks.append(s+str(i1))
            #cv2.imshow("idk",rIm)
    print(" - dataset of " + str(len(ims)) +" ready - ")
    return ims,ranks


def checkWhiteVal(image,dataset,ranks=[]):
    '''Function compares imaged card with each member of 
        dataset and returns most likely match, if it exists'''
    #image = image[10:-10,10:-10]
    im2 = image
    guess = image
    gSub = image
    cardChoice = 0
    ws=69000
    #print(str(len(dataset)))
    for q in range(8):
        #if q == 1:
        
        image =cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
        #cv2.imshow("idk",image)
        for i,im1 in enumerate(dataset):
            #im1 = im1[10:-10, 10:-10]
            if q==4:
                im1 =cv2.rotate(im1,cv2.ROTATE_90_CLOCKWISE)
            #cv2.imshow("im",image)
            im2=cv2.absdiff(image,im1)
            #cv2.imshow("im1",im1)
            
            #cv2.imshow("image", image)
            #cv2.imshow("im2",im2)
            #cv2.waitKey(0)
            ws1 = cv2.countNonZero(im2)
            ws2 = cv2.countNonZero(im1)
            ws3 = cv2.countNonZero(image)
            if ws > ws1 and ws2 > ws1 and ws1 < ws3:
                ws = ws1
                guess=im1
                gsub=im2
                #cv2.imshow("im2",im2)
                cardChoice = ranks[i]  
    
    #debug mode
    if showAndDelay != 0:
        if cardChoice != 0:
            print(str(cardChoice) + ',  ws= ' + str(ws))
        try:
            cv2.imshow("subbed=" + str(cardChoice),gsub)
            cv2.waitKey(0)         
        except: pass

    return cardChoice


def procCardIm(image):
    '''converts card image to a thresholded and greyscaled image'''
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    idk, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    return thresh


def checkIntersect(im,c1,c2):
    '''Checks if contours are overlapping,
        returns bool '''
    for c3 in c2[:12]:
        c=[c1,c3]
        blank = numpy.zeros(im.shape[0:2])
        im1 = cv2.drawContours(blank.copy(),c,0,1)
        im2 = cv2.drawContours(blank.copy(),c,1,1)
        intersect = numpy.logical_and(im1,im2)
        if intersect.any():
            return True
        else:
            return False


def process(frame,dataset,ranks):
    '''main function to deal with all webcam processing'''
    image = frame.copy()
    dilate = normalize(frame)
    contours = getContours(dilate)
    global playerCards, playerSum, pList
    global dealerCards, dealerSum, dList

    #only use up to 4 most likely contours to compare
    for c in contours[:4]:
        #get possible cardd location
        ((x,y),(w,h),a) = rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = numpy.int0(box)
        #clearing lists
        if playerCards==[]:
            pList=[]
        if dealerCards==[]:
            dList=[]

        guessCardVal = ""

        ar = w/float(h) if w>h>0 else h/float(w)
        #check ratio and area of current contour in loop
        #check contours are not overlapping another
        if 1.35<=ar<=1.6 and w*h<120000 and w*h>16000 and not checkIntersect(image,c,contours):
            #print(w*h) #area of current contour for debugging
            #show rectangle on webcam view
            cv2.drawContours(image, [box], -1, (0, 0, 255), 3)
            #get card image
            cimage = cropImage(image, rect)
            if cimage.shape[0] > 0 and cimage.shape[1] > 0:
                #cv2.imshow("Original Image",cimage)
                #cv2.waitKey(0)
                cimage = procCardIm(cimage)
                #cv2.imshow("Greyscale Image",cimage)
                #cv2.waitKey(0)
                cimage = resizeCard(cimage)
                blur1 = cv2.bilateralFilter(cimage,10,15,15)
                #cv2.imshow("Blurred Image",blur1)
                #cv2.waitKey(0)
                #only show edges or colour changes
                edges1 = cv2.Canny(blur1,ctMin,ctMax,True)
                #cv2.imshow("Thin Edges Image",edges1)
                #cv2.waitKey(0)
                kernel1 = numpy.ones(kThicc,numpy.uint8)
                dilate1 = cv2.dilate(edges1, kernel1,iterations=1)
                idek = resizeCard(dilate1)
                #cv2.imshow("Fully Processed Image",idek)
                #cv2.waitKey(0)
                #compare current card with dataset
                guessCardVal = checkWhiteVal(idek,dataset,ranks)
                #check if dealer card or player card
                if y<220:
                    #append card value to dealer list...
                    playerCards=[]
                    if guessCardVal not in dealerCards and guessCardVal != 0:
                        if len(dealerCards)>1:
                            dealerCards.pop(-1)
                            dealerCards.pop(0)
                        dealerCards.append(guessCardVal)
                    #check if enough frames returned same card compare guess for dealer
                    if len(dList) > 30:
                        if len(set(dList[-29:-1][0])) <3:   
                            printToFile(dealerCards[0:1],[]) 
                            print("click any key to continue ")
                            cv2.waitKey(0)
                            dealerCards=[]
                            dList=[]
                else:
                    #append card value to player list...
         #           cv2.imshow("testing",idek)
                    dealerCards=[]
                    if guessCardVal not in playerCards and guessCardVal != 0:
                        if len(playerCards)>1:
                            playerCards.pop(-1)
                            playerCards.pop(0)
                        playerCards.append(guessCardVal)
                        
                    #optional mode to save cards to custom dataset
                    if saveCardsMode == 1:
                        cv2.imshow("newcard", blur1)
                        cname = input("input card suit + value:  ")
                        if cname == "q":
                            pass
                        else:
                            cv2.imwrite(cname + ".jpg",blur1)

                    #check if enough frames returned same card compare guess for player
                    if len(pList) > 30:
                        if len(set(pList[-29:-1][0])) <3:  
                            #print("printing cards to playerHand.txt")
                            printToFile([],playerCards[0:1])
                            cv2.waitKey(0)
                            pList=[]
                            playerCards=[]

                #overlay text on webcam view with card guess
                #jack=11, queen=12, king=13
                if guessCardVal != 0:
                    cv2.putText(image, "player, " + str(guessCardVal) if y>220 else "dealer, " + str(guessCardVal),(int(x - w / 2), int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, .75, [10,30,70], 3)
                    if len(playerCards)>0 and playerCards !=[0]:
                        pList.append(playerCards)
                        print("player cards = ", playerCards)
                        pass

                    if len(dealerCards)>0 and dealerCards !=[0]:
                        dList.append(dealerCards)
                        print("dealer cards = ", dealerCards)
                        pass
                #cv2.imshow("cimage",cimage)
                #cv2.imshow("idk",image)
    #show webcam image and processed webcam image side by sideS
    cv2.imshow("Image", numpy.hstack((image, cv2.cvtColor(dilate, cv2.COLOR_RGB2BGR))))
    
    
#main function
def main():
    global dList,pList,dealerCards,playerCards,printedCards
    ranks = []
    
    #get current directory
    path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)

    #clearing files for later use
    fp = open("playerHand.txt","w")
    fd = open("dealerHand.txt","w")
    fd.close()

    isTryR = 0
    try:
        #fill dataset
        dataset,ranks = getAllCards()
        isTryR = 1
    except:
        print("ERROR: failed to get dataset, exiting.")
        exit()
        #dataset = getAllCards2()
        isTryR = 0
        
    print(" - getting webcam ready -  \n   [ approx time 40 sec, nothing i can do ]")
    t0=time.time()
    cap = cv2.VideoCapture(0)
    t1=time.time()-t0
    print('took ' + str(t1) + 'seconds')
    print(" - webcam on - ")
    print(" -  running  - ")
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame,640)
        #cv2.imshow("Image",frame)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            #quits program
            break
        elif cv2.waitKey(1) & 0xFF == ord(' '):
            # clicking spacebar will clear current lists and files
            fp = open("playerHand.txt","w")
            fd = open("dealerHand.txt","w")
            fd.close()
            fp.close()
            pList = []
            dList = []
            playerCards = []
            dealerCards = []
            printedCards = []
            print("cleared list and files")

        #calling main processing function
        process(frame,dataset,ranks)

    cap.release()
    cv2.destroyAllWindows()


main()
