# CardRecog
uses computer webcam to recognize playing cards on a table


Requires pip libraries:
 - cv2
 - imutils
 - numpy
 - glob
 - os
 - time
 - sys
 - socket
    - (socket optional, can be used to send data to unity etc)


CardRec.py:
  - main file to run, requires images of basic deck of cards to work
  - code uses number in corner to identify card
  - will open window showing webcam view, cards being recognized will be highlighted on screen
  
  
DeckSaver.py:
  - use to upload custom deck of cards, optional
  - can adjust individual files if necessary
  
