import cv2
import sys
import random
from random import randint, uniform
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.externals import joblib
import numpy as np

clf = joblib.load('ann150norm.pkl')
scaler = joblib.load('scaler150norm.pkl')

class Pool:
	def __init__(self, capacity):
		self.elm = []
		self.capacity = capacity

	def add(self, x):
		self.elm.append(x)
		if len(self.elm) > self.capacity:
			del self.elm[0]

	def mode(self):
		t = {}
		for i in self.elm:
			if i in t.keys():
				t[i] += 1
			else:
				t[i] = 1
		# print t
		k = sorted(t.keys(), key=lambda x: t[x], reverse=True)
		# print k
		return k[0]

def merge_proba(x, y):
	return x + y if x + y < 1 else 1

recent_moods = Pool(10)
def predict(face):
	X = scaler.transform(cv2.resize(face, (40, 40)).reshape(1,-1)).astype(np.float64)
	recent_moods.add(clf.predict(X)[0])
	proba = clf.predict_proba(X)[0]
	return recent_moods.mode(), [proba[0], proba[1], merge_proba(proba[2], proba[3])]


def polish(f, r, moods):
	x, y, w, h = r
	ww = w - 20
	hh = h/len(moods)
	#cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
	for i in range(len(moods)):
		cv2.line(f, (x + 10, y + i*hh + hh/2), ((x + 10 + ww, y + i*hh + hh/2)), (0, 0, 0), 2)
		cv2.line(f, (x + 10, y + i*hh + hh/2), (x + 10 + int(moods[i]*ww), y + i*hh + hh/2), \
		      (200, 200, 200), 2)
		cv2.circle(f, (x + 10 + int(moods[i]*ww), y + i*hh + hh/2), 4, (0,200,0), -1)

def intersect(r1, r2):
	x1, y1, w1, h1 = r1
	x2, y2, w2, h2 = r2
	return not ((y1 + h1 < y2 or y1 > y2 + h2) and (x1 + w1 < x2 or x1 > x2 + w2))

def unduplicatize(faces):
	return [faces[k] for k in range(len(faces)) if not k in \
			[j for i in range(len(faces)) for j in range(i + 1, len(faces)) \
	 		if intersect(faces[i], faces[j])]]

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

moods = ['aperto', 'chiuso', 'Pardon']

def in_left(f, r):
	x,_,w,_ = r
	return x+w<f.shape[1]/2

def in_right(f, r):
	x,_,_,_ = r
	return x > f.shape[1]/2

rps = ['aperto', 'chiuso','Pardon']
def judge(a, b):
	return [[ 0, -5,  5],
			[ 5,  0, -5],
			[-5,  5,  0]][a][b];

p1 = 100
p1_color = (0,0,200)
p2 = 100
p2_color = (200,200,0)
p1_prev = 0
p2_prev = 0

def update(a, b):
	global p1, p2
	p1 += judge(a, b)
	p2 -= judge(a, b)

white = (255, 255, 255)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30)
	)

    for r in faces:
		(x, y, w, h) = r
		i, proba = predict(gray[y:y+h,x:x+w].T)
		i = i if i !=3 else 2
		if in_left(frame, r):
			cv2.rectangle(frame, (x, y), (x+w, y+h), p1_color, 2)
			cv2.putText(frame, rps[i],(x,y+h+30), font, 1,white,2)
			if p1_prev == -1 or p1_prev != i:
				p1_prev = i
				update(p1_prev, p2_prev)
		elif in_right(frame, r):
			cv2.rectangle(frame, (x, y), (x+w, y+h), p2_color, 2)
			cv2.putText(frame, rps[i],(x,y+h+30), font, 1,white,2)
			if p2_prev == -1 or p2_prev != i:
				p2_prev = i
				update(p1_prev, p2_prev)
		else:
			cv2.rectangle(frame, (x, y), (x+w, y+h), white, 2)

		cv2.putText(frame, moods[i],(x,y-5), font, 1,white,2)
		# print predict(gray[y:y+h,x:x+w].T)
		polish(frame, (x+w+20, y, h/2, h), proba)

    f = frame
    #cv2.putText(f, 'Player1: {}'.format(p1), (10,40),font,1.2,white,3)
    #cv2.putText(f, 'Player1: {}'.format(p1), (10,40),font,1.2,p1_color,2)

    cv2.line(f, (f.shape[1]/2, 0), ((f.shape[1]/2, f.shape[0])), (255, 255, 255), 1)
    #cv2.putText(f, 'Player2: {}'.format(p2), (10+f.shape[1]/2,40),font,1.2,white,3)
    #cv2.putText(f, 'Player2: {}'.format(p2), (10+f.shape[1]/2,40),font,1.2,p2_color,2)


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
