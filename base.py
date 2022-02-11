import cv2
import face_recognition
import numpy
import os

def encoding(imagelist):
    encoding=[]
    for image in imagelist:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = face_recognition.face_encodings(image)[0]

        encoding.append(image)
    return encoding

path_known = "known_persons"
images=[]
knownPersonNames=[]
knownPersonList = os.listdir(path_known)
print(knownPersonList)

for n in knownPersonList:
    #reading the image and storing it as object in images list
    currentImage = cv2.imread(f'{path_known}/{n}')
    images.append((currentImage))
    #spliting the name and extension
    name = os.path.splitext(n)
    name1=name[0]
    knownPersonNames.append(name1)

print("names are",knownPersonNames)
encodeList = encoding(images)

bhupinder = face_recognition.load_image_file(path_known+"/bhupinder.jpg")
bhupinder =cv2.resize(bhupinder,(0,0),None,0.25,0.25)
bhupinder = cv2.cvtColor(bhupinder,cv2.COLOR_BGR2RGB)
bhupinder =face_recognition.face_encodings(bhupinder)[0]
bhupinder1 = face_recognition.compare_faces(encodeList,bhupinder)
distance =face_recognition.face_distance(encodeList,bhupinder)
print("distances are ",distance)
print("bhupinder is ",bhupinder1)
print("Encoding of known persons completed")
print(len(encodeList))


# operating webcamera
camera = cv2.VideoCapture(0)

while (True):

    print("-------------capturing-------")
    ret, frame = camera.read()
    # resizing the image to 1/4  here fx =0.25 and fy =0.25 i.e. its now 1/4 size
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

    # encoding
    # small_frame_RGB = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    small_frame_RGB = small_frame[:, :, ::-1]
    # face_locations = face_recognition.face_locations(rgb_small_frame)
    # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    CurrentFaceLocation = face_recognition.face_locations(small_frame_RGB)
    small_frame = face_recognition.face_encodings(small_frame_RGB,CurrentFaceLocation)
#----------------------------------------------------------------

# #----------------------------------------------------
    if small_frame==[]:
        # to prevent from out of range like if no person is there then continue
        print("No person Here")


    else:
        # othervise get the encodings from list
        print("There is person")
        small_frame = small_frame
        # encoding of frame captured is done
        # now here is to get all the faces showing in the frame
        face_locations = face_recognition.face_locations(small_frame_RGB)
        print("there are ",len(face_locations)," persons in web camera")

        #now we are going to compare the capture frame to the known perons images
        for frameEncoded,face_location in zip(small_frame,face_locations):
            print("Enter1")
            matches = face_recognition.compare_faces(encodeList, frameEncoded)
            print("Enter2")
            print(matches)
            nameOfPerson ="Unknow"
            facedistance = face_recognition.face_distance(encodeList,frameEncoded)
            print(facedistance)

            if True in matches:
                first_index = numpy.argmin(facedistance)
                print("Selected index is ",first_index)

                nameOfPerson=knownPersonNames[first_index].upper()
                # making rectangle
                # to make it to compatible for original size
                y1,x2,y2,x1 = face_location
                y1=y1*4
                x2=x2*4
                y2=y2*4
                x1=x1*4

                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3)
                cv2.rectangle(frame,(x1,y2),(x2,y2+20),(255,0,0), cv2.FILLED)
                cv2.putText(frame,nameOfPerson,(x1+3,y2+15),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)



        print("Recognised as ",nameOfPerson)


    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break


camera.release()

cv2.destroyAllWindows().imshow('webcam',frame)