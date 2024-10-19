
import cv2 
import matplotlib.pyplot as plt

# Function to add text on the frame
def add_text(frame, text, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, 
             font_scale=1, font_color=(255, 125, 255), thickness=2):
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness)

# define a video capture object 
vid = cv2.VideoCapture(0) 

# Local video
# cap = cv.VideoCapture('vtest.avi')

print('Width:\n',vid.get(cv2.CAP_PROP_FRAME_WIDTH))
print('Height:\n',vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print("Frame per second (FPS)\n",vid.get(cv2.CAP_PROP_FPS))
# # 29.97002997002997
# print(vid.get(cv2.CAP_PROP_FRAME_COUNT))
# # 360.0
x= 32
while(True): 

	ret, frame = vid.read() 

	if not ret:
		print("Failed to grab a frame")
		break

	add_text(frame, f'Hello {x}!')

	cv2.imshow('frame', frame) 
	
	# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	# as opencv loads in BGR format by default, we want to show it in RGB.
	# plt.show()
	
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

vid.release() 
cv2.destroyAllWindows()
