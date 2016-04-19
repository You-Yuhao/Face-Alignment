an Auto Face Segmentation program based on C++

I directly use other open source program to cover the face detection and face key points location. I mainly focus on developing the function of Auto Face Segmentation.In this program, the cropped face are aligned by eyes’ coordinates. In the 100*100 cropped face picture, the eyes are kept at the 30px(Y coordinate), and the eyes` distance is kept as 40px.

face detection:[https://github.com/ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection)

face key points location:[https://github.com/delphifirst/FaceX](https://github.com/delphifirst/FaceX)

In the "Release" folder, you can get a demo. 
Firstly, you should choose the size of cropped face picture, the size of cropped eyes picture and so on.In this program I provide 3 choices: 200*200, 200*200, 100*100. It`s the size of cropped face picture. And the 1st one has a smaller size of eyes picture and so on, while the 2nd and 3rd have a bigger size of eyes picture and so on.
Secondly, you should give the number of pictures you want to use which should be put in the "Release" folder. The pictures should be named like "1.jpg", and only "jpg" is allowed.

Finally, you will get these pictures:

SourePic:

![SourePic](http://7xr8d2.com1.z0.glb.clouddn.com/AutoFaceSegmentation/jpg/22.jpg)

Face:
![Face](http://7xr8d2.com1.z0.glb.clouddn.com/AutoFaceSegmentation/jpg/face.jpg)

Brows:
![Brows](http://7xr8d2.com1.z0.glb.clouddn.com/AutoFaceSegmentation/jpg/brows.jpg)

leftBrow:
![leftBrow](http://7xr8d2.com1.z0.glb.clouddn.com/AutoFaceSegmentation/jpg/leftBrow.jpg)

rightBrow:
![rightBrow](http://7xr8d2.com1.z0.glb.clouddn.com/AutoFaceSegmentation/jpg/rightBrow.jpg)

Eyes:
![Eyes](http://7xr8d2.com1.z0.glb.clouddn.com/AutoFaceSegmentation/jpg/eyes.jpg)

leftEye:
![leftEye](http://7xr8d2.com1.z0.glb.clouddn.com/AutoFaceSegmentation/jpg/leftEye.jpg)

rightEye:
![rightEye](http://7xr8d2.com1.z0.glb.clouddn.com/AutoFaceSegmentation/jpg/rightEye.jpg)

nose:
![nose](http://7xr8d2.com1.z0.glb.clouddn.com/AutoFaceSegmentation/jpg/nose.jpg)

mouth:
![mouth](http://7xr8d2.com1.z0.glb.clouddn.com/AutoFaceSegmentation/jpg/mouth.jpg)
