# streamlit-objectdetection 
this app is developed with the streamlit 
in this app we have used maskrnn and ssd 2 deep learning neural network for object detection
this also supports YOLO model(by just making small changes) as YOLO is too large in size i didnot inclued that

if u want to run it locally just download the above code and adjust the requirements as in requirements file(if u are using IDEs it will adjust itself)

to run the application just run this comand
'''
streamlit run main.py
'''

then this popups
![home page of app](https://github.com/Harishmmhari/streamlit-objectdetection/blob/main/openpage.png)
https://github.com/Harishmmhari/streamlit-objectdetection/blob/main/openpage.png
then you can select required video or image u want to recognize
even you can you webcam as source by clicking web cam button

here is some examples
## maskrnn
  this is model built to idetifiy the object and its proability of presence of that object in specific pixel
  this model's base architecture is RNN and CNN both 
  so there is two key parameters we have set to tune the output in this
  1.confidence
      this value is actually tells the model to give an output( a rectangle box value)
      where the object is present if and only if it is confiedent above this a value
  2.threshold
       this value gives to compute the proablity of each pixel in that box(a rectangle produced by taking confidence value to consideration)
       whether that pixel contains specific object detected.if that proabilty is more than threshold than it is masked output is asshoem below





## ssd
  this is small model which detects object as it is trained with very less classes and it has very small architecture compare to other models like(yolo,maskrnn)
  outputs are shown below
  
  
  
  
  
  
note: as i said yolo can use same code and use YOLO model too just make samll changes


       
