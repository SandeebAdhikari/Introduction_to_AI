# Video Search Assignment
__youtubeDownloader.py__ -- This python file download youtube video and the close caption of the video

### To Access the downloaded youtube video and close caption:
__Build Docker Image__
docker build -t youtube-video-downloader .

__On Docker__
Run the image, then go to Files
On ./app/Video_Search_Assignment/Downloads/Caption you will find captions 
and On /app/Video_Search_Assignment/Downloads/YouTube-Videos you will find YouTube Videos

# Detection_Results
__preprocessVideo.py__ -- This pre-process the video and detects the object of an image that belong to MS COCO classes. 
Reports the result in the tabular form: [vidId, frameNum, timestamp, detectedObjId, detectedObjClass, confidence, bbox info]
__Downloaded Table:__ check ./Detection_Results

# Train and detect
__embeddingModel.py__-- It is the main file to execute convolution autoencoder and also to index video image embedding vectors in the database.
__autoencoder.py__ -- It has the code for the autoencoder and to train the autoencoder.
