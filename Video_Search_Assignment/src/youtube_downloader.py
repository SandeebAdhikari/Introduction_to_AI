from pytube import YouTube

class youtubeDownloader:
    def __init__(self, videoDownloadPath='./YouTube-Videos/'):
        self.videoDownloadPath = videoDownloadPath
       
    
    def downloadVideos(self, videoURLs):
        for videoURL in videoURLs:
            yt = YouTube(videoURL)
            stream = yt.streams.get_highest_resolution()
        
            print(f"Downloading video: {yt.title}...")
            stream.download(output_path=self.videoDownloadPath)
            print("Video downloaded successfully!")
    
        
if __name__ == "__main__":
    video_download_directory = './YouTube-Videos/'
    videoURLs = [
        "https://www.youtube.com/watch?v=wbWRWeVe1XE",
        "https://www.youtube.com/watch?v=FlJoBhLnqko",
        "https://www.youtube.com/watch?v=Y-bVwPRy_no"
      ] 

    downloader = youtubeDownloader(videoDownloadPath=video_download_directory)
    downloader.downloadVideos(videoURLs)