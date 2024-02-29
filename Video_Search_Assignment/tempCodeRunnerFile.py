from pytube import YouTube
from pytube import Caption

class youtubeDownloader:
    def __init__(self, videoDownloadPath='./Video_Search_Assignment/Downloads/YouTube-Videos/',captionDownloadPath ='./Video_Search_Assignment/Downloads/Captions/'):
        self.videoDownloadPath = videoDownloadPath
        self.captionDownloadPath = captionDownloadPath
    
    def downloadVideoWithCaptions(self, videoURLs):
        for videoURL in videoURLs:
            yt = YouTube(videoURL)
            stream = yt.streams.get_highest_resolution()
        
            print(f"Downloading video: {yt.title}...")
            stream.download(output_path=self.videoDownloadPath)
            print("Video downloaded successfully!")
        
            if yt.captions:
                caption = yt.captions.get_by_language_code('en')
                if caption:
                    print("Downloading captions...")
                    caption.download(title= yt.title, srt= False ,output_path=self.captionDownloadPath)
                    print("Captions downloaded successfully!")
                else:
                    print("No English caption available for this video.")
            else:
                print("No captions available for this video.")

if __name__ == "__main__":
    video_download_directory = './Video_Search_Assignment/Downloads/YouTube-Videos/'
    caption_download_directory = './Video_Search_Assignment/Downloads/Captions/'
    videoURLs = [
        "https://www.youtube.com/watch?v=wbWRWeVe1XE",
        "https://www.youtube.com/watch?v=FlJoBhLnqko",
        "https://www.youtube.com/watch?v=Y-bVwPRy_no"
    ]

    downloader = youtubeDownloader(videoDownloadPath=video_download_directory, captionDownloadPath= caption_download_directory)
    downloader.downloadVideoWithCaptions(videoURLs)
