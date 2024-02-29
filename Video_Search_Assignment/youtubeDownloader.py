from pytube import YouTube
from pytube import Caption

class youtubeDownloader:
    def __init__(self, downloadPath='./'):
        self.downloadPath = downloadPath
    
    def downloadVideoWithCaptions(self, videoURLs):
        for videoURL in videoURLs:
            yt = YouTube(videoURL)
            stream = yt.streams.get_highest_resolution()
        
        print(f"Downloading video: {yt.title}...")
        stream.download(output_path=self.downloadPath)
        print("Video downloaded successfully!")
        
        if yt.captions:
            caption = yt.captions.get_by_language_code('en')
            if caption:
                print("Downloading captions...")
                caption.download(title= yt.title, srt= False ,output_path=self.downloadPath)
                print("Captions downloaded successfully!")
            else:
                print("No English caption available for this video.")
        else:
            print("No captions available for this video.")

if __name__ == "__main__":
    download_directory = './downloads/'
    videoURLs = [
        "https://www.youtube.com/watch?v=wbWRWeVe1XE",
        "https://www.youtube.com/watch?v=FlJoBhLnqko",
        "https://www.youtube.com/watch?v=Y-bVwPRy_no"
    ]

    downloader = youtubeDownloader(downloadPath=download_directory)
    downloader.downloadVideoWithCaptions(videoURLs)
