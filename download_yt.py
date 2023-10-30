import pytube
import os
from pytube import YouTube
from pytube import Channel
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# sanitizes file names for windows files
def sanitize_filename(filename):
    # List of characters that are invalid in Windows filenames
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# need to use fix from pr #1409 for channels
# apply here: "C:\Python311\Lib\site-packages\pytube\contrib\channel.py"
# need to use fix from pr #1085 for captions
# apply here: "C:\Python311\Lib\site-packages\pytube\captions.py"
c = Channel('https://www.youtube.com/user/NPR')
current_directory = os.path.dirname(os.path.realpath(__file__))
videos_output_directory = os.path.join(current_directory, 'videos')
captions_output_directory = os.path.join(current_directory, 'captions')

print(f'Downloading videos by: {c.channel_name}')

i = 0
count = 50 # number of videos to download
vid_count = 0
while i < len(c.videos) and vid_count < count:
	# have to use this method before accessing other properties like captions
	# https://github.com/pytube/pytube/issues/1674#issuecomment-1706105785
	# save videos
	video = c.videos[i]
	# try catch block handles age restricted and non captioned videos
	try:
		video.bypass_age_gate()
		caption = video.captions['en']
		video.streams.first().download(output_path=videos_output_directory)
		print(f'Video #{vid_count} - Downloading {video.title}')
		caption_content = caption.generate_srt_captions()
		caption_filename = f"{sanitize_filename(video.title)}.srt"
		caption_path = os.path.join(captions_output_directory, caption_filename)
		with open(caption_path, 'w', encoding='utf-8') as file:
			file.write(caption_content)
		vid_count += 1
		i += 1
	except:
		i += 1
		continue


# upload to google drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# create gdrive folder for whole dataset
folder_metadata = {'title': 'cs482-project-dataset','mimeType': 'application/vnd.google-apps.folder'}
folder = drive.CreateFile(folder_metadata)
folder.Upload()
folder_main_id = folder['id']
print(f"Folder captions created with ID: {folder_main_id}")

# create gdrive folders for captions
folder_metadata = {'title': 'captions','mimeType': 'application/vnd.google-apps.folder', 'parents': [{'id': folder_main_id}]}
folder = drive.CreateFile(folder_metadata)
folder.Upload()
folder_id = folder['id']
print(f"Folder captions created with ID: {folder_id}")

# upload to captions gdrive
for file_name in os.listdir(captions_output_directory):
    file_path = os.path.join(captions_output_directory, file_name)
    if os.path.isfile(file_path):
        print(f"Uploading {file_name} to folder captions")
        gfile = drive.CreateFile({
        		'title': file_name,
        		'parents': [{'id': folder_id}]
        	})
        gfile.SetContentFile(file_path)
        gfile.Upload()
        print(f"{file_name} has been uploaded to captions")

# create gdrive folders for videos
folder_metadata = {'title': 'videos','mimeType': 'application/vnd.google-apps.folder', 'parents': [{'id': folder_main_id}]}
folder = drive.CreateFile(folder_metadata)
folder.Upload()
folder_id = folder['id']
print(f"Folder videos created with ID: {folder_id}")

# upload to videos gdrive
for file_name in os.listdir(videos_output_directory):
    file_path = os.path.join(videos_output_directory, file_name)
    if os.path.isfile(file_path):
        print(f"Uploading {file_name} to folder videos")
        gfile = drive.CreateFile({
        		'title': file_name,
        		'parents': [{'id': folder_id}]
        	})
        gfile.SetContentFile(file_path)
        gfile.Upload()
        print(f"{file_name} has been uploaded to videos")