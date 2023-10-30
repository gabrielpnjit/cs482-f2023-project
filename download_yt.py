import pytube
import os
from pytube import YouTube
from pytube import Channel

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
count = 50
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