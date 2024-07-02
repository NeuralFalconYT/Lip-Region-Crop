import argparse
import subprocess

def get_video_duration(video_path):
    command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    duration = subprocess.check_output(command, stderr=subprocess.STDOUT).decode('utf-8').strip()
    return float(duration)

def extract_audio_from_video_by_given_duration(video_path, duration, output_audio_path):
    command = ['ffmpeg', '-y', '-i', video_path, '-t', str(duration), '-vn', output_audio_path]
    subprocess.run(command, check=True)

def merge_audio_to_video(video_path, audio_path, output_video_path):
    # Get video duration
    video_duration = get_video_duration(video_path)

    # Merge audio into video
    command = ['ffmpeg', '-y', '-i', video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-shortest', output_video_path]
    subprocess.run(command, check=True)

    return output_video_path

def main():
    parser = argparse.ArgumentParser(description='Process video with audio merging using ffmpeg')
    parser.add_argument('input_video_path', help='Path to input video')
    parser.add_argument('audio_video_path', help='Path to video from which to extract audio')
    parser.add_argument('output_audio_path', help='Path to save the extracted audio')
    parser.add_argument('output_video_path', help='Path to save the output video with merged audio')
    args = parser.parse_args()

    # Get duration of input_video_path
    duration = get_video_duration(args.input_video_path)

    # Extract audio from audio_video_path with the given duration
    extract_audio_from_video_by_given_duration(args.audio_video_path, duration, args.output_audio_path)

    # Merge audio into input_video_path and save as output_video_path
    merge_audio_to_video(args.input_video_path, args.output_audio_path, args.output_video_path)

    print(f'Video with merged audio saved to: {args.output_video_path}')

if __name__ == '__main__':
    main()

# python mix_audio.py temp.mp4 video.mp4 temp.mp3 lip.mp4
