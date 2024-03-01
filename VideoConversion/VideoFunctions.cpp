#include <iostream>
#include <string>

// Function to convert video to PPM frames and extract audio
void videoToPpms(const std::string& inputVideoPath, const std::string& outputFolderPath) {
    // Construct FFmpeg command for extracting frames as PPM
    std::string frameCmd = "ffmpeg -i " + inputVideoPath + " " + outputFolderPath + "/frame%03d.ppm";
    // Construct FFmpeg command for extracting audio
    std::string audioCmd = "ffmpeg -i " + inputVideoPath + " -q:a 0 -map a " + outputFolderPath + "/audio.mp3";
    
    // Execute commands
    system(frameCmd.c_str());
    system(audioCmd.c_str());
}

// Function to create video from PPM frames and add audio
void ppmsToVideo(const std::string& inputFramesPathPattern, const std::string& inputAudioPath, const std::string& outputVideoPath) {
    // Construct FFmpeg command for creating video from PPM frames
    std::string videoCmd = "ffmpeg -framerate 24 -i " + inputFramesPathPattern + " -i " + inputAudioPath + " -c:v libx264 -pix_fmt yuv420p -c:a copy " + outputVideoPath;
    
    // Execute command
    system(videoCmd.c_str());
}

int main() {
    // Example usage: Replace "input.mp4" and "output_folder" with your actual video file and desired output directory
    videoToPpms("input.mp4", "output_folder");
    // Replace "output_folder/frame%03d.ppm", "output_folder/audio.mp3", and "reconstructed_video.mp4" with your actual paths
    ppmsToVideo("output_folder/frame%03d.ppm", "output_folder/audio.mp3", "reconstructed_video.mp4");
    
    return 0;
}
