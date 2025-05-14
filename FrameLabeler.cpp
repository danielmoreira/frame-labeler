/** Video Frame Labeler
 *
 * Singleton CPP implementation of a video frame labeler for binary classification.
 * I know OO but, c'mon, not today. =)
 * Author: Daniel Moreira (now at dmoreira1@luc.edu)
 *
 * Needed libraries to compile this code:
 * - OpenCV (strong dependence)
 * - Boost (weak dependence)
 */

/* Imported libraries. */
#include <fstream>
#include <thread>
#include <dirent.h>
#include <sys/stat.h>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>

/* Namespaces' declaration. */
using namespace std;
using namespace cv;
using namespace boost;

/* Configuration operation values of the labeler. */
/** Size of the buffers to hold in memory part of the frames of the video to be tagged. */
int VIDEO_FRAME_BUFFERS_SIZE = 64; // times 3 buffers

/** Number of frames to jump when wanted (by the means of the w/z keys). */
int FRAME_JUMP_SIZE = 100;

/** Returns the current date and time. */
string getCurrentDateTime() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

/** Reads a list with absolute paths to videos, from a given text file. */
void readVideoFilePathList(string inputFilePath, vector <string> *answer) {
    // input text file reader
    ifstream fileReader;
    fileReader.open(inputFilePath.data());
    if (fileReader.fail()) {
        cerr << "Could not open file " << inputFilePath << "." << endl;
        throw -1;
    }

    // holds the current line read from the input file
    string line;

    // while there are paths to be read, adds them to the answer
    while (getline(fileReader, line))
        answer->push_back(line);

    // closes the input file
    fileReader.close();
}

/** Determines new width and height values for the frames extracted from a given video file,
 *  of which original dimensions are given as parameters, assuming that the user decided for
 *  a new number of pixels per frame, and that the original video aspect ratio must be
 *  maintained. If the desired new number of pixels per frame is greater than the original
 *  one, this function does nothing (i.e. it simply returns the original video width and
 *  height. */
void calculateNewWidthAndHeight(int originalWidth, int originalHeight,
                                int desiredPixelCount, int *newWidth, int *newHeight) {
    int originalPixelCount = originalWidth * originalHeight;

    // calculates the new width and height
    if (desiredPixelCount < originalPixelCount) {
        double videoAspectRatio = originalWidth * pow(originalHeight, -1);

        *newHeight = round(sqrt(desiredPixelCount / videoAspectRatio));
        *newWidth = round(videoAspectRatio * *newHeight);
    }
        // does not change video resolution, if the desired pixel count is greater than
        // the current one
    else {
        *newHeight = originalHeight;
        *newWidth = originalWidth;
    }
}

/** Extracts all the frames from a given video, and saves them in the given directory.
 *  It is recommended for the video to be in H.264 MPEG-4 format. The frames are output as
 *  JPG images, named with the video file name + frame number + a sequence number (from
 *  00000001 to N). The size of the saved frames can be informed as a new desired total
 *  number of pixels per frame, or 0 if the original size shall be maintained. If the new
 *  desired total number of pixels is greater than the original one, the sizes of the frames
 *  are simply maintained. */
void extractAndSaveVideoFrames(string videoFilePath, string frameDirPath,
                               int totalPixelCount) {
    // tries to open the given dir path to store the extracted i-frames
    DIR *pDir;
    pDir = opendir(frameDirPath.data());
    if (pDir == NULL)
        // tries to create the directory
        mkdir(videoFilePath.data(), 0777);

    pDir = opendir(frameDirPath.data());
    if (pDir == NULL) {
        cerr << "Could not open neither create directory " << frameDirPath
             << "." << endl;
        throw -1;
    }
    closedir(pDir);

    // obtains the name of the original video file
    vector <string> *videoFilePathTokens = new vector<string>;
    split(*videoFilePathTokens, videoFilePath, is_any_of("/"));
    string videoFileName = videoFilePathTokens->back();
    videoFilePathTokens->clear();
    delete videoFilePathTokens;

    // video reader
    VideoCapture *videoReader = new VideoCapture(videoFilePath);

    // determines new frame width and frame height values, if it is the case
    int frameWidth = 0, frameHeight = 0;
    if (totalPixelCount > 0) {
        Mat firstFrame;
        videoReader->read(firstFrame);

        int originalFrameWidth, originalFrameHeight;
        originalFrameWidth = firstFrame.cols;
        originalFrameHeight = firstFrame.rows;

        calculateNewWidthAndHeight(originalFrameWidth, originalFrameHeight,
                                   totalPixelCount, &frameWidth, &frameHeight);

        videoReader->release();
        videoReader = new VideoCapture(videoFilePath);
    }

    // extracts the frames
    int frameCount = -1;
    Mat currentFrame;
    while (videoReader->read(currentFrame)) {
        // one more frame obtained
        frameCount++;

        // if the frame is to be resized, does it
        if (frameWidth > 0 && frameHeight > 0)
            resize(currentFrame, currentFrame, Size(frameWidth, frameHeight),
                   INTER_CUBIC);

        // mounts the name of the file of the current frame
        // completes the number of the frame with zeros
        char frameNumberChars[8];
        sprintf(frameNumberChars, "%.7d", frameCount);

        // mounts the name of the current file
        stringstream frameFilePathStream;
        frameFilePathStream << frameDirPath << "/" << videoFileName << "-"
                            << frameNumberChars << ".jpg";

        // saves the frame
        imwrite(frameFilePathStream.str(), currentFrame);
    }

    // frees memory
    videoReader->release();
    delete videoReader;
}

/** Reads a given input file and obtains a list with the file paths of the frames
 *  previously extracted from a video to be annotated. */
void readFrameFilePaths(string inputFilePath, vector <string> *frameFilePaths) {
    ifstream fileReader;
    fileReader.open(inputFilePath.data());
    if (fileReader.fail()) {
        cerr << "Could not open file " << inputFilePath << "." << endl;
        throw -1;
    }

    string line;
    while (getline(fileReader, line))
        frameFilePaths->push_back(line);

    fileReader.close();
}

/** Reads the content of a given ETF file, regarding the annotation of a video
 *  of interest, of which file name is given as a parameter.
 *
 *  The frame rate of the video must also be informed, in terms of FPS.
 *
 *  Parameter <positiveFrames> outputs the number of the frames marked as
 *  positive in the ETF file.
 *  Parameter <negativeFrames>, on its turn, outputs the number of the frames
 *  marked as negative.
 *
 *  ETF file: format created within the MediaEval (https://multimediaeval.github.io/)
 *  violent scenes localization task. */
void readInputETFFile(string videoFileName, double videoFPS, string etfFilePath,
                      set<int> *positiveFrames, set<int> *negativeFrames) {
    // opens the ETF file
    ifstream etfReader;
    etfReader.open(etfFilePath.data());
    if (etfReader.fail()) {
        cerr << "Could not open file " << etfFilePath << "." << endl;
        throw -1;
    }

    // keeps on reading the ETF file,
    // until it reaches the lines related to the wanted video file
    string etfLine;
    while (getline(etfReader, etfLine))
        if (etfLine[0] != '#' && etfLine.find(videoFileName) != string::npos) {
            // parses the current line
            stringstream etfLineStream;
            etfLineStream << etfLine;

            // obtained data
            string generalToken, label;
            double beginTime, duration;
            etfLineStream >> generalToken >> generalToken >> beginTime
                          >> duration >> generalToken >> generalToken >> generalToken
                          >> generalToken >> label;
            if (etfLineStream.fail()) {
                cerr << "File " << etfFilePath << " is not a valid ETF one."
                     << endl;
                throw -2;
            }

            // fulfills the numbers of positive or negative frames
            double firstFrameNumber = beginTime * videoFPS;
            double lastFrameNumber = firstFrameNumber + duration * videoFPS;

            if (label == "t")
                for (int i = round(firstFrameNumber); i < lastFrameNumber; i++)
                    positiveFrames->insert(i);
            else
                for (int i = round(firstFrameNumber); i < lastFrameNumber; i++)
                    negativeFrames->insert(i);
        }

    // closes the ETF file
    etfReader.close();
}

/** Loads video frames into the given buffer <frameBuffer>, accordingly to the
 *  given frame number interval [<initialFrameNumber>, <finalFrameNumber>),
 *  from the given list of frame file paths <frameFilePaths>.
 *
 *  Visually adjusts the read frame to contain some program operation info. */
void loadVideoFrames(vector <Mat> *frameBuffer, int initialFrameNumber,
                     int finalFrameNumber, vector <string> *frameFilePaths) {
    for (int i = initialFrameNumber; i < finalFrameNumber; i++) {
        Mat currentFrame = imread(frameFilePaths->at(i));

        Mat treatedFrame = Mat::zeros(50, currentFrame.cols,
                                      currentFrame.type());
        treatedFrame.push_back(currentFrame);

        Mat frameFootnote = Mat::zeros(60, currentFrame.cols,
                                       currentFrame.type());
        string line1 =
                "[space] play-stop / [r]everse / [+] faster / [-] slower / [q]uit";
        string line2 =
                "[a] previous / [s] next / [w] previous 100 / [z] next 100 / [b]egin / [e]nd";
        string line3 =
                "[0] negative / [1] positive / [j] previous mark / [k] next mark / [l] record label";
        putText(frameFootnote, line1, Point(10, 15), FONT_HERSHEY_PLAIN, 1,
                Scalar(0, 200, 0));
        putText(frameFootnote, line2, Point(10, 35), FONT_HERSHEY_PLAIN, 1,
                Scalar(0, 200, 0));
        putText(frameFootnote, line3, Point(10, 55), FONT_HERSHEY_PLAIN, 1,
                Scalar(0, 200, 0));
        treatedFrame.push_back(frameFootnote);

        frameBuffer->push_back(treatedFrame);
    }
}

/** Loads the video frames related to the given frame buffer <frameBuffer>,
 *  accordingly to the given first frame number of reference
 *  <refCurrentBufferedFrameNumber>.
 *
 *  Parameter <next> is TRUE if the buffer to be filled refers to the next
 *  frames to be shown, FALSE otherwise.
 *
 *  Parameter <frameFilePaths> contains the file paths to the video frames,
 *  previously extracted.
 *
 *  Parameter <loadVideoFramesMutex> is a mutex to control the access to the
 *  given buffer. */
void loadVideoFrameBuffer(vector <Mat> *frameBuffer,
                          int *refCurrentBufferedFrameNumber, bool next,
                          vector <string> *frameFilePaths, Mutex *loadVideoFramesMutex) {
    while (!frameFilePaths->empty())
        if (frameBuffer->empty()) {
            loadVideoFramesMutex->lock();

            int initialFrameNumber, finalFrameNumber;
            if (next) {
                initialFrameNumber = *refCurrentBufferedFrameNumber
                                     + VIDEO_FRAME_BUFFERS_SIZE;
                if (initialFrameNumber >= frameFilePaths->size())
                    initialFrameNumber = *refCurrentBufferedFrameNumber;

                finalFrameNumber = initialFrameNumber
                                   + VIDEO_FRAME_BUFFERS_SIZE;
                if (finalFrameNumber >= frameFilePaths->size())
                    finalFrameNumber = frameFilePaths->size();
            } else {
                initialFrameNumber = *refCurrentBufferedFrameNumber
                                     - VIDEO_FRAME_BUFFERS_SIZE;
                if (initialFrameNumber < 0)
                    initialFrameNumber = 0;

                finalFrameNumber = initialFrameNumber
                                   + VIDEO_FRAME_BUFFERS_SIZE;
                if (finalFrameNumber >= frameFilePaths->size())
                    finalFrameNumber = frameFilePaths->size();
            }

            loadVideoFrames(frameBuffer, initialFrameNumber, finalFrameNumber,
                            frameFilePaths);

            loadVideoFramesMutex->unlock();
        } else
            // put thread to sleep
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

/** Adjusts the given frame <frame>, preparing it to be shown with some info
 *  about its annotation process:
 *
 *  - Frame number, by means of parameter <frameNumber>;
 *
 *  - Total frame count, by means of parameter <framesCount>;
 *
 *  - The delay used to show the given frame, by means of the parameter <videoShowingDelay>;
 *
 *  - If the frames are being shown in reverse mode, by means of parameter <playReverse>;
 *
 *  - If the label of the frame is being overwritten while being shown, by means of
 *    parameter <overwriteLabels>;
 *
 *  - The label of the current frame (1 for positive, or 0 for negative), by means of
 *    parameter <currentLabel>;
 *
 *  - The positive and negative already annotated frames, by means of parameters
 *    <positiveFrames> and <negativeFrames>. */
void prepareToRenderFrameStatus(Mat *frame, int frameNumber, int framesCount,
                                int videoShowingDelay, bool playReverse, bool overwriteLabels,
                                int currentLabel, set<int> *positiveFrames, set<int> *negativeFrames) {
    rectangle(*frame, Point(50, 5), Point(1000, 45), Scalar(0, 0, 0), -1);

    stringstream controlStream1;
    controlStream1 << "frame number: " << frameNumber << "/" << framesCount;
    if (videoShowingDelay == 0)
        controlStream1 << ", stopped";
    else if (!playReverse)
        controlStream1 << ", playing @mspf " << videoShowingDelay;
    else
        controlStream1 << ", reverse @mspf " << videoShowingDelay;

    stringstream controlStream2;
    if (!overwriteLabels)
        controlStream2 << "just showing...";
    else
        controlStream2 << "labeling as "
                       << (currentLabel == 0 ? "NEGATIVE" : "POSITIVE");

    putText(*frame, controlStream1.str(), Point(55, 20), FONT_HERSHEY_PLAIN, 1,
            Scalar(0, 200, 0));
    putText(*frame, controlStream2.str(), Point(55, 40), FONT_HERSHEY_PLAIN, 1,
            Scalar(0, 200, 0));

    if (positiveFrames->find(frameNumber) != positiveFrames->end())
        rectangle(*frame, Point(0, 0), Point(40, 40), Scalar(0, 0, 200), -1);
    else if (negativeFrames->find(frameNumber) != positiveFrames->end())
        rectangle(*frame, Point(0, 0), Point(40, 40), Scalar(0, 200, 0), -1);

    if (overwriteLabels)
        circle(*frame, Point(20, 20), 10, Scalar(0, 0, 0), -1);
}

/** Treats the given key <key> command input with the keyboard, and controls
 *  the video annotation process.
 *
 *  Parameter <currentVideoFrameNumber> contains the number of the current frame
 *  being shown.
 *
 *  Parameter <refCurrentBufferedFrameNumber> contains the number of the first
 *  frame held at the buffer of current frames.
 *
 *  Parameter <videoShowingDelay> contains the delay used to show the video frames.
 *
 *  Parameter <playReverse> is TRUE if the frames are being shown in reverse mode,
 *  FALSE otherwise.
 *
 *  Parameter <overwriteLabels> is TRUE if the labels of the frames are being overwritten
 *  while they are being shown, FALSE otherwise.
 *
 *  Parameter <currentLabel> contains the label of the current frame: 0 for negative,
 *  1 for positive.
 *
 *  Parameter <frameFilePaths> is a list containing the file paths of the video frames,
 *  properly sorted in exhibition time.
 *
 *  Parameter <currentVideoFrameBuffer> is the buffer of the frames currently being shown.
 *  Parameter <previousVideoFrameBuffer> is the buffer of the frames previously being shown.
 *  Parameter <nextVideoFrameBuffer> is the buffer of the frames supposed to be shown in the
 *  near future.
 *
 *  Parameter <positiveFrames> contains the numbers of the video frames already annotated as
 *  positive.
 *  Parameter <negativeFrames> contains the numbers of the video frames already annotated as
 *  negative.
 *
 *  Parameter <loadFramesIntoPreviousBufferMutex> is a mutex controlling the access the
 *  <previousVideoFrameBuffer> buffer.
 *  Parameter <loadFramesIntoNextBufferMutex> is a mutex controlling the access the
 *  <nextVideoFrameBuffer> buffer. */
void treatKeyboardInput(char key, int *currentVideoFrameNumber,
                        int *refCurrentBufferedFrameNumber, int *videoShowingDelay,
                        bool *playReverse, bool *overwriteLabels, int *currentLabel,
                        vector <string> *frameFilePaths, vector <Mat> *currentVideoFrameBuffer,
                        vector <Mat> *previousVideoFrameBuffer,
                        vector <Mat> *nextVideoFrameBuffer, set<int> *positiveFrames,
                        set<int> *negativeFrames, Mutex *loadFramesIntoPreviousBufferMutex,
                        Mutex *loadFramesIntoNextBufferMutex) {
    int frameNumber;

    switch (key) {
        case 'q':
            frameFilePaths->clear(); // makes the program finish and save results
            break;

        case '+':
            *videoShowingDelay > 20 ?
                    *videoShowingDelay = *videoShowingDelay - 20 :
                    *videoShowingDelay = 1;
            break;

        case '-':
            *videoShowingDelay = *videoShowingDelay + 20;
            break;

        case ' ':
            if (*videoShowingDelay > 0)
                *videoShowingDelay = 0;
            else {
                *videoShowingDelay = 40;
                *playReverse = false;
            }
            break;

        case 'r':
            *videoShowingDelay = 40;
            *overwriteLabels = false;
            *playReverse = true;
            break;

        case 'l':
            *videoShowingDelay = 0;
            *overwriteLabels = !(*overwriteLabels);
            break;

        case '0':
            *overwriteLabels = true;
            *videoShowingDelay = 0;
            *currentLabel = 0;
            break;

        case '1':
            *overwriteLabels = true;
            *videoShowingDelay = 0;
            *currentLabel = 1;
            break;

        case 'a': // left arrow
            *overwriteLabels = false;
            *videoShowingDelay = 0;
            *currentVideoFrameNumber > 0 ?
            (*currentVideoFrameNumber)-- : *currentVideoFrameNumber = 0;

            // if out of buffer
            if (*currentVideoFrameNumber < *refCurrentBufferedFrameNumber) {
                loadFramesIntoNextBufferMutex->lock();
                loadFramesIntoPreviousBufferMutex->lock();

                // the next buffer is the current one
                nextVideoFrameBuffer->clear();
                *nextVideoFrameBuffer = *currentVideoFrameBuffer;

                // the current buffer is the previous one
                currentVideoFrameBuffer->clear();
                *currentVideoFrameBuffer = *previousVideoFrameBuffer;

                // restarts the previous buffer
                previousVideoFrameBuffer->clear();

                // updates the reference number of the 1st frame inside the current buffer
                if (*refCurrentBufferedFrameNumber - VIDEO_FRAME_BUFFERS_SIZE >= 0)
                    *refCurrentBufferedFrameNumber = *refCurrentBufferedFrameNumber
                                                     - VIDEO_FRAME_BUFFERS_SIZE;
                // else maintains the number

                loadFramesIntoNextBufferMutex->unlock();
                loadFramesIntoPreviousBufferMutex->unlock();
            }
            break;

        case 's': // right arrow
            *videoShowingDelay = 0;
            *currentVideoFrameNumber < frameFilePaths->size() - 1 ?
            (*currentVideoFrameNumber)++ :
                    *currentVideoFrameNumber = frameFilePaths->size() - 1;

            // if out of buffer
            // (i.e. buffers need to be updated)
            if (*currentVideoFrameNumber
                >= *refCurrentBufferedFrameNumber + VIDEO_FRAME_BUFFERS_SIZE) {
                loadFramesIntoNextBufferMutex->lock();
                loadFramesIntoPreviousBufferMutex->lock();

                // the previous buffer is the current one
                previousVideoFrameBuffer->clear();
                *previousVideoFrameBuffer = *currentVideoFrameBuffer;

                // the current buffer is the next one
                currentVideoFrameBuffer->clear();
                *currentVideoFrameBuffer = *nextVideoFrameBuffer;

                // restarts the next buffer
                nextVideoFrameBuffer->clear();

                // updates the reference number of the 1st frame inside the current buffer
                if (*refCurrentBufferedFrameNumber + VIDEO_FRAME_BUFFERS_SIZE
                    < frameFilePaths->size())
                    *refCurrentBufferedFrameNumber = *refCurrentBufferedFrameNumber
                                                     + VIDEO_FRAME_BUFFERS_SIZE;
                // else maintains the number

                loadFramesIntoNextBufferMutex->unlock();
                loadFramesIntoPreviousBufferMutex->unlock();
            }
            break;

        case 'w': // up arrow
            loadFramesIntoNextBufferMutex->lock();
            loadFramesIntoPreviousBufferMutex->lock();

            *overwriteLabels = false;
            *videoShowingDelay = 0;
            *currentVideoFrameNumber < frameFilePaths->size() - FRAME_JUMP_SIZE ?
                    *currentVideoFrameNumber = *currentVideoFrameNumber
                                               + FRAME_JUMP_SIZE :
                    *currentVideoFrameNumber = frameFilePaths->size() - 1;
            *refCurrentBufferedFrameNumber = int(
                    *currentVideoFrameNumber / VIDEO_FRAME_BUFFERS_SIZE)
                                             * VIDEO_FRAME_BUFFERS_SIZE;

            currentVideoFrameBuffer->clear();
            previousVideoFrameBuffer->clear();
            nextVideoFrameBuffer->clear();

            loadFramesIntoNextBufferMutex->unlock();
            loadFramesIntoPreviousBufferMutex->unlock();

            // gathers the current buffer of frames
            loadVideoFrames(currentVideoFrameBuffer, *refCurrentBufferedFrameNumber,
                            (*refCurrentBufferedFrameNumber + VIDEO_FRAME_BUFFERS_SIZE
                             < frameFilePaths->size() ?
                             *refCurrentBufferedFrameNumber
                             + VIDEO_FRAME_BUFFERS_SIZE :
                             frameFilePaths->size()), frameFilePaths);
            break;

        case 'z': // down arrow
            loadFramesIntoNextBufferMutex->lock();
            loadFramesIntoPreviousBufferMutex->lock();

            *overwriteLabels = false;
            *videoShowingDelay = 0;
            *currentVideoFrameNumber > FRAME_JUMP_SIZE ?
                    *currentVideoFrameNumber = *currentVideoFrameNumber
                                               - FRAME_JUMP_SIZE :
                    *currentVideoFrameNumber = 0;
            *refCurrentBufferedFrameNumber = int(
                    *currentVideoFrameNumber / VIDEO_FRAME_BUFFERS_SIZE)
                                             * VIDEO_FRAME_BUFFERS_SIZE;

            currentVideoFrameBuffer->clear();
            previousVideoFrameBuffer->clear();
            nextVideoFrameBuffer->clear();

            loadFramesIntoNextBufferMutex->unlock();
            loadFramesIntoPreviousBufferMutex->unlock();

            // gathers the current buffer of frames
            loadVideoFrames(currentVideoFrameBuffer, *refCurrentBufferedFrameNumber,
                            (*refCurrentBufferedFrameNumber + VIDEO_FRAME_BUFFERS_SIZE
                             < frameFilePaths->size() ?
                             *refCurrentBufferedFrameNumber
                             + VIDEO_FRAME_BUFFERS_SIZE :
                             frameFilePaths->size()), frameFilePaths);
            break;

        case 'b':
            loadFramesIntoNextBufferMutex->lock();
            loadFramesIntoPreviousBufferMutex->lock();

            *overwriteLabels = false;
            *videoShowingDelay = 0;
            *currentVideoFrameNumber = 0;
            *refCurrentBufferedFrameNumber = 0;

            currentVideoFrameBuffer->clear();
            previousVideoFrameBuffer->clear();
            nextVideoFrameBuffer->clear();

            loadFramesIntoNextBufferMutex->unlock();
            loadFramesIntoPreviousBufferMutex->unlock();

            // gathers the current buffer of frames
            loadVideoFrames(currentVideoFrameBuffer, *refCurrentBufferedFrameNumber,
                            (*refCurrentBufferedFrameNumber + VIDEO_FRAME_BUFFERS_SIZE
                             < frameFilePaths->size() ?
                             *refCurrentBufferedFrameNumber
                             + VIDEO_FRAME_BUFFERS_SIZE :
                             frameFilePaths->size()), frameFilePaths);
            break;

        case 'e':
            loadFramesIntoNextBufferMutex->lock();
            loadFramesIntoPreviousBufferMutex->lock();

            *overwriteLabels = false;
            *videoShowingDelay = 0;
            *currentVideoFrameNumber = frameFilePaths->size() - 1;
            *refCurrentBufferedFrameNumber = int(
                    *currentVideoFrameNumber / VIDEO_FRAME_BUFFERS_SIZE)
                                             * VIDEO_FRAME_BUFFERS_SIZE;

            currentVideoFrameBuffer->clear();
            previousVideoFrameBuffer->clear();
            nextVideoFrameBuffer->clear();

            loadFramesIntoNextBufferMutex->unlock();
            loadFramesIntoPreviousBufferMutex->unlock();

            // gathers the current buffer of frames
            loadVideoFrames(currentVideoFrameBuffer, *refCurrentBufferedFrameNumber,
                            (*refCurrentBufferedFrameNumber + VIDEO_FRAME_BUFFERS_SIZE
                             < frameFilePaths->size() ?
                             *refCurrentBufferedFrameNumber
                             + VIDEO_FRAME_BUFFERS_SIZE :
                             frameFilePaths->size()), frameFilePaths);
            break;

        case 'j':
            loadFramesIntoNextBufferMutex->lock();
            loadFramesIntoPreviousBufferMutex->lock();

            frameNumber = *currentVideoFrameNumber;

            if (frameNumber > 0) {
                frameNumber--;

                if (positiveFrames->find(frameNumber) != positiveFrames->end())
                    while (positiveFrames->find(frameNumber)
                           != positiveFrames->end())
                        frameNumber--;
                else
                    while (negativeFrames->find(frameNumber)
                           != negativeFrames->end())
                        frameNumber--;

                frameNumber++;
            }

            *overwriteLabels = false;
            *videoShowingDelay = 0;
            *currentVideoFrameNumber = frameNumber;
            *refCurrentBufferedFrameNumber = int(
                    *currentVideoFrameNumber / VIDEO_FRAME_BUFFERS_SIZE)
                                             * VIDEO_FRAME_BUFFERS_SIZE;

            currentVideoFrameBuffer->clear();
            previousVideoFrameBuffer->clear();
            nextVideoFrameBuffer->clear();

            loadFramesIntoNextBufferMutex->unlock();
            loadFramesIntoPreviousBufferMutex->unlock();

            // gathers the current buffer of frames
            loadVideoFrames(currentVideoFrameBuffer, *refCurrentBufferedFrameNumber,
                            (*refCurrentBufferedFrameNumber + VIDEO_FRAME_BUFFERS_SIZE
                             < frameFilePaths->size() ?
                             *refCurrentBufferedFrameNumber
                             + VIDEO_FRAME_BUFFERS_SIZE :
                             frameFilePaths->size()), frameFilePaths);
            break;

        case 'k':
            loadFramesIntoNextBufferMutex->lock();
            loadFramesIntoPreviousBufferMutex->lock();

            frameNumber = *currentVideoFrameNumber;

            if (frameNumber < frameFilePaths->size() - 1) {
                frameNumber++;

                if (positiveFrames->find(frameNumber) != positiveFrames->end())
                    while (positiveFrames->find(frameNumber)
                           != positiveFrames->end())
                        frameNumber++;
                else
                    while (negativeFrames->find(frameNumber)
                           != negativeFrames->end())
                        frameNumber++;
            }

            *overwriteLabels = false;
            *videoShowingDelay = 0;
            *currentVideoFrameNumber = (
                    frameNumber < frameFilePaths->size() ?
                    frameNumber : frameFilePaths->size() - 1);
            *refCurrentBufferedFrameNumber = int(
                    *currentVideoFrameNumber / VIDEO_FRAME_BUFFERS_SIZE)
                                             * VIDEO_FRAME_BUFFERS_SIZE;

            currentVideoFrameBuffer->clear();
            previousVideoFrameBuffer->clear();
            nextVideoFrameBuffer->clear();

            loadFramesIntoNextBufferMutex->unlock();
            loadFramesIntoPreviousBufferMutex->unlock();

            // gathers the current buffer of frames
            loadVideoFrames(currentVideoFrameBuffer, *refCurrentBufferedFrameNumber,
                            (*refCurrentBufferedFrameNumber + VIDEO_FRAME_BUFFERS_SIZE
                             < frameFilePaths->size() ?
                             *refCurrentBufferedFrameNumber
                             + VIDEO_FRAME_BUFFERS_SIZE :
                             frameFilePaths->size()), frameFilePaths);
            break;

        default:
            break;
    }
}

/** Shows the frames related to the given file paths <frameFilePaths>.
 *
 *  Parameters <positiveFrames> and <negativeFrames> are sets containing the
 *  numbers of the positive and of the negative frames, respectively. */
void showVideoFrames(vector <string> *frameFilePaths, set<int> *positiveFrames,
                     set<int> *negativeFrames) {
    // delay to show video frames (milliseconds per frame, MSPF)
    int videoShowingDelay = 0; // 0: wait key

    // buffers that hold the frames of the video being annotated
    vector <Mat> currentVideoFrameBuffer, previousVideoFrameBuffer,
            nextVideoFrameBuffer;

    // holds the number of the first frame put in the current buffer
    int refCurrentBufferFrameNumber = 0;

    // holds the number of the video frame currently being shown
    int currentVideoFrameNumber = 0;

    // indicates if the video is supposed to be displayed in reversed order
    bool playReverse = false;

    // indicates if the labels are to be overwritten
    bool overwriteLabels = false;

    // holds the current label to give to the played frames:
    // 0 for negative, 1 for positive
    int currentLabel = 0;

    // mutexes to control the filling of the frame buffers
    Mutex *loadFramesIntoPreviousBufferMutex = new Mutex();
    Mutex *loadFramesIntoNextBufferMutex = new Mutex();
    // threads to keep on feeding the buffers
    thread *previousBufferThread = new thread(loadVideoFrameBuffer, &previousVideoFrameBuffer,
                                              &refCurrentBufferFrameNumber, false, frameFilePaths,
                                              loadFramesIntoPreviousBufferMutex);
    thread *nextBufferThread = new thread(loadVideoFrameBuffer, &nextVideoFrameBuffer,
                                          &refCurrentBufferFrameNumber, true, frameFilePaths,
                                          loadFramesIntoNextBufferMutex);

    // gathers the current buffer of frames
    while (currentVideoFrameBuffer.empty()) {
        loadFramesIntoPreviousBufferMutex->lock();
        currentVideoFrameBuffer = previousVideoFrameBuffer;
        loadFramesIntoPreviousBufferMutex->unlock();
    }

    // keeps on showing the video frames, until 'q' is pressed
    // (it will clear frameFilePaths)
    while (!frameFilePaths->empty()) {
        Mat currentFrame;

        if (currentVideoFrameNumber >= 0
            && currentVideoFrameNumber < frameFilePaths->size()) {
            currentVideoFrameBuffer.at(
                    currentVideoFrameNumber % VIDEO_FRAME_BUFFERS_SIZE).copyTo(
                    currentFrame);

            // treats possible changes in the current frame label
            if (currentLabel == 0 && overwriteLabels) {
                positiveFrames->erase(currentVideoFrameNumber);
                negativeFrames->insert(currentVideoFrameNumber);
            } else if (currentLabel == 1 && overwriteLabels) {
                positiveFrames->insert(currentVideoFrameNumber);
                negativeFrames->erase(currentVideoFrameNumber);
            }

            // prepares the current frame to be rendered
            prepareToRenderFrameStatus(&currentFrame, currentVideoFrameNumber,
                                       frameFilePaths->size() - 1, videoShowingDelay, playReverse,
                                       overwriteLabels, currentLabel, positiveFrames,
                                       negativeFrames);

            // increases the current frame number
            // and prepares the buffers, if it is the case
            if (videoShowingDelay > 0) {
                // if the video is being played not reversed
                if (!playReverse
                    && currentVideoFrameNumber < frameFilePaths->size() - 1
                    && !nextVideoFrameBuffer.empty()) {
                    // next frame
                    currentVideoFrameNumber++;

                    // if out of buffer
                    // (i.e. buffers need to be updated)
                    if (currentVideoFrameNumber
                        >= refCurrentBufferFrameNumber
                           + VIDEO_FRAME_BUFFERS_SIZE) {
                        loadFramesIntoNextBufferMutex->lock();
                        loadFramesIntoPreviousBufferMutex->lock();

                        // the previous buffer is the current one
                        previousVideoFrameBuffer.clear();
                        previousVideoFrameBuffer = currentVideoFrameBuffer;

                        // the current buffer is the next one
                        currentVideoFrameBuffer = nextVideoFrameBuffer;

                        // restarts the next buffer
                        nextVideoFrameBuffer = vector<Mat>();

                        // updates the reference number of the 1st frame inside the current buffer
                        if (refCurrentBufferFrameNumber
                            + VIDEO_FRAME_BUFFERS_SIZE
                            < frameFilePaths->size())
                            refCurrentBufferFrameNumber =
                                    refCurrentBufferFrameNumber
                                    + VIDEO_FRAME_BUFFERS_SIZE;
                        // else maintains the number

                        loadFramesIntoNextBufferMutex->unlock();
                        loadFramesIntoPreviousBufferMutex->unlock();
                    }
                }

                    // else, the video is being played reversed
                else if (playReverse && currentVideoFrameNumber > 0
                         && !previousVideoFrameBuffer.empty()) {
                    // previous frame
                    currentVideoFrameNumber--;

                    // if out of buffer
                    if (currentVideoFrameNumber < refCurrentBufferFrameNumber) {
                        loadFramesIntoNextBufferMutex->lock();
                        loadFramesIntoPreviousBufferMutex->lock();

                        // the next buffer is the current one
                        nextVideoFrameBuffer.clear();
                        nextVideoFrameBuffer = currentVideoFrameBuffer;

                        // the current buffer is the previous one
                        currentVideoFrameBuffer = previousVideoFrameBuffer;

                        // restarts the previous buffer
                        previousVideoFrameBuffer = vector<Mat>();

                        // updates the reference number of the 1st frame inside the current buffer
                        if (refCurrentBufferFrameNumber
                            - VIDEO_FRAME_BUFFERS_SIZE >= 0)
                            refCurrentBufferFrameNumber =
                                    refCurrentBufferFrameNumber
                                    - VIDEO_FRAME_BUFFERS_SIZE;
                        // else maintains the number

                        loadFramesIntoNextBufferMutex->unlock();
                        loadFramesIntoPreviousBufferMutex->unlock();
                    }
                }
            }
        }

        // shows the current frame
        namedWindow("Frame Labeler", WINDOW_AUTOSIZE);
        imshow("Frame Labeler", currentFrame);
        char key = waitKey(videoShowingDelay);

        // treats an eventual pressed key
        treatKeyboardInput(key, &currentVideoFrameNumber,
                           &refCurrentBufferFrameNumber, &videoShowingDelay, &playReverse,
                           &overwriteLabels, &currentLabel, frameFilePaths,
                           &currentVideoFrameBuffer, &previousVideoFrameBuffer,
                           &nextVideoFrameBuffer, positiveFrames, negativeFrames,
                           loadFramesIntoPreviousBufferMutex,
                           loadFramesIntoNextBufferMutex);
    }

    // frees some memory
    nextBufferThread->join();
    previousBufferThread->join();
    delete nextBufferThread;
    delete previousBufferThread;
    delete loadFramesIntoNextBufferMutex;
    delete loadFramesIntoPreviousBufferMutex;
}

/** Generates and saves the ETF file in the given path <etfFilePath>.
 *
 *  Parameter <event> is a string containing the name of annotated event.
 *
 *  Parameter <videoFPS> defines the frame rate of the annotated video.
 *
 *  Parameter <videoFileName> contains the file name of the annotated video.
 *
 *  Parameter <totalFramesCount> contains the total number of frames extracted
 *  from the annotated video.
 *
 *  Parameters <positiveFrames> and <negativeFrames> are sets containing the
 *  numbers of the positive and of the negative annotated frames, respectively.
 *
 *  ETF file: format created within the MediaEval (https://multimediaeval.github.io/ violent scenes loc. task. */
void generateAndSaveETFFile(string etfFilePath, string event, double videoFPS,
                            string videoFileName, int totalFramesCount, set<int> *positiveFrames,
                            set<int> *negativeFrames) {
    // turns the sets into vectors
    vector<int> positiveFramesVector, negativeFramesVector;
    if (!positiveFrames->empty())
        positiveFramesVector = vector<int>(positiveFrames->begin(),
                                           positiveFrames->end());
    if (!negativeFrames->empty())
        negativeFramesVector = vector<int>(negativeFrames->begin(),
                                           negativeFrames->end());

    // holds the time marks
    vector<int> videoTimeMarks;
    videoTimeMarks.push_back(0);
    bool beginsNegative;

    // there are only negative marks
    if (positiveFramesVector.empty() && !negativeFramesVector.empty())
        beginsNegative = true;

        // there are only positive marks
    else if (!positiveFramesVector.empty() && negativeFramesVector.empty())
        beginsNegative = false;

        // there are negative and positive marks
    else {
        if (negativeFramesVector.front() < positiveFramesVector.front()) {
            beginsNegative = true;

            videoTimeMarks.push_back(positiveFramesVector.front());
            int lastPositive = positiveFramesVector.front();

            for (int i = 1; i < positiveFramesVector.size(); i++) {
                if (lastPositive + 1 != positiveFramesVector.at(i)) {
                    videoTimeMarks.push_back(
                            positiveFramesVector.at(i - 1) + 1);
                    videoTimeMarks.push_back(positiveFramesVector.at(i));
                }

                lastPositive = positiveFramesVector.at(i);
            }

            videoTimeMarks.push_back(positiveFramesVector.back() + 1);
        } else {
            beginsNegative = false;

            videoTimeMarks.push_back(negativeFramesVector.front());
            int lastNegative = negativeFramesVector.front();

            for (int i = 1; i < negativeFramesVector.size(); i++) {
                if (lastNegative + 1 != negativeFramesVector.at(i)) {
                    videoTimeMarks.push_back(
                            negativeFramesVector.at(i - 1) + 1);
                    videoTimeMarks.push_back(negativeFramesVector.at(i));
                }

                lastNegative = negativeFramesVector.at(i);
            }

            videoTimeMarks.push_back(negativeFramesVector.back() + 1);
        }
    }

    // last video frame
    videoTimeMarks.push_back(totalFramesCount);

    // ETF file writer
    ofstream etfFileWriter(etfFilePath.data());
    if (etfFileWriter.fail()) {
        cerr << "Could not write file " << etfFilePath << "." << endl;
        throw -1;
    }

    for (int i = 0; i < videoTimeMarks.size() - 1; i++) {
        double time = videoTimeMarks.at(i) / videoFPS;
        double duration = (videoTimeMarks.at(i + 1) / videoFPS) - time;

        if (duration > 0) {
            etfFileWriter << videoFileName << " 1 " << time << " " << duration
                          << " event - " << event << " - "
                          << (beginsNegative ? 'f' : 't') << endl;
        }

        beginsNegative = !beginsNegative;
    }

    // adds the numbers of the positive frames to the ETF file as comments
    // (a little help to non-ETF format enthusiasts)
    if (!positiveFramesVector.empty()) {
        etfFileWriter << "# positive frames" << endl;
        for (int i = 0; i < positiveFramesVector.size(); i++)
            etfFileWriter << "# " << positiveFramesVector.at(i) << endl;
    }

    // closes the ETF file
    etfFileWriter.close();
}

/** Annotates a given video as entirely negative.
 *
 *  Parameter <etfFilePath> refers to the path of ETF file output as annotation.
 *  Parameter <event> is a string defining the event being annotated as negative. */
void annotateEntireVideoAsNegative(string videoFilePath, string etfFilePath,
                                   string event) {
    // obtains the frame rate, with the help of ffprobe
    double frameRate = 0;
    {
        stringstream shellScript;
        shellScript << "ffprobe -i " << videoFilePath
                    << " -v quiet -show_streams -select_streams v 2>&1 | grep 'avg_frame_rate=' | cut -c16-";

        FILE *pipe = popen(shellScript.str().data(), "r");
        if (!pipe) {
            cerr << "WARNING: Could not obtain video frame rate, with FFprobe."
                 << endl;
            throw -1;
        } else {
            char buffer[128];
            string result = "";

            while (!feof(pipe))
                if (fgets(buffer, 128, pipe) != NULL)
                    result += buffer;
            pclose(pipe);

            vector <string> frameRateTokens;
            split(frameRateTokens, result, is_any_of("/"));

            double numerator, denominator;
            numerator = atof(frameRateTokens.front().data());
            denominator = atof(frameRateTokens.back().data());

            frameRate = numerator / denominator;
        }
    }

    // obtains the total number of frames, with the help of ffprobe
    int frameCount = 0;
    {
        stringstream shellScript;
        shellScript << "ffprobe -i " << videoFilePath
                    << " -v quiet -show_streams -select_streams v 2>&1 | grep 'nb_frames=' | cut -c11-";

        FILE *pipe = popen(shellScript.str().data(), "r");
        if (!pipe) {
            cerr << "WARNING: Couldn't obtain video frame count, with FFprobe."
                 << endl;
            throw -2;
        } else {
            char buffer[128];
            string result = "";

            while (!feof(pipe))
                if (fgets(buffer, 128, pipe) != NULL)
                    result += buffer;
            pclose(pipe);

            frameCount = atoi(result.data());
        }
    }

    // calculates the duration of the video
    double duration = frameCount / frameRate;

    // obtains the video file name
    string videoFileName;

    vector <string> tokens;
    split(tokens, videoFilePath, is_any_of("/"));
    videoFileName = tokens.back();
    tokens.clear();

    // generates the ETF output file
    ofstream etfFileWriter(etfFilePath.data());
    if (etfFileWriter.fail()) {
        cerr << "Could not write file " << etfFilePath << "." << endl;
        throw -3;
    }

    etfFileWriter << videoFileName << " 1 0 " << duration << " event - "
                  << event << " - f" << endl;
    etfFileWriter.close();
}

/** Individually extracts the frames from the videos refereed by the given list of video
 *  file paths. It is recommended for the videos to be in H.264 MPEG-4 format. The frames
 *  are output as JPG images, named with their video file name + frame number + a sequence
 *  number (from 00000001 to N). The size of the saved frames can be informed as a new
 *  desired total number of pixels per frame, or 0 if the original size shall be maintained.
 *  If the new desired total number of pixels is greater than the original one, the sizes of
 *  the frames are simply maintained. The number of threads to let run simultaneously when
 *  extracting the frames must also be informed. */
void runVideoFrameExtraction(vector <string> *videoFilePaths,
                             string frameDirPath, int totalPixelCount, int simThreadCount) {
    // time register
    cout << "Begin time: " << getCurrentDateTime() << endl;

    // holds the number of treated video files
    int filesCount = 0;

    // for each video file path
    for (int i = 0; i < videoFilePaths->size(); i = i + simThreadCount) {
        // current group of up to <simThreadCount> threads
        std::vector <std::thread> descriptionThreadGroup;

        for (int j = 0; j < simThreadCount; j++) {
            if (i + j < videoFilePaths->size()) {
                // file path of the current video
                string currentVideoFilePath = videoFilePaths->at(i + j);

                // thread creation to extract the chosen frames from the current video
                descriptionThreadGroup.emplace_back(
                        extractAndSaveVideoFrames,
                        currentVideoFilePath, frameDirPath, totalPixelCount
                );

                // counts one more treated file
                filesCount++;
            } else
                break;
        }

        // executes the current thread group
        for (auto &thread: descriptionThreadGroup)
            if (thread.joinable())
                thread.join();

        // logging
        cout << "Progress: treated files " << filesCount << "/"
             << videoFilePaths->size() << "." << endl;
    }

    // time register
    cout << "End time: " << getCurrentDateTime() << endl;
}

/** Executes the interface to support the annotation of a given list of frames,
 *  related to a target video.
 *
 *  The frames are determined by a file containing their file paths, one per line.
 *  The path of such input file must be in <inputFilePath>.
 *
 *  Parameter <videoFPS> defines the frame rate of the target video being annotated.
 *
 *  Parameter <inputETFFilePath> is the file path of a previous annotation of the
 *  target video. Please give NULL is none was done.
 *
 *  Parameter <outputETFFilePath> is the file path of the new annotation of the target
 *  video.
 *
 *  Parameter <event> is a string defining the event being annotated. */
void runVideoAnnotationSupport(string inputFilePath, double videoFPS,
                               string *inputETFFilePath, string outputETFFilePath, string event) {
    // begin time
    cout << "Begin time: " << getCurrentDateTime() << endl;

    // obtains a list with the file paths to the frames of the video to be annotated
    vector <string> frameFilePaths;
    readFrameFilePaths(inputFilePath, &frameFilePaths);

    // sets with the positive and negative frames of the video
    set<int> positiveFrames, negativeFrames;

    // obtains the video file name
    string videoFileName;

    vector <string> tokens;
    split(tokens, frameFilePaths.front(), is_any_of("/"));
    videoFileName = tokens.back();
    tokens.clear();

    split(tokens, videoFileName, is_any_of("-"));
    videoFileName = tokens.front();
    tokens.clear();

    // holds the total number of frames
    int totalFramesCount = frameFilePaths.size();

    // reads the eventual input ETF file
    if (inputETFFilePath != NULL) {
        // obtains the name of the video file to be annotated
        string parsedVideoFileName;

        vector <string> tokens;
        split(tokens, frameFilePaths.front(), is_any_of("/"));
        parsedVideoFileName = tokens.back();
        tokens.clear();

        split(tokens, parsedVideoFileName, is_any_of("-"));
        parsedVideoFileName = tokens.front();
        tokens.clear();

        readInputETFFile(parsedVideoFileName, videoFPS, *inputETFFilePath,
                         &positiveFrames, &negativeFrames);
    }

        // else, all the frames are negative
    else
        for (int i = 0; i < frameFilePaths.size(); i++)
            negativeFrames.insert(i);

    // shows the video content, with annotation support
    showVideoFrames(&frameFilePaths, &positiveFrames, &negativeFrames);

    // generates and saves the ETF file
    cout << "Saving ETF file at path: " << outputETFFilePath << endl;
    generateAndSaveETFFile(outputETFFilePath, event, videoFPS, videoFileName,
                           totalFramesCount, &positiveFrames, &negativeFrames);

    // end time
    cout << "End time: " << getCurrentDateTime() << endl;
}

/** Annotates the videos listed in <videoFilePaths> as entirely negative, with respect to
 *  a given event, by the means of its string name <event>.
 *
 *  Parameter <etfDirPath> is the directory path of the ETF files generated, one for each
 *  given video file. */
void runVideoAnnotationAsNegative(vector <string> *videoFilePaths, string event,
                                  string etfDirPath) {
    // tries to open the given directory path to store the extracted frames
    DIR *pDir;
    pDir = opendir(etfDirPath.data());
    if (pDir == NULL)
        // tries to create the directory
        mkdir(etfDirPath.data(), 0777);

    pDir = opendir(etfDirPath.data());
    if (pDir == NULL) {
        cerr << "Could not open neither create directory " << etfDirPath << "."
             << endl;
        throw -1;
    }
    closedir(pDir);

    // time register
    cout << "Begin time: " << getCurrentDateTime() << endl;

    // for each video file
    for (int i = 0; i < videoFilePaths->size(); i++) {
        // obtains the current video file path
        string currentVideoFilePath = videoFilePaths->at(i);

        // defines the name of the current ETF file
        vector <string> currentVideoFilePathTokens;
        split(currentVideoFilePathTokens, currentVideoFilePath, is_any_of("/"));
        string currentVideoFileName = currentVideoFilePathTokens.back();
        currentVideoFilePathTokens.clear();

        stringstream currentETFFilePathStream;
        currentETFFilePathStream << etfDirPath << "/" << currentVideoFileName
                                 << ".etf";
        string currentETFFilePath = currentETFFilePathStream.str();

        // annotates the current video as entirely negative
        annotateEntireVideoAsNegative(currentVideoFileName, currentETFFilePath,
                                      event);

        // logging
        cout << "Progress: treated file " << (i + 1) << "/"
             << videoFilePaths->size() << "." << endl;
    }

    // time register
    cout << "End time: " << getCurrentDateTime() << endl;
}

/** Turns this singleton into an executable file. */
int main(int paramCount, char **params) {
    cout << "*** FrameLabeler Execution. *** " << endl;

    // main parameters
    int mode = -1;
    try {
        // if there are no given arguments, raises an error
        if (paramCount <= 1)
            throw -1;

        // mode argument capture
        stringstream modeStream;
        modeStream << params[1];
        modeStream >> mode;
        if (mode != 0 && mode != 1 && mode != 2)
            throw -2;

        // mode to extract video frames
        if (mode == 0) {
            // parameters
            string videoListFilePath = "";  // -i parameter
            string frameDirPath = "";    // -f parameter
            int totalPixelCount = 0;        // -p parameter
            int simThreadCount = 1;            // -t parameter

            try {
                if (paramCount <= 2)
                    throw -3;

                // gathering of parameters
                for (int i = 2; i < paramCount; i = i + 2) {
                    stringstream currentParameterStream;
                    currentParameterStream << params[i] << params[i + 1];

                    char parameterType;
                    currentParameterStream >> parameterType >> parameterType;

                    switch (parameterType) {
                        case 'i':
                            currentParameterStream >> videoListFilePath;
                            if (videoListFilePath.length() <= 0) {
                                cerr << "Please verify the -i parameter." << endl;
                                throw -4;
                            }
                            break;

                        case 'f':
                            currentParameterStream >> frameDirPath;
                            if (frameDirPath.length() <= 0) {
                                cerr << "Please verify the -f parameter." << endl;
                                throw -5;
                            }
                            break;

                        case 'p':
                            totalPixelCount = -1; // invalid value
                            currentParameterStream >> totalPixelCount;
                            if (totalPixelCount < 0) {
                                cerr
                                        << "The -p parameter must be equal or greater than ZERO."
                                        << endl;
                                throw -6;
                            }
                            break;

                        case 't':
                            simThreadCount = 0; // invalid value
                            currentParameterStream >> simThreadCount;
                            if (simThreadCount < 1) {
                                cerr
                                        << "The -t parameter must be equal or greater than ONE."
                                        << endl;
                                throw -7;
                            }
                            break;

                        default:
                            throw -8;
                    }
                }

                // treatment of mandatory parameters
                if (videoListFilePath.length() <= 0) {
                    cerr << "Please verify the -i parameter." << endl;
                    throw -4;
                } else if (frameDirPath.length() <= 0) {
                    cerr << "Please verify the -f parameter." << endl;
                    throw -5;
                }

                // logging the parameters, if they are ok
                cout << "Parameters:" << endl << " <mode>: " << mode << endl
                     << " -i: " << videoListFilePath << endl << " -f: "
                     << frameDirPath << endl << " -p: " << totalPixelCount
                     << endl << " -t: " << simThreadCount << endl;
            } catch (int e) {
                cerr
                        << "Usage (with option parameters in any order): framelabeler 0"
                        << endl << " -i video_list_file_path" << endl
                        << " -f saved_frames_dir_path" << endl
                        << " -p total_pixel_count (get 0, maintain: 0, default: 0)"
                        << endl << " -t sim_thread_count (get 1, default: 1)"
                        << endl;
                return 10 * e;
            }

            // parameters are ok...
            // tries to obtain the file names of the videos
            vector <string> videoFilePaths;
            try {
                readVideoFilePathList(videoListFilePath, &videoFilePaths);
            } catch (int e) {
                cerr << "Could not obtain the paths to the video files." << endl;
                return 100 * e;
            }

            // frame extraction
            try {
                runVideoFrameExtraction(&videoFilePaths, frameDirPath,
                                        totalPixelCount, simThreadCount);
            } catch (int e) {
                cerr << "Could not read extract videos frames." << endl;
                return 1000 * e;
            }
        }

            // mode to annotate video frames
        else if (mode == 1) {
            string inputFilePath = "";     // -i parameter
            double videoFPS = 25.0;        // -f parameter
            string inputETFFilePath = "";  // -g parameter
            string event = "violence";      // -e parameter
            string outputETFFilePath = ""; // -o parameter

            try {
                if (paramCount <= 2)
                    throw -3;

                // gathering of parameters
                for (int i = 2; i < paramCount; i = i + 2) {
                    stringstream currentParameterStream;
                    currentParameterStream << params[i] << params[i + 1];

                    char parameterType;
                    currentParameterStream >> parameterType >> parameterType;

                    switch (parameterType) {
                        case 'i':
                            currentParameterStream >> inputFilePath;
                            if (inputFilePath.length() <= 0) {
                                cerr << "Please verify the -i parameter." << endl;
                                throw -4;
                            }
                            break;

                        case 'f':
                            videoFPS = 0; // invalid value
                            currentParameterStream >> videoFPS;
                            if (videoFPS <= 0) {
                                cerr
                                        << "The -f parameter must be greater than ZERO."
                                        << endl;
                                throw -5;
                            }
                            break;

                        case 'g':
                            currentParameterStream >> inputETFFilePath;
                            if (inputETFFilePath.length() <= 0) {
                                cerr << "Please verify the -g parameter." << endl;
                                throw -6;
                            }
                            break;

                        case 'e':
                            event = ""; // invalid value
                            currentParameterStream >> event;
                            if (event.length() <= 0) {
                                cerr << "Please verify the -e parameter." << endl;
                                throw -7;
                            }
                            break;

                        case 'o':
                            currentParameterStream >> outputETFFilePath;
                            if (outputETFFilePath.length() <= 0) {
                                cerr << "Please verify the -o parameter." << endl;
                                throw -8;
                            }
                            break;

                        default:
                            throw -9;
                    }
                }

                // treatment of mandatory parameters
                if (inputFilePath.length() <= 0) {
                    cerr << "Please verify the -i parameter." << endl;
                    throw -4;
                } else if (event.length() <= 0) {
                    cerr << "Please verify the -e parameter." << endl;
                    throw -7;
                } else if (outputETFFilePath.length() <= 0) {
                    cerr << "Please verify the -o parameter." << endl;
                    throw -8;
                }

                // logging the parameters, if they are ok
                cout << "Parameters:" << endl << " <mode>: " << mode << endl
                     << " -i: " << inputFilePath << endl << " -f: "
                     << videoFPS << endl << " -g: "
                     << (inputETFFilePath.length() <= 0 ?
                         "none" : inputETFFilePath) << endl << " -e: "
                     << event << endl << " -o: " << outputETFFilePath
                     << endl;
            } catch (int e) {
                cerr
                        << "Usage (with option parameters in any order): framelabeler 1"
                        << endl << " -i input_file_path_with_frame_file_paths"
                        << endl << " -f video_fps (gt 0, default: 25.0)" << endl
                        << " -g input_etf_file_path" << endl
                        << " -e event (string, default: violence)" << endl
                        << " -o output_etf_file_path" << endl;
                return 10 * e;
            }

            // parameters are ok...
            try {
                runVideoAnnotationSupport(inputFilePath, videoFPS,
                                          (inputETFFilePath.length() <= 0 ?
                                           NULL : &inputETFFilePath), outputETFFilePath,
                                          event);
            } catch (int e) {
                cerr << "Could not annotate videos." << endl;
                return 100 * e;
            }
        }

            // else, mode to annotate files as entirely negative
        else {
            // parameters
            string videoListFilePath = "";  // -i parameter
            string etfDirPath = "";        // -o parameter
            string event = "violence";       // -e parameter

            try {
                if (paramCount <= 2)
                    throw -3;

                // gathering of parameters
                for (int i = 2; i < paramCount; i = i + 2) {
                    stringstream currentParameterStream;
                    currentParameterStream << params[i] << params[i + 1];

                    char parameterType;
                    currentParameterStream >> parameterType >> parameterType;

                    switch (parameterType) {
                        case 'i':
                            currentParameterStream >> videoListFilePath;
                            if (videoListFilePath.length() <= 0) {
                                cerr << "Please verify the -i parameter." << endl;
                                throw -3;
                            }
                            break;

                        case 'o':
                            currentParameterStream >> etfDirPath;
                            if (etfDirPath.length() <= 0) {
                                cerr << "Please verify the -o parameter." << endl;
                                throw -4;
                            }
                            break;

                        case 'e':
                            event = ""; // invalid value
                            currentParameterStream >> event;
                            if (event.length() <= 0) {
                                cerr << "Please verify the -e parameter." << endl;
                                throw -5;
                            }
                            break;

                        default:
                            throw -6;
                    }
                }

                // treatment of mandatory parameters
                if (videoListFilePath.length() <= 0) {
                    cerr << "Please verify the -i parameter." << endl;
                    throw -4;
                } else if (etfDirPath.length() <= 0) {
                    cerr << "Please verify the -o parameter." << endl;
                    throw -5;
                } else if (event.length() <= 0) {
                    cerr << "Please verify the -e parameter." << endl;
                    throw -6;
                }

                // logging the parameters, if they are ok
                cout << "Parameters:" << endl << " <mode>: " << mode << endl
                     << " -i: " << videoListFilePath << endl << " -o: "
                     << etfDirPath << endl << " -e: " << event << endl;
            } catch (int e) {
                cerr
                        << "Usage (with option parameters in any order): framelabeler 2"
                        << endl << " -i video_list_file_path" << endl
                        << " -o output_etf_dir_path" << endl
                        << " -e event (string, default: violence)" << endl;
                return 10 * e;
            }

            // parameters are ok...
            // tries to obtain the file names of the videos
            vector <string> videoFilePaths;
            try {
                readVideoFilePathList(videoListFilePath, &videoFilePaths);
            } catch (int e) {
                cerr << "Could not obtain the paths to the video files."
                     << endl;
                return 100 * e;
            }

            // annotates the file
            try {
                runVideoAnnotationAsNegative(&videoFilePaths, event,
                                             etfDirPath);
            } catch (int e) {
                cerr << "Could not annotate videos." << endl;
                return 100 * e;
            }

        }
    } catch (int e) {
        cerr
                << "Usage: frame_labeler <mode (extract frames: 0 | annotate frames: 1 | annotate negative videos: 2)>"
                << endl;
        return e;
    }

    // everything went ok
    cout << "*** Acabou! *** " << endl;
    return 0;
}
