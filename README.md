# Video Frame Labeler

A software tool to extract and label video frames for binary classification.

## Features

The tool is executed from a command-line interface.
It runs in different "modes", depending on a given argument:

- Mode "0": video frame extraction in preparation for subsequent labeling.
- Mode "1": label visualization and quick annotation with keyboard shortcuts.
- Mode "2" (rare usage): quick annotation of all the video's frames as negative content.

The tool's input and output fulfill the following overall ideas:

- Typical input: text file containing a list of videos' or frames' file paths.
- Annotation output: files in the [MediaEval](https://multimediaeval.github.io/) *Violent Scenes Localization*
  competition's ETF format.

## Needed Software and Libraries

To compile and use the tool, one must install:

- [FFmpeg](https://ffmpeg.org/) (software only).
- [OpenCV](https://opencv.org/) (library with CPP headers).
- [Boost](https://www.boost.org/) (library with CPP headers).

## Compilation

To compile the tool, we recommend using [CMake](https://cmake.org/) in the command-line interface:

1. Check out the GitHub [master](https://github.com/danielmoreira/frame-labeler) branch.
    ```
    git clone https://github.com/danielmoreira/frame-labeler.git
    ```

2. Create a *build* folder.
    ```
    mkdir frame-labeler/build; cd frame-labeler/build
    ```

3. Run the *cmake* and *make* commands.
    ```
    cmake ..; make
    ```

4. Run the tool and follow the usage instructions.
   ```
    ./framelabeler
    ```

## Usage Examples

Shell scripts with usage examples over a single video are available
[here](https://github.com/danielmoreira/frame-labeler/tree/main/usage_examples).
They include:

1. A [frame extraction](https://github.com/danielmoreira/frame-labeler/blob/main/usage_examples/01_frame_extraction.sh)
   example.
2. A [frame labeling](https://github.com/danielmoreira/frame-labeler/blob/main/usage_examples/02_frame_labeling.sh)
   example.
3. A [label viewing](https://github.com/danielmoreira/frame-labeler/blob/main/usage_examples/03_label_viewing.sh)
   example.

## About

Developed by [Daniel Moreira](https://danielmoreira.github.io/), an assistant professor of Computer Science at
[Loyola University Chicago](https://www.luc.edu/).
