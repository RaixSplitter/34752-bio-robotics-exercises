
# 34752 Bio-Inspired control for robots

Exercises for the course "Bio-Inspired control for robots" at the Technical University of Denmark.

## Getting started

> Note that it is assumed that you have downloaded the given "Required Tools" zip file and extracted it into the root of the directory. The directory comes with 2 essential local build packages, `exercise_camera_tools` and `fable_api_2`, which are required for the exercises.

1. Setup virtual environment

    ```bash
    python -m venv .venv
    ```

2. Activate the virtual environment

    - On Windows:

      ```bash
      .venv\Scripts\activate
      ```

    - On macOS/Linux:

      ```bash
      source .venv/bin/activate
      ```

3. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

4. Install local packages.

    ```bash
    pip install -e ./Required_Tools/additions/exercise_camera_tools
    pip install -e ./Required_Tools/additions/fable_api_2
    pip install -e .
    ```

## Usage

