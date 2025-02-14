
## Tips for setting up

1. Python projects normally have a `requirements.txt` file with all the packages you need to install. To install the ones we currently picked for you, use:
    ```
    pip install -r requirements.txt
    ```
   Feel free to add more packages to the `requirements.txt` file, but make sure to ask your TA if you are allowed to use them!

2. It is good practice to use a Virtual Environment such that you can easily switch between Python projects (projects usually have different requirements, and some of them don't work well together). Some commonly used ones are 
    - `venv` (Built-in)  
    ```
    python -m venv myenv
    source myenv/bin/activate  # Linux/Mac
    myenv\Scripts\activate     # Windows
    ```
    - or `conda` (Anaconda Prompt)
    ```
    conda create --name myenv python=3.9
    conda activate myenv
    conda install <package_name_here>
    pip install -r requirements.txt
    ```

3. Help: `cv2 package not found`!   
Try installing `opencv-python`, so `pip install opencv-python`.

4. Help: `ffmpeg error` (even though we don't explicitly make use of ffmpeg, you might get this error when running your project because opencv uses it in the backend). Check if ffmpeg is installed with `ffmpeg -version`, if not, install it:
   - Windows: Download from [official website](https://ffmpeg.org/download.html) and add it to the PATH variable.
   - Linux: `sudo apt install ffmpeg`
   - Mac: `brew install ffmpeg`


