# Python-Mini-Coin-Counter-Aruco-Detector-Image-Unwarper
A small Python Program which uses OpenCV, PIL and Tkinter to unwarp an image, detect the coins in it, tries too gauge their values and detects an aruco marker. The program was created in July 2023 as part of university coursework. Feel free to play around with the different settings constants.

## Usage
To use the program first make sure that you have python installed. The easiest way to download python would be to download it from the Microsoft Store if you're on WindowsOS. The program was made
using Python 3.11.4. Also make sure to have the python packages **openCV** (ver. 4.6.0), **PIL (pillow)** (ver. 10.0.0), **tk (tkinter)** (ver. 8.6.12) and **np (numpy)** (ver. 1.25.2) installed. Other versions of python or of the aforementioned packages may or may not work.
To install the necessary packages, simply open cmd.exe and type "**pip install tk pillow numpy opencv-python**". The necessary packages will then be downloaded and installed. Alternatively you can download the necessary packages using a package manager of your choice like Anaconda.
You can either run the program by using the command line or by running it using an IDE like Pycharm. To use the program via the command line (e.g. cmd.exe), navigate to the 
folder where the "**Python-Mini-Coin-Counter-Aruco-Detector-Image-Unwarper.py**" script is located by typing "**cd [Absolute Path to folder in which Python-Mini-Coin-Counter-Aruco-Detector-Image-Unwarper.py is located]**". 
After you've done that, type "**Python-Mini-Coin-Counter-Aruco-Detector-Image-Unwarper.py**". To pass an image to the program either select it via the file explorer by running the script as described. Alternatively you can provide an image path via the --image_path command line argument or provide an image path via the code or even take a webcam snapshot. To do this modify the constants under "**# Miscellaneous constants #**". You can now use the program to unwarp the image, detect coins in the image, gauge their values and detect an aruco marker.

### Showcase

<table>
  <tr>
    <td><kbd> <img src="Python-Mini-Coin-Counter-Aruco-Detector-Image-Unwarper Showcase 1.png" width="500" /> </kbd></td>
    <td><kbd> <img src="Python-Mini-Coin-Counter-Aruco-Detector-Image-Unwarper Showcase 2.png" width="500" /> </kbd></td>
  </tr>
</table>
