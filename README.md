# Sign-Language-Translator
A Sign Language Translator, made using mediapipe.
Prerequisites: Mediapipe, Tensorflow, PyQt5
A Sign Language Translator inspired by the [Google - American Sign Language Fingerspelling Recognition Competition](https://www.kaggle.com/competitions/asl-fingerspelling).

Run the main.py file to run the app. Information on how to use it can be found within the app itself, in Help->How to use this application? 
The model used can also be trained from scratch, to do this, run trainModel.py. Otherwise, the app is also capable of improving the model over time, to use this feature
you must click on 'Yes' in the prompt, where it checks whether a response was accurate. Then navigate to Model->Update Model within the app, to retrain.

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
