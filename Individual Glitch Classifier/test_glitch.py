import cv2
import numpy as np
import sys
import pickle
import importlib
import warnings
warnings.filterwarnings("ignore")


test_directory = "test.npy"
model_directory = ""



images = np.load(test_directory)
print(images.shape[0], " images were loaded.")
pred = np.zeros(len(images))

glitches = ["discoloration", "screen_tearing", "random_patch", "parallel_lines", 
            "shader", "square_patch", "texture_popin",
            "morse_code", "radial_dotted_line",
            "shape", "stuttering", "triangulation"]
#"line_pixelation",  "dotted_line",

gbl = globals()
for glitch in glitches:
    filename = model_directory + glitch + "_test." +  glitch + "_test"
    gbl[filename] = importlib.import_module(filename)
    #import str(filename)
    labels = gbl[filename].test(images)
    pred = np.logical_or(pred, labels)
    print('tested ', glitch)

np.save("test_result.npy", pred)
print("saved the result")
