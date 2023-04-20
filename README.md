### Gut-Understanding Tool System (GUTS)
#### A Gut Fullness Assessment Tool for Noctiluca Scintillans

GUTS is a Python based tool that is being developed in order to asses the Gut Fullness of Noctiluca scintillans individuals through image analysis.
Noctiluca scintillans is a cosmopolitan Dinoflagellate zooplankton species with a relatively transparent body. This transparent body allows us to view what is inside of an individual when we take a picture of it. This tool aims to asses how much food is inside of an individual, or it's Gut Fullness, based on such pictures.


### Installation

In order to use GUTS, the entire filestructure needs to be downloaded from this github page. Once downloaded, it can be moved to a folder of your choosing on your PC. The folder in which you want to use the tool needs to have the following file structure:

```
Tool folder/
  data/
    input/
      image_library
      ...
    output/
      pre-classification
      output/gut_fullness
  GUTS_pre_classifier/
    GUTS_pre_classifier.py
  GUTS.py
```

### Usage

In order to use the tool, simply move all images of Noctiluca scintillans you want to analyze into the "image_library" folder.
After the images have been moved to the image library, the pre-classification step can be ran. Make sure to enter a memorable name in the prompt when starting the run.
Once the pre-classification has taken place, the gut fullness assessment can start, for which you need to supply the script with the filename you entered in the last script.
When the gut fullness of every applicable individual has been assessed, the gut fullness for each image can be found in a .csv file in the output folder.

### Credits
All the credits in the world go to Pieter Hovenkamp for developing the classifier used in the pre-classification aspect of the tool and helping me train the classification model.
