### Gut Unraveling Tool System (GUTS)
#### A Gut Fullness Assessment Tool for Noctiluca Scintillans

GUTS is a Python based tool that is being developed in order to asses the Gut Fullness of Noctiluca scintillans individuals through image analysis.
Noctiluca scintillans is a cosmopolitan Dinoflagellate zooplankton species with a relatively transparent body. This transparent body allows us to view what is inside of an individual when we take a picture of it. This tool aims to asses how much food is inside of an individual, or it's Gut Fullness, based on such pictures.


### Installation

In order to use GUTS, the entire filestructure needs to be downloaded from this github page. Once downloaded, it can be moved to a folder of your choosing on your PC.

```
GUTS/
  Data/
    Exports/
      contents_output/
      surface_output/
      classifier_output/
      gut_fullness_output/
    Imports/
      images/
        sample_pack/
          sample_images.png
        trained_segmentation_images/
          ground_truth/
            ground_truth_images.png
          noctiluca_sample/
            noctiluca_samples.png
  db_utils/
    set_paths.py
  ml_data/
    checkpoints/
      model_checkpoints
    keras_tuner/
    saved_models/
      saved_models
    tb_logs/
  plankton_cnn/
    pvnp_build.py
    pvnp_models.py
    pvnp_train.py
    pvnp_visualize.py
    visualization.py
    pvnp_import.py
    pvnp_save_and_load_utils.py
    pvnp_use.py
    run_classification_model.py
  Gut_fullness_algorithm.py
  GUTS.py
```

### Usage

In order to use the tool, simply move all images of Noctiluca scintillans you want to analyze into a new folder in the the "Data/Imports/" folder.
After the images have been moved to the imports folder, GUTS can be used. In order to apply guts to the images in the Imports folder, use the name of the new folder as the "run name".

### Credits
All the credits in the world go to Pieter Hovenkamp for developing the classifier used in the pre-classification aspect of the tool and helping me train the classification model.
