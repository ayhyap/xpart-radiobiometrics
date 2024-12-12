# Cross-part radiobiometrics

Code repository for paper under review.

# 1. Environment setup
To reproduce, use Anaconda.
Download repository, create conda environment with environment.yml
```
git clone https://github.com/ayhyap/xpart-radiobiometrics.git
cd xpart-radiobiometrics
conda env create -f environment.yml
conda activate xpart
```
Set data paths in 'globals.json' to the relevant paths on your system.
"image_dir" is the top directory containing your images. Images can be in subfolders as long as the path them is properly defined in the metadata csv.
"metadata_csv" is a csv containing information about each image. See "3. Metadata processing" below.

# 2. Image processing
To reproduce the training process, data from OAI is required.
OAI is hosted on the NIMH Data Archive (NDA), but is unavailable at time of writing.
Due to data use agreements, we are not permitted to share this data either.

Assuming you have access to OAI imaging data, the dicom thumbnail jpg files (*_1x1.jpg) are usable BUT were improperly windowed which resulted in loss of quality.
Ideally, one should use the original dicom images for original quality.
Our preprocessing script extracts the x-ray images from the data archives, performs resizing, naiive cropping, and histogram equalization.
For ease of use, the main image processing steps have been incorporated in the script so preprocessing the images is not strictly necessary.
Refer to image_preprocessing.py and dataset.py for details.

During or after this process you should be able to build a list of image file names which will be necessary for the next step.

# 3. Metadata processing
Running the training script (main.py) requires a csv file containing the following columns:
- src_subject_id: the ID of the subject in the image
- file: the file name of the image as saved on your system. Can be in subdirectories as long as you supply the correct top-level directory in globals.json
- exam: the body part imaged. This is hard-coded to be one of {'AP Pelvis','Bilateral PA Fixed Flexion Knee','PA Bilateral Hand','PA Right Hand'}, case-sensitive.
- xval_split: the cross-validation split of the image. In our case we perform 4-fold cross validation, testing folds 0-3 inclusive on images marked with the corresponding split number.

Performing subgroup analysis requires these additional columns:
- interview_age: the age of the imaged subject in years
- sex: the gender of the imaged subject
- race: the race of the image subject, mapped to integer values: 0 (other), 1 (white/caucasian), 2 (black/african american), 3 (asian), or '' (blank) if not provided

Performing linear probe analysis requires all of the above columns, and the following:
- KL_max: the maximum KL grade recorded for a Knee x-ray, between the two knees. Ranges from 0 to 4, blank if not available or invalid.
- knee_prosthesis: the presence of any visible prosthetic in the knee x-ray. 0 or 1, blank if invalid.
- pelvis_prosthesis: the presence of any visible prosthetic in the pelvis x-ray. 0 or 1, blank if invalid.

Repeating what was said in the previous step, to reproduce the training process, data from OAI is required.
OAI is hosted on the NIMH Data Archive (NDA), but is unavailable at time of writing.
Due to data use agreements, we are not permitted to share this data either.

The following section assumes you already have the entire OAI dataset.
To get the list of images, you'll want the metadata file named 'Image' or 'image03' under Image Details of the OAI release in the NDA query tool.
From this file, you can filter for x-ray images under the 'scan_type' column, then use the image_file column as a reference to build your list.

If you do not have this file, but have all of the images available, it is also possible to build a list from the XRay**.txt files in the "Complete OAI Dataset" from the OAI Full Data Downloads page, and the enrolee**.txt which comes with the image archives.

You'll also want the enrollee demographics file 'oai_enrollee01' under demographics and x-ray outcomes also in NDA.

Now create a new csv file which contains information for each x-ray image with the following columns (using these names as headers):
- src_subject_id: unchanged from image03, XRay**.txt, or enrolee**.txt
- interview_age: the interview_age from image03 or enrolee**.txt, divided by 12 and floored to give age in years
- sex: the gender column from image03 or enrolee**.txt
- race: the race column from enrolee**.txt joined using src_subject_id as a key, mapped to integer values: 0 (other), 1 (white/caucasian), 2 (black/african american), 3 (asian), or '' (blank) if not provided
- file: the file name of the image as saved on your system. Can be in subdirectories as long as you supply the correct top-level directory in globals.json
- exam: the image_description column from image03 or the V**EXAMTP from XRay**.txt
- xval_split: the src_subject_id column modulo 4
- KL_max: the max value from the KL column for the knee radiograph in kxr_sq_bu**.txt, blank if unavailable or invalid
- knee_prosthesis: from manual inspection: 1 if present, 0 if not, blank if not knee radiograph
- pelvis_prosthesis: from manual inspection: 1 if present, 0 if not, blank if not pelvis radiograph

# 4. Running the training script
To reproduce the training procedure, run main.py
```
python main.py [hyperparameters].json [device (cuda)] [cross-validation fold(s)]
python main.py runs/baseline.json 0 0123
```
Obviously use a GPU, otherwise it will take forever.
The training process was done using a 16GB GPU, so lower the batch size in the hyperparameter json if necessary.
The command above runs all 4 cross-validation splits in sequence.
To run on individual cross-validation folds, supply only a single number in the 3rd argument like:
```
python main.py runs/baseline.json 0 2
```

# 5. Running the evaluation script
Although the training script does regular validations and tests when training is over, full testing including subset analyses and linear probes are done separately.
For full evaluation, all required columns must be present in the metadata csv. See '3. metadata processing' above.
Run full_eval.py
```
python full_eval.py runs/baseline.json 0
```
This will also output csv files for ROCs in the corresponding subfolder, and also print some results to terminal.

# 6. Running the model on your own images

WORK IN PROGRESS.
