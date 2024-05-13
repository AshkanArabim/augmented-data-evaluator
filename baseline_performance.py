import os, gc
import torch
import time

from sourcefiles.feature_extractor import FeatureExtractor
from sourcefiles.typical_atypical_classifier import Clip
from sourcefiles.typical_atypical_classifier import Participant
from sourcefiles.typical_atypical_classifier import TypicalClassifier


# start timer
t0 = time.time()


feature_extractor = FeatureExtractor()


def participant_path_to_participant(participant_clip_directory: str) -> Participant:
    levels = participant_clip_directory.split('/')
    participant_id = levels[-1]
    group_label = levels[-2]
    
    print(f"Starting feature extraction for participant {participant_id}...")
    clips_list = []
    for clip_filename in os.listdir(participant_clip_directory):
        clip_path = os.path.join(participant_clip_directory, clip_filename)
        print(f"Extracting features from: {clip_path}")
        # I have to move the tensors to CPU because the GPU runs out of space...
        # feel free to remove `to.("cpu")` if your gpu has > 12GB vram
        # These will be loaded after the model has been deallocated (depending on
        # selected `DEVICE` at the top of typical_atypical_classifier
        feature_avgs_all_layers = feature_extractor.get_features_averages_from_fp(clip_path).to("cpu")
        new_clip = Clip(feature_avgs_all_layers, clip_path)
        clips_list.append(new_clip)
    print(f"Feature extraction for participant {participant_id} done.")
    return Participant(participant_id=participant_id, group_label=group_label, clips=clips_list, age=0, gender="male")


participants = [participant_path_to_participant(path) for path in [
    os.path.join(root, dir) for root, dirs, files in os.walk("./data/andy_prosody") for dir in dirs
] if "ASD-Mono/" in path or "NT-Mono/" in path]


feature_extractor = None # to free up the GPU
gc.collect
torch.cuda.empty_cache()
del feature_extractor

# classifier
classifier = TypicalClassifier(participants, "NT-Mono", "ASD-Mono")
classifier.run()
classifier.write_all_csv_results("results/", "andy_prosody")


# end timer
t1 = time.time()
print(f"time taken: {t1 - t0}")
