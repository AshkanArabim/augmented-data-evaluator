import os, gc
import torch

from sourcefiles.feature_extractor import FeatureExtractor
from sourcefiles.typical_atypical_classifier import Clip
from sourcefiles.typical_atypical_classifier import Participant
from sourcefiles.typical_atypical_classifier import TypicalClassifier


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

def get_participants_from_ds_path(path):
    return [participant_path_to_participant(path) for path in [
        os.path.join(root, dir) for root, dirs, files in os.walk(path) for dir in dirs
    ] if "ASD-Mono/" in path or "NT-Mono/" in path]

def speakers_and_orders_match(p1_list: list[Participant], p2_list: list[Participant]) -> bool:
    print("Checking if datasets have the same speakers in the same order...")
    if len(p1_list) != len(p2_list):
        return False
    
    p1_ids = [p1.participant_id for p1 in p1_list]
    p2_ids = [p2.participant_id for p2 in p2_list]
    
    if sorted(p1_ids) != sorted(p2_ids):
        return False
    
    return True
    

# make two lists for augmented and normal participants
non_aug_participants = get_participants_from_ds_path("./data/andy_prosody_minimal")
aug_participants = get_participants_from_ds_path("./data/andy_prosody_minimal_augmented")


feature_extractor = None # to free up the GPU
gc.collect
torch.cuda.empty_cache()
del feature_extractor


# # check if the participants order and names match EXACTLY
# if not speakers_and_orders_match(non_aug_participants, aug_participants):
#     raise ValueError("Participants order and names do not match.")

# classifier
classifier = TypicalClassifier(None, "NT-Mono", "ASD-Mono", 
                               non_aug_participants=non_aug_participants, 
                               aug_participants=aug_participants)

classifier.run()
classifier.write_all_csv_results("results/", "andy_prosody")
