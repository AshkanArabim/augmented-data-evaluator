import os, threading

from sourcefiles.feature_extractor import FeatureExtractor
from sourcefiles.typical_atypical_classifier import Clip
from sourcefiles.typical_atypical_classifier import Participant
from sourcefiles.typical_atypical_classifier import TypicalClassifier


feature_extractor = FeatureExtractor()

NT_dir = os.path.join("data", "andy_prosody", "NT-Mono")
ASD_dir = os.path.join("data", "andy_prosody", "ASD-Mono")
NT_participant_clip_ids = os.listdir(NT_dir)
ASD_participant_clip_ids = os.listdir(ASD_dir)


# load NT participants
def extract_group_features(parent_dir: str, participant_ids: list, group_label: str):
    group_participants = []
    for participant_id in participant_ids:
        print(f"Starting feature extraction for participant {participant_id}...")
        participant_clip_directory = os.path.join(parent_dir, participant_id)
        clips_list = []
        for clip_filename in os.listdir(participant_clip_directory):
            clip_path = os.path.join(participant_clip_directory, clip_filename)
            # debug
            print(f"Extracting features from: {clip_path}")
            feature_avgs_all_layers = feature_extractor.get_features_averages_from_fp(clip_path)
            new_clip = Clip(feature_avgs_all_layers, clip_path)
            clips_list.append(new_clip)
        group_participants.append(Participant(participant_id=participant_id, group_label=group_label, clips=clips_list))
        print(f"Feature extraction for participant {participant_id} done.")
    
    return group_participants


participants = extract_group_features(NT_dir, NT_participant_clip_ids, "NT")
participants += extract_group_features(ASD_dir, ASD_participant_clip_ids, "ASD")


# classifier
classifier = TypicalClassifier(participants, 'NT', 'ASD')
classifier.run()
classifier.write_all_csv_results("results/", "baseline")

