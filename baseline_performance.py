import os
from multiprocessing import Pool

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
        feature_avgs_all_layers = feature_extractor.get_features_averages_from_fp(clip_path)
        new_clip = Clip(feature_avgs_all_layers, clip_path)
        clips_list.append(new_clip)
    print(f"Feature extraction for participant {participant_id} done.")
    return Participant(participant_id=participant_id, group_label=group_label, clips=clips_list)


with Pool(os.cpu_count()) as pool:
    participants = pool.map(
        participant_path_to_participant,
        [
            path for path in [
                os.path.join(root, dir) for root, dirs, files in os.walk("./data") for dir in dirs
            ] if "ASD-Mono" in path or "NT-Mono" in path
        ]
    )


# classifier
classifier = TypicalClassifier(participants, "NT-Mono", "ASD-Mono")
classifier.run()
classifier.write_all_csv_results("results/", "baseline")

