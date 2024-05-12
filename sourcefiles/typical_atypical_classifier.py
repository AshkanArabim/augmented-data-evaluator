import sys
import csv
import os
from sourcefiles.cosine_similarity import get_cosine_similarity
import torch
import numpy as np


# uncomment the top one if your GPU has >12GB vram
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"


# ques: WHY do we need cosine similarity for knn? this makes no damn sense
def cosine_similarity_knn(list_1, list_2):
    """
    Gets the cosine_similarity of two lists and adjusts it to be used in the
    KNN algorithm
    """
    cs = get_cosine_similarity(list_1, list_2)
    return 1 - cs


# cmt: commenting cuz I don't use it anyore
# def get_knn_predictions(test_list, train_list, labels, k):
#     """
#     Trains the KNN algorithm on the train_list and labels then passes
#     the test_list to get a prediction for each element in the test_list.
#     :param train_list: A list full of features averages to train the KNN algorithm.
#     :param test_list: A list full of features averages to test the KNN algorithm.
#     :param labels: A list of labels for each list of features averages. Label
#     values are taken from participant.group_label.
#     :param k: The value of k for the kNN algorithm.
#     :return: a list of strings signifying if each features averages in the test
#     list was given a prediction of SLI or TD.
#     """
    
#     print("--- start of knn ---") # debug
    
#     # cmt: dim of train_list & test_list = [<num_clips>, 1024]
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     train_list = torch.tensor(train_list).to(device)
#     test_list = torch.tensor(test_list).to(device)
    
#     # get the indices of K nearest neighbors to each of our test clips
#     # distances dim: [test_list, train_list]
#     distances = torch.cosine_similarity(train_list[None, :, :], test_list[:, None, :], dim=-1)
    
#     # get neighbors with highest cosine similarity
#     # dim: [test_list, k]
#     _, nearest_indices = torch.topk(distances, k=k, dim=-1)
    
#     # map all indices to the corresponding class
#     # labels are received as strings, so we first convert them to integers
#     labels_int_to_string = tuple(set(labels))
#     labels_string_to_int = {s: i for i, s in enumerate(labels_int_to_string)}
#     int_labels = torch.tensor([labels_string_to_int[s] for s in labels]).to(device)
#     nearest_int_labels = int_labels[nearest_indices]
    
#     # get the most common label for each inference input
#     knn_results, _ = torch.mode(nearest_int_labels)
    
#     # changes int labels back to string
#     knn_results = [labels_int_to_string[i] for i in knn_results]
    
#     print("--- end of knn ---") # debug
    
#     return knn_results
#     # note: (done) ported till here ^^, vv in progress


class Participant:
    def __init__(self, participant_id, group_label, clips, age=None, gender=None):
        self.participant_id = participant_id
        self.group_label = group_label
        self.clips: list[Clip] = clips  # list of Clip objects
        self.age = age
        self.gender = gender
        self.results = None

    def add_results(self, group_label, predictions, test_clips_names):
        self.results = ParticipantResults(self.participant_id, group_label, predictions, test_clips_names)

    def get_num_clips(self):
        return len(self.clips)

    def to_dict(self):
        return {
            'participant_id': self.participant_id,
            'group_label': self.group_label,
            'clips': [c.to_dict() for c in self.clips],
            'age': self.age,
            'gender': self.gender
        }

    @staticmethod
    def from_dict(data):
        clips = [Clip.from_dict(d['layers'], d['clip_name']) for d in data['clips']]
        return Participant(data['participant_id'], data['group_label'], clips, age=data['age'], gender=data['gender'])


class Clip:
    """
    Holds the feature averages for each transformation layer for a single clip.
    features_averages can be taken directly from the get_features_averages_from_fp
    method in the FeatureExtractor class.
    """
    def __init__(self, features_averages: torch.tensor, clip_name):
        # self.layers = {i: list(features) for i, features in enumerate(features_averages)}
        self.layers : torch.tensor = features_averages # dim: [24, 1024]
        self.clip_name = clip_name

    # cmt: note that now the indices start from 0!
    def get_transformation_layer(self, layer_idx):
        return self.layers[layer_idx]

    def to_dict(self):
        return {'layers': self.layers, 'clip_name': self.clip_name}

    @staticmethod
    def from_dict(layers_data, clip_name):
        clip = Clip([], clip_name)
        clip.layers = layers_data
        return clip


# ques: I don't fully get what this is supposed to do...
class ParticipantResults:
    # ques: what are the reuslts here? prediction of whether they're typical or atypical?
    """
    Gets the predictions for a participant and calculates the results
    Each layer of hubert has a different set of labels for the participant's clips
    self.layers_predictions holds a list of labels signifying the label that layer gave to each clip
    self.clip_results will hold a list of boolean values l. l[3] == True means layer 4 correctly predicted this clip
    self.layer_results holds a boolean value for each layer. True means number of correct clip labels > wrong labels
    """
    def __init__(self, participant_id, group_label, layers_predictions, clip_names, num_layers=24):
        self.participant_id = participant_id
        self.group_label = group_label
        # cmt: for each layer, stores a list of the label that layer gave to each track
        self.layers_predictions = {i: predictions for i, predictions in enumerate(layers_predictions)}
        self.clip_names = clip_names
        # cmt: shows that the system correctly predicted a certain track
        self.clip_results = {name: [] for name in self.clip_names}
        # cmt: shows what layers predicted correctly more than incorrectly on average
        self.layers_results = {i: None for i in range(num_layers)}
        self.winner = False
        self.calculate_results()

    # cmt: literally just populates the data structures above
    def calculate_results(self):
        """
        Iterate through the predictions of each layer.
        :return:
        """
        for layer_idx in self.layers_predictions:
            single_layer_predictions = self.layers_predictions[layer_idx]
            correct_count = list(single_layer_predictions).count(self.group_label)
            wrong_count = len(single_layer_predictions) - correct_count
            self.layers_results[layer_idx] = (correct_count > wrong_count,
                                              correct_count / (correct_count + wrong_count))
            self.calculate_clip_results(layer_idx)

        all_layers_winner_count = sum(1 for is_winner, _ in list(self.layers_results.values()) if is_winner)
        all_layers_loser_count = len(self.layers_results) - all_layers_winner_count
        self.winner = all_layers_winner_count > all_layers_loser_count
        print(f"{self.participant_id} winner = {self.winner}")

    def calculate_clip_results(self, layer_idx):
        """
        Given a layer_idx, this method will update self.clip_results so that
        every clip will get a boolean value added to their list representing
        if that clip was correctly classified in that layer.
        :param layer_idx:
        :return:
        """
        for idx, prediction in enumerate(self.layers_predictions[layer_idx]):
            self.clip_results[self.clip_names[idx]].append(prediction == self.group_label)

    # cmt: read till here ^^
    # ans: what is self.clip_results?? --> holds which layers classified which audio tracks correctly
    def get_clip_results(self):
        """
        This method goes through the self.clip_results and for each clip,
        it gets the ratio of correct/total predictions.
        :return: A list of tuples t where t[0] is the clip_name and t[1]
        is the ratio of correct/total predictions
        """
        clip_results = []
        for clip_name in self.clip_results:
            single_clip_results = self.clip_results[clip_name]
            num_true = single_clip_results.count(True)
            num_false = single_clip_results.count(False)
            clip_results.append((clip_name, num_true / (num_true + num_false)))
        return clip_results

    def get_layer_results(self):
        """
        Gets the performance of each layer
        :return: A list of tuples: t where t[0] is a boolean indicating if the layer won
        and t[1] is the ratio of correct/total predictions
        """
        return [self.layers_results[layer_idx] for layer_idx in self.layers_results]


class TypicalClassifier:
    def __init__(self, participants_list, typical_label, atypical_label, non_aug_participants: list = None, 
                aug_participants: list = None, num_layers=24, k=7):
        
        if participants_list == None: # when using augmented for train, and original for test
            if None in [non_aug_participants, aug_participants]:
                raise ValueError("must pass both non_aug_participants and aug_participants if participants_list is None")
            
            self.non_aug_participants = non_aug_participants
            self.aug_participants = aug_participants
        else:
            self.participants_list = participants_list  # list of Participant objects
        
        self.typical_label = typical_label
        self.atypical_label = atypical_label
        self.num_layers = num_layers
        self.k = k
        self.seen_participant_ids = set()
        self.counters = {
            'winners': {'total': 0, 'typical': 0, 'atypical': 0},
            'losers': {'total': 0, 'typical': 0, 'atypical': 0}
        }

    def run(self):
        if hasattr(self, 'non_aug_participants'):
            print("Augmented and non-augmented datasets detected. Using hybrid evaluation.")
            # FIXME: revert
            # self.process_all_participants_hybrid()
        else:
            self.process_all_participants()
        self.print_results()
    
    # FIXME: revert
    # def process_all_participants_hybrid(self):
    #     """
    #     Very similar to `process_all_participants`, but instead iterates over
    #     non_aug_participants to test them on knn trained on aug_participants.
    #     :return:
    #     """
    #     # select one of the non-augmented participants for testing
    #     for participant in self.non_aug_participants:
    #         if participant.participant_id in self.seen_participant_ids:
    #             print(f'ERROR participant {participant.participant_id} already processed. Duplicate IDs')
    #             sys.exit()
    #         self.seen_participant_ids.add(participant.participant_id)
    #         self.process_single_participant_hybrid(participant)

    def process_all_participants(self):
        for participant in self.participants_list:
            if participant.participant_id in self.seen_participant_ids:
                print(f'ERROR participant {participant.participant_id} already processed. Duplicate IDs')
                sys.exit()
            self.seen_participant_ids.add(participant.participant_id)
            self.process_single_participant(participant)
    
    # FIXME: revert
    # def process_single_participant_hybrid_new(self, non_aug_participant):
    #     # FIXME:
    #     pass
    
    # def process_single_participant_new(self, participant):
    #     # FIXME: (done)
        
    #     """
    #     makes a single participant be the test data while the rest of the participants
    #     are used to train the knn algorithm.
    #     :param participant: Participant object representing participant to be tested
    #     :param group_label:
    #     :return:
    #     """
    #     participant_id = participant.participant_id
        
    #     if participant_id in self.seen_participant_ids:
    #         print(f'ERROR participant {participant_id} already processed. Duplicate IDs')
    #         sys.exit()
    #     self.seen_participant_ids.add(participant_id)
        
    #     group_label = participant.group_label
    #     print(f"processing participant {participant_id}")
        
    #     # this section ported from get_train_test()
        
    #     # test_clips dim: [24, <num_clips_test>, 1024] (each person can have multiple)
    #     # test_clips_names dim: [<num_clips_test>] (stores the literal wav paths)
    #     # train_clips dim: [24, <num_clips_train>, 1024] (each person can have multiple)
    #     # train_labels dim: [<num_clips_train>]
        
    #     test_clips, test_clips_names, train_clips, train_labels = self.get_train_test(participant_id)
        
    # FIXME: revert
    # def process_single_participant_hybrid(self, non_aug_participant):
    #     # note: ported till here ^^
    #     """
    #     Same as `process_single_participang`, but uses augmented participants
    #     for training and normal participants for testing.
    #     """
    #     participant_id = non_aug_participant.participant_id
        
    #     # if participant_id in self.seen_participant_ids:
    #     #     print(f'ERROR participant {participant_id} already processed. Duplicate IDs')
    #     #     sys.exit()
    #     # self.seen_participant_ids.add(participant_id)
        
    #     group_label = non_aug_participant.group_label
    #     print(f"processing participant {participant_id} (hybrid method)")

    #     test_clips, test_clips_names, train_clips, train_labels = self.get_train_test_hybrid(participant_id)
    #     predictions = self.get_predictions(test_clips, train_clips, train_labels)

    #     non_aug_participant.add_results(group_label, predictions, test_clips_names)

    #     outcome = 'winners' if non_aug_participant.results.winner else 'losers'
    #     label_type = 'typical' if non_aug_participant.group_label == self.typical_label else 'atypical'

    #     # Increment the total outcome counter
    #     self.counters[outcome]['total'] += 1

    #     # Increment the specific label type outcome counter
    #     self.counters[outcome][label_type] += 1
    
    # cmt: okay it doesn't have a global KNN, so why this deadlocks is beyond me
    def process_single_participant(self, participant):
        """
        makes a single participant be the test data while the rest of the participants
        are used to train the knn algorithm.
        :param participant: Participant object representing participant to be tested
        :param group_label:
        :return:
        """
        participant_id = participant.participant_id
        
        # if participant_id in self.seen_participant_ids:
        #     print(f'ERROR participant {participant_id} already processed. Duplicate IDs')
        #     sys.exit()
        # self.seen_participant_ids.add(participant_id)
        
        group_label = participant.group_label
        print(f"processing participant {participant_id}")

        test_clips, test_clips_names, train_clips, train_labels = self.get_train_test(participant_id)
        predictions = self.get_predictions(test_clips, train_clips, train_labels)

        participant.add_results(group_label, predictions, test_clips_names)

        outcome = 'winners' if participant.results.winner else 'losers'
        label_type = 'typical' if participant.group_label == self.typical_label else 'atypical'

        # Increment the total outcome counter
        self.counters[outcome]['total'] += 1
        # note: ported till here ^^, vv is in progress

        # Increment the specific label type outcome counter
        self.counters[outcome][label_type] += 1

    # TODO: this would benefit from a dictionary data structure instead of lists
    # FIXME: revert
    # def get_train_test_hybrid(self, participant_id):
    #     """
    #     Same as `get_train_test`, but sources the train data from augmented
    #     participants, and the test data from non-augmented participants.
    #     :param participant_id: participant_id of participant to be tested
    #     :return: test_clips, test_clip_names, train_clips, train_labels
    #     """
    #     test_clips = []
    #     test_clips_names = []
    #     train_clips = []
    #     train_labels = []
    #     # assuming exact parity between 
    #     for (aug_participant, non_aug_participant) in zip(self.aug_participants, self.non_aug_participants):
    #         if non_aug_participant.participant_id == participant_id:
    #             test_clips = non_aug_participant.clips
    #             test_clips_names = [clip.clip_name for clip in test_clips]
    #         else:
    #             train_clips += aug_participant.clips
    #             train_labels += [aug_participant.group_label for _ in range(aug_participant.get_num_clips())]
    #     return test_clips, test_clips_names, train_clips, train_labels
    
    def get_train_test(self, participant_id):
        """
        given a participant id this method will separate the test and train clips
        and metadata for the kNN algorithm.
        Note that this function will also move all the resulting tensors to GPU
        in preparation for the torch KNN function.
        :param participant_id: participant_id of participant to be tested
        :return: test_clips, test_clip_names, train_clips, train_labels
        """
        # test_clips = []
        # test_clips_names = []
        # train_clips = []
        # train_labels = []
        # for participant in self.participants_list:
        #     if participant.participant_id == participant_id:
        #         test_clips = participant.clips
        #         test_clips_names = [clip.clip_name for clip in test_clips]
        #     else:
        #         train_clips += participant.clips
        #         train_labels += [participant.group_label for _ in range(participant.get_num_clips())]
        # return test_clips, test_clips_names, train_clips, train_labels
        # note: (done) ported till here ^^, vv is in progress
        
        # refactored -------------------------
        
        # test_clips dim: [24, <num_clips_test>, 1024] (each person can have multiple)
        # test_clips_names dim: [<num_clips_test>] (stores the literal wav paths)
        # train_clips dim: [24, <num_clips_train>, 1024] (each person can have multiple)
        # train_labels dim: [<num_clips_train>]
        
        test_clips = torch.zeros([24, 0, 1024]).to(DEVICE)
        test_clips_names = []
        train_clips = torch.zeros([24, 0, 1024]).to(DEVICE)
        train_labels = []
        
        for participant in self.participants_list:
            if participant.participant_id == participant_id:
                test_clips = torch.cat([clip.layers.to(DEVICE)[:, None, :] for clip in participant.clips], dim=1)
                test_clips_names = [clip.clip_name for clip in participant.clips]
            else:
                # adding new clips to the end of what we already have, just like append
                train_clips = torch.cat([train_clips] + [clip.layers.to(DEVICE)[:, None, :] for clip in participant.clips], dim=1)
                train_labels += [participant.group_label for _ in range(participant.get_num_clips())]
        
        return test_clips, test_clips_names, train_clips, train_labels

    def get_predictions(self, test_clips: torch.tensor, train_clips: torch.tensor, train_labels, k=7) -> list[str]:
        """
        Gets the knn predictions for all n layers.
        :param test_clips:
        :param train_clips:
        :param train_labels:
        :return:
        """
        # all_layers_predictions = []
        # for layer_idx in range(1, self.num_layers + 1):
        #     test_clips_features = self.get_layer_features(layer_idx, test_clips)
        #     train_clips_features = self.get_layer_features(layer_idx, train_clips)
        #     predictions = get_knn_predictions(test_clips_features, train_clips_features, train_labels, self.k)
        #     all_layers_predictions.append(predictions)
        # return all_layers_predictions
            # # note: (done) commented unused funcs till here ^^, vv in progress
            # # note: (done) ported till here ^^, vv in progress
    
        # refactored ---------------------
        # FIXME: (done)
        print("--- start of knn for all 24 ---")
        
        # test_clips dim: [24, <num_clips_test>, 1024] (each person can have multiple)
        # test_clips_names dim: [<num_clips_test>] (stores the literal wav paths)
        # train_clips dim: [24, <num_clips_train>, 1024] (each person can have multiple)
        # train_labels dim: [<num_clips_train>]
        
        # get the indices of K nearest neighbors to each of our test clips
        # distances dim: [24, test_list, train_list]
        distances = torch.cosine_similarity(train_clips[:, None, :, :], test_clips[:, :, None, :], dim=-1)
        
        # get neighbors with highest cosine similarity
        # dim: [24, test_list, k]
        _, nearest_indices = torch.topk(distances, k=k, dim=-1)
        
        # map all indices to the corresponding class
        # labels are received as strings, so we first convert them to integers
        train_labels = np.asarray(train_labels) # dim: [24, <num_clips_train>] cmt: check this
        labels_int_to_string = tuple(set(train_labels.flatten()))
        labels_string_to_int = {s: i for i, s in enumerate(labels_int_to_string)}
        int_train_labels = torch.tensor(np.vectorize(lambda x: labels_string_to_int[x])(train_labels)).to(DEVICE)
        
        # pytorch supports nd tensor indexing so this should work without modding
        nearest_int_labels = int_train_labels[nearest_indices]
        
        # get the most common label for each inference input
        # dim: [<num_clips_test>]
        # cmt: this was called knn_results in get_knn_predictions()
        predictions, _ = torch.mode(nearest_int_labels, dim=-1)
        
        # changes int labels back to string
        predictions = np.vectorize(lambda x: labels_int_to_string[x])(np.asarray(predictions.to("cpu")))
        
        print("--- end of knn for all 24 ---")
        
        return list(predictions)

    # cmt: commenting cuz I don't use it anymore
    # def get_layer_features(self, layer_idx, clips_list):
    #     """
    #     gets the features for each clip processed by a single layer
    #     :param layer_idx: Layer to be processed
    #     :param clips_list: List of Clip objects.
    #     :return:
    #     """
    #     return [clip.get_transformation_layer(layer_idx) for clip in clips_list]

    def print_results(self):
        print(f"total num winners = {self.counters['winners']['total']}")
        print(f"total num losers = {self.counters['losers']['total']}")
        print('-----------------------')

        print(f'num {self.typical_label} winners = {self.counters["winners"]["typical"]}')
        print(f'num {self.typical_label} losers = {self.counters["losers"]["typical"]}')
        print('-----------------------')

        print(f'num {self.atypical_label} winners = {self.counters["winners"]["atypical"]}')
        print(f'num {self.atypical_label} losers = {self.counters["losers"]["atypical"]}')

    def write_layer_results_csv(self, csv_path):
        header = ['participant_id', 'participant_group', 'participant_age', 'participant_gender', 'winner']
        header += ['layer' + str(i) for i in range(1, self.num_layers + 1)]
        rows_data = []
        for participant in self.participants_list:
            row_data = [participant.participant_id, participant.group_label,
                            participant.age, participant.gender, participant.results.winner]
            row_data += [result[1] for result in participant.results.get_layer_results()]
            rows_data.append(row_data)
        write_csv(csv_path, header, rows_data)

    def write_participant_clip_results_csv(self, csv_path, participant_id):
        participant = next((p for p in self.participants_list if p.participant_id == participant_id), None)
        if participant is None:
            print(f'{participant_id} is an invalid id, cant write csv')
            return
        header = ['clip_name', 'clip_performance']
        rows = []
        for clip_name, clip_result in participant.results.get_clip_results():
            rows.append([clip_name, clip_result])
        write_csv(csv_path, header, rows)

    def write_age_results_csv(self, csv_path):
        participant_ages = [p.age for p in self.participants_list]
        min_age = min(participant_ages)
        max_age = max(participant_ages)
        counters_by_age = {key: 0 for age in range(min_age, max_age + 1)
                           for key in (str(age) + '_winners', str(age) + '_losers')}
        counters_by_age_and_type = {
            f"{age}_{label}_{status}": 0
            for age in range(min_age, max_age + 1)
            for label in (self.typical_label, self.atypical_label)
            for status in ("winners", "losers")
        }
        all_counters = counters_by_age | counters_by_age_and_type
        for participant in self.participants_list:
            age = str(participant.age)
            winning_status = '_winners' if participant.results.winner else '_losers'
            all_counters[age + winning_status] += 1
            all_counters[age + '_' + participant.group_label + winning_status] += 1
        header = ['counter_id', 'count']
        rows = [[counter_id, all_counters[counter_id]] for counter_id in all_counters]
        write_csv(csv_path, header, rows)

    def write_gender_results_csv(self, csv_path):
        gender_counter = {'male_winner': 0, 'male_loser': 0,
                          'female_winner': 0, 'female_loser': 0,
                          f'male_{self.typical_label}_winner': 0, f'male_{self.typical_label}_loser': 0,
                          f'male_{self.atypical_label}_winner': 0, f'male_{self.atypical_label}_loser': 0,
                          f'female_{self.typical_label}_winner': 0, f'female_{self.typical_label}_loser': 0,
                          f'female_{self.atypical_label}_winner': 0, f'female_{self.atypical_label}_loser': 0}
        for participant in self.participants_list:
            winning_status = '_winner' if participant.results.winner else '_loser'
            gender_counter[participant.gender + winning_status] += 1
            gender_counter[participant.gender + '_' + participant.group_label + winning_status] += 1
        header = ['gender_group', 'count']
        rows = [[group_label, gender_counter[group_label]] for group_label in gender_counter]
        write_csv(csv_path, header, rows)

    def write_all_csv_results(self, directory, project_name):
        # TODO: is solution can have unwanted consequences
        if hasattr(self, "non_aug_participants"):
            self.participants_list = self.non_aug_participants
        
        self.write_layer_results_csv(directory + project_name + '_layers.csv')
        self.write_age_results_csv(directory + project_name + '_age.csv')
        self.write_gender_results_csv(directory + project_name + '_gender.csv')

    def get_all_participants(self):
        return self.participants_list


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        csvwriter.writerows(rows)
