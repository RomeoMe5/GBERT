import argparse
import logging
from transformers import logging as transformers_logging

from core.config import *
from core.utils import *
from core.cli import *

transformers_logging.set_verbosity_error()
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def process_text(model_predictors, custom_dataset, grnti_dict_lvl_1, grnti_dict_lvl_2, output_file, top_n=3, threshold=0.001, classification_level='RGNTI3', normalization='not'):
    with open(output_file, 'w') as f:
        f.write('id\tresult\trubricator\tlanguage\tthreshold\tversion\tnormalize\tcorrect\r\n')
        for idx in range(len(custom_dataset)):
            result_raw = []

            data = custom_dataset[idx]
            text_id = data['text_id']
            if data['title'][0] not in ['.', '!', '?']:
                title = data['title'] + '.' 
            print(title)

            text = title + data['text'] + data['keywords']

            try:
                predicted_labels_1, predicted_probabilities_1 = model_predictors["SRTSI_1"].make_prediction(text, top_n=top_n, type='sigmoid')

                for predicted_label_1, predicted_probability_1 in zip(predicted_labels_1, predicted_probabilities_1):
                    feature1 = grnti_dict_lvl_1[str(predicted_label_1)]
                    concatenated_text_1 = f"{feature1}." + text 
                    if classification_level == 'RGNTI1':
                        result_raw.append([predicted_label_1, predicted_probability_1])

                    if classification_level in ['RGNTI2', 'RGNTI3']:
                        predicted_labels_2, predicted_probabilities_2 = model_predictors["SRTSI_2"].make_prediction(concatenated_text_1, top_n=2)
                        for predicted_label_2, predicted_probability_2 in zip(predicted_labels_2, predicted_probabilities_2):
                            feature2 = grnti_dict_lvl_2[f"{predicted_label_2[0:2]}.{predicted_label_2[2:4]}"]
                            concatenated_text_2 = f"{feature1}.{feature2}." + text 
                            if classification_level == 'RGNTI2':
                                result_raw.append([f"{predicted_label_2[0:2]}.{predicted_label_2[2:4]}", predicted_probability_1 * predicted_probability_2])
                            
                            if classification_level == 'RGNTI3':
                                predicted_labels_3, predicted_probabilities_3 = model_predictors["SRTSI_3"].make_prediction(concatenated_text_2, top_n=2)
                                for predicted_label_3, predicted_probability_3 in zip(predicted_labels_3, predicted_probabilities_3):
                                    result_raw.append([f"{predicted_label_3[0:2]}.{predicted_label_3[2:4]}.{predicted_label_3[4:6]}", predicted_probability_1 * predicted_probability_2 * predicted_probability_3])
                results = format_data(result_raw, threshold, normalization)
            except Exception as e:
                print("Error occurred:", e)
                results = "REJECT"
            print(text_id, results)
            f.write(f"{text_id}\t{results}\t{classification_level}\t{LANGUAGE}\t{threshold}\t{VERSION}\t{normalization}\t###\r\n")


def main():
    args = parse_arguments()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    grnti_dict_lvl_1 = load_dictionary(DICT_FILE_PATHS["lvl_1"])
    grnti_dict_lvl_2 = load_dictionary(DICT_FILE_PATHS["lvl_2"])

    model_predictors = {}
    for model_name, paths in MODEL_LABEL_PATHS.items():
        model_predictors[model_name] = ModelPredictor(paths['model'], paths['labels'], tokenizer)

    custom_dataset = CustomDataset(args.file_path, tokenizer)

    process_text(model_predictors, custom_dataset, grnti_dict_lvl_1, grnti_dict_lvl_2, args.output_file, threshold=args.threshold, classification_level=args.classification_level, normalization=args.normalization)

if __name__ == "__main__":
    main()
