import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="GNRI classifier")
    parser.add_argument("-i", "--file_path", help="input file path", required=True)
    parser.add_argument("-o", "--output_file", help="output file path", required=True)
    parser.add_argument("-t", "--threshold", type=float, default=0.01, help="threshold value", required=True)
    parser.add_argument("-id", "--classification_level", choices=['RGNTI1', 'RGNTI2', 'RGNTI3'], default='RGNTI3', help="classification level", required=True)
    parser.add_argument("-n", "--normalization", choices=['not', 'some', 'all'], default='not', help="normalization option", required=False)
    parser.add_argument("-f", "--format", choices=['plain', 'multidoc'], default='multidoc', help="file format option", required=False)
    parser.add_argument("-l", "--language", choices=['en', 'ru'], default='en', help="language option", required=False)

    return parser.parse_args()
