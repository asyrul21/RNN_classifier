import csv


class TextToCsv:
    # constructor
    def __init__(self):
        pass

    def convert(self, file, output):

        with open(file, 'r', encoding="ISO-8859-1") as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(",") for line in stripped if line)
            with open(output, 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerow(('label', 'question'))
                writer.writerows(lines)


converter = TextToCsv()
converter.convert('../data/train_5500.txt', 'train_5500.csv')
