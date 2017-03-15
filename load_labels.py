import csv

def get_sign_titles():
    sign_titles = []
    with open('signnames.csv', 'rt', encoding='utf8') as f:
        sign_names = csv.reader(f, delimiter=',')
        for i, row in enumerate(sign_names):
            if i != 0:
                sign_titles.append(row[1])
    return sign_titles