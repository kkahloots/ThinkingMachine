import csv
#memory_file = 'learner_tracing.csv'

def insert(filename, learner, score):
    with open(r'memory/learner_tracing.csv', 'a', newline='') as csvfile:
        fieldnames = ['filename', 'learner', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'filename': filename, 'learner': learner, 'score': score})
