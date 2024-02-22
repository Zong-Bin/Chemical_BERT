import os

def get_chem(path):
    ts = []
    for file in os.listdir(path):
        if file.endswith('.conll'):
            with open(os.path.join(path, file)) as f:
                lines = f.readlines()
                for line in lines:

                    if line != '\n':
                        line = line.strip('\n').split('\t')
                        label = line[0]
                        word = line[-1]
                        label = label.replace('STARTING_MATERIAL', 'MAT')
                        label = label.replace('REACTION_PRODUCT',  'MAT')
                        label = label.replace('SOLVENT', 'MAT')
                        label = label.replace('OTHER_COMPOUND', 'MAT')
                        label = label.replace('REAGENT_CATALYST', 'MAT')
                        if 'MAT' not in label:
                            label = 'O'
                        ts.append(word+' '+label+'\n')
                    else:
                        ts.append(line)
    with open('../chem/train.txt', 'w') as w:
        for line in ts:
            w.write(line)

if __name__ == '__main__':
    path = '/mnt/sda/bert-ner-给小时/dataset/task1a-ner-train-dev/train'
    get_chem(path)
