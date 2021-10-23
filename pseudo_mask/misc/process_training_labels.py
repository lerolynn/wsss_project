

def main():
    with open('../data/train_label.txt', 'r') as file:
        rows = file.readlines()
        sorted_rows = sorted(rows, key=lambda x:int(x.split(" ")[0].split(".")[0]))
    
    with open('../data/sorted_train_labels.txt','w') as f:
        for row in sorted_rows:
            f.write(row)



if __name__ == "__main__":
    main()
