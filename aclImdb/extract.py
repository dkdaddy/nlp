f = open("train_neg_files.txt")
f0 = open("train_neg_reviews.txt", mode="w")
for name in f.readlines():
    print(name)
    f1 = open("train/neg/"+name.split()[0])
    review = f1.read()
    f0.write(review + "\n")
    f1.close()
    # print(review)
f0.close()