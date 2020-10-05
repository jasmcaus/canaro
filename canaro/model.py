# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Surpressing Tensorflow Warnings
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# Importing the necessary packages
def saveModel(model, base_name, learn_rate ,attempt):
    model.save_weights(f'{base_name}-{learn_rate}-{attempt}.h5')
    model.save(f'{base_name}_{attempt}.h5')


def testModel(model,classes):
    pass
    # X_test, y_test = preprocess(array) # y_test will be empty
    # x = np.array(X_test)
    # x = normalize(x)
    # test_datagen = imageDataGenerator()

    # # Plotting
    # columns = 5
    # i=0
    # test_labels = []
    # plt.figure(figsize=(30,30))
    # for batch in test_datagen.flow(x, batch_size=1):
    #     pred = model.predict(batch)
    #     if pred > 0.5:
    #         test_labels.append(str(categories[1]))
    #     else:
    #         test_labels.append(str(categories[0]))
    #     plt.subplot(5/columns+1, columns, i+1)
    #     plt.title(f'This is a {test_labels[i]}')
    #     i += 1
    #     # Displaying the first 10 images
    #     if i%10:
    #         break

    #     plt.show()