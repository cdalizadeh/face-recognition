import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def linear_regression_cost_function(x, theta, y):
    m = x.shape[0]
    h = np.matmul(x, theta)
    cost = 0.5 / m * np.sum(np.square(h - y))
    return cost

def linear_regression_gradient(x, theta, y):
    m = x.shape[0]
    h = np.matmul(x, theta)
    grad = 1.0 / m * np.matmul(x.T, h - y)
    return grad

def gradient_descent(x_train, y_train, x_val, y_val, theta_0, alpha = 0.005, max_iterations = 50000, early_stop = True):
    cost_history = np.zeros((max_iterations, 2))
    theta = theta_0
    for i in range(max_iterations):
        theta = theta - alpha * linear_regression_gradient(x_train, theta, y_train)
        cost_history[i, 0] = linear_regression_cost_function(x_train, theta, y_train)
        cost_history[i, 1] = linear_regression_cost_function(x_val, theta, y_val)
        if early_stop and cost_history[i, 1] > cost_history[i - 1, 1] and i != 0:      #if validation error increases, breaks to avoid overfitting
            cost_history = cost_history[:i, :]
            break
    return theta, cost_history

def plot_weightings(theta):
    theta  = theta.reshape(32, 32)
    imgplot = plt.imshow(theta)
    plt.show()

def get_data_in_dir(dir_name, data):
    files_list = os.listdir(dir_name)
    for i in range(len(files_list)):
        file_name = files_list[i]
        img = mpimg.imread(dir_name + file_name) / 255.0
        data[i] = np.concatenate((img.reshape(1, 1024), np.ones((1, 1))), axis = 1)

def calculate_performance(x, theta, y):
    m = x.shape[0]
    h = np.matmul(x, theta)
    results = h * y
    num_success = (results > 0).astype(int).sum()
    return float(num_success) / m

def make_prediction(x, theta):
    h = np.matmul(x, theta)
    if h > 0:
        return "Alec Baldwin"
    else:
        return "Steve Carell"


def baldwin_carell_classification():
    labels_pos = np.ones((70, 1))
    labels_neg = -1 * np.ones((70, 1))
    train_labels = np.concatenate((labels_pos, labels_neg), axis = 0)
    
    labels_pos = np.ones((10, 1))
    labels_neg = -1 * np.ones((10, 1))
    validation_labels = np.concatenate((labels_pos, labels_neg), axis = 0)
    test_labels = np.concatenate((labels_pos, labels_neg), axis = 0)
    
    train = np.zeros((140, 1025))
    validation = np.zeros((20, 1025))
    test = np.zeros((20, 1025))

    get_data_in_dir("./data/organized/Alec Baldwin/train/", train[:70,:])
    get_data_in_dir("./data/organized/Steve Carell/train/", train[70:,:])
    get_data_in_dir("./data/organized/Alec Baldwin/validation/", validation[:10,:])
    get_data_in_dir("./data/organized/Steve Carell/validation/", validation[10:,:])
    get_data_in_dir("./data/organized/Alec Baldwin/test/", test[:10,:])
    get_data_in_dir("./data/organized/Steve Carell/test/", test[10:,:])
    
    theta = np.zeros((1025, 1))

    theta, cost_history = gradient_descent(train, train_labels, validation, validation_labels, theta)
    plt.plot(cost_history)
    plt.show()

    plot_weightings(theta[:1024])

    print("Final cost of training set: " + str(cost_history[-1, 0]))
    print("Final cost of validation set: " + str(cost_history[-1, 1]))
    print("Performance on training set: " + str(calculate_performance(train, theta, train_labels)))
    print("Performance on validation set: " + str(calculate_performance(validation, theta, validation_labels)))
    print("Performance on test set: " + str(calculate_performance(test, theta, test_labels)))

def gender_classification():
    labels_pos = np.ones((210, 1))
    labels_neg = -1 * np.ones((210, 1))
    train_labels = np.concatenate((labels_pos, labels_neg), axis = 0)
    
    labels_pos = np.ones((30, 1))
    labels_neg = -1 * np.ones((30, 1))
    validation_labels = np.concatenate((labels_pos, labels_neg), axis = 0)
    test_labels = np.concatenate((labels_pos, labels_neg), axis = 0)

    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    alt = ['America Ferrera', 'Fran Drescher', 'Kristin Chenoweth', 'Daniel Radcliffe', 'Michael Vartan', 'Gerard Butler']

    train = np.zeros((420, 1025))
    validation = np.zeros((60, 1025))
    test = np.zeros((60, 1025))

    for i in range(len(act)):
        get_data_in_dir("./data/organized/" + act[i] + "/train/", train[70 * i:70 * (i + 1),:])
        get_data_in_dir("./data/organized/" + act[i] + "/validation/", validation[10 * i:10 * (i + 1),:])
        get_data_in_dir("./data/organized/" + alt[i] + "/test/", test[10 * i:10 * (i + 1),:])

    theta = np.zeros((1025, 1))

    sizes = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420]
    train_performance = []
    validation_performance = []
    test_performance = []
    for i in sizes:
        print i
        np.random.seed(i)
        np.random.shuffle(train)
        np.random.seed(i)
        np.random.shuffle(train_labels)

        theta = np.zeros((1025, 1))
        theta, cost = gradient_descent(train[:i,:], train_labels[:i,:], validation, validation_labels, theta)

        train_performance.append(calculate_performance(train[:i,:], theta, train_labels[:i,:]))
        validation_performance.append(calculate_performance(validation, theta, validation_labels))
        test_performance.append(calculate_performance(test, theta, test_labels))

    plt.plot(sizes, train_performance, label="Train")
    plt.plot(sizes, validation_performance, label="Validation")
    plt.plot(sizes, test_performance, label="New Actors")
    plt.xlabel('Training Set Size')
    plt.ylabel('Classifier Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    


if __name__ == "__main__":
    #baldwin_carell_classification()
    gender_classification()
