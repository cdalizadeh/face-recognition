import numpy as np
import matplotlib.pyplot as plt
import linear_regression as lr      #import previous file to use existing functions

def one_hot_cost_function(x, y, theta):
    m = x.shape[0]
    h = np.matmul(x, theta)
    cost = 0.5 / m * np.sum(np.square(h - y))
    return cost

def one_hot_gradient(x, y, theta):
    m = x.shape[0]
    h = np.matmul(x, theta)
    grad = 1.0 / m * np.matmul(x.T, h - y)
    return grad

def gradient_check(x, y, theta):
    h_values = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001]
    grad_history = np.zeros((len(h_values), 1))
    grad = one_hot_gradient(x, y, theta)

    for i in range(len(h_values)):
        h = h_values[i] * np.ones((theta.shape[0], theta.shape[1]))
        f_prime = float(one_hot_cost_function(x, y, theta + h) - one_hot_cost_function(x, y, theta)) / h

        grad_history[i, 0] = np.sum(np.square(f_prime - grad))
    
    print grad[0,0]
    print f_prime[0, 0]
    plt.plot(grad_history)
    plt.show()

def gradient_descent(x_train, y_train, x_val, y_val, theta_0, alpha = 0.005, max_iterations = 50000, early_stop = True):
    cost_history = np.zeros((max_iterations, 2))
    theta = theta_0
    for i in range(max_iterations):
        theta = theta - alpha * one_hot_gradient(x_train, y_train, theta)
        cost_history[i, 0] = one_hot_cost_function(x_train, y_train, theta)
        cost_history[i, 1] = one_hot_cost_function(x_val, y_val, theta)
        if early_stop and cost_history[i, 1] > cost_history[i - 1, 1] and i != 0:      #if validation error increases, breaks to avoid overfitting
            cost_history = cost_history[:i, :]
            break
    return theta, cost_history

def calculate_performance(x, theta, y):
    m = x.shape[0]
    h = np.matmul(x, theta)

    num_success = 0
    for i in range(m):
        index = np.argmax(h[i,:])
        if y[i, index] == 1:
            num_success += 1
    
    return float(num_success) / m

def plot_weightings(theta):
    theta = theta.reshape(32, 32)
    imgplot = plt.imshow(theta)
    plt.show()

def actor_classification():
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

    train = np.zeros((420, 1025))
    validation = np.zeros((60, 1025))
    test = np.zeros((60, 1025))

    train_labels = np.zeros((420, 6))
    validation_labels = np.zeros((60, 6))
    test_labels = np.zeros((60, 6))

    for i in range(len(act)):
        train_labels[70 * i:70 * (i + 1),i] = np.ones((70))
        validation_labels[10 * i:10 * (i + 1),i] = np.ones((10))
        test_labels[10 * i:10 * (i + 1),i] = np.ones((10))
        lr.get_data_in_dir("./data/organized/" + act[i] + "/train/", train[70 * i:70 * (i + 1),:])
        lr.get_data_in_dir("./data/organized/" + act[i] + "/validation/", validation[10 * i:10 * (i + 1),:])
        lr.get_data_in_dir("./data/organized/" + act[i] + "/test/", test[10 * i:10 * (i + 1),:])

    theta = np.ones((1025, 6))

    gradient_check(train, train_labels, theta)
    theta, cost_history = gradient_descent(train, train_labels, validation, validation_labels, theta)

    for i in range(theta.shape[1]):
        plot_weightings(theta[:-1,i])

    plt.plot(cost_history[:,0], label="Train")
    plt.plot(cost_history[:,1], label="Validation")
    plt.xlabel('Iterations')
    plt.ylabel('Classifier Error')
    plt.legend(loc='upper right')
    axes = plt.gca()
    axes.set_ylim([0,25])
    plt.show()

    print("Performance on training set: " + str(calculate_performance(train, theta, train_labels)))
    print("Performance on validation set: " + str(calculate_performance(validation, theta, validation_labels)))
    print("Performance on test set: " + str(calculate_performance(test, theta, test_labels)))

if __name__ == "__main__":
    actor_classification()
