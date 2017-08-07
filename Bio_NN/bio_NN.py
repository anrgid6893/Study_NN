import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility

def data_input():
    xy = np.loadtxt('train.csv', delimiter=',', dtype=np.float32)
    x_data = xy[:, 1:]
    y_data = xy[:, [0]]

    xy_test = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
    x_test = xy_test[:, 1:]
    y_test = xy_test[:, [0]]
    return x_data, y_data, x_test, y_test

# input data
x_data, y_data, x_test, y_test = data_input()

# parameters
learning_rate = 0.001

# tensorboard logdir
LOGDIR = './tmp/tb_save'
if (os.path.isdir(LOGDIR) == False):
    os.makedirs(LOGDIR)


X = tf.placeholder(tf.float32, [None, len(x_data[0])])
Y = tf.placeholder(tf.float32, [None, 1])

# dropout (keep_prob) rate 0.5 on training, but test dropout rate 1
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("layer1"):
    W1 = tf.get_variable("w1", shape=[133, 133], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([133]))
    L1 = tf.nn.relu(tf.matmul(X, W1)+b1)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

with tf.name_scope("layer2") as scope:
    W2 = tf.get_variable("w2", shape=[133, 133], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([133]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

with tf.name_scope("layer3") as scope:
    W3 = tf.get_variable("w3", shape=[133, 10], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([10]))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
            
with tf.name_scope("layer4") as scope:
    W4 = tf.get_variable("w4", shape=[10, 1], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([1]))
    logits = tf.sigmoid(tf.matmul(L3, W4) + b4)

with tf.name_scope("cost") as scope:
    # Cross entropy cost/loss function
    cost = -tf.reduce_mean(Y * tf.log(logits) + (1 - Y) * tf.log(1 - logits))
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.cast(logits > 0.5, dtype=tf.float32)

actual = y_data
    
# Count true positives, true negatives, false positives and false negatives.
tp = tf.count_nonzero(prediction * actual)
tn = tf.count_nonzero((prediction - 1) * (actual - 1))
fp = tf.count_nonzero(prediction * (actual - 1))
fn = tf.count_nonzero((prediction - 1) * actual)
    

# Calculate accuracy, precision, recall and F1 score.
accuracy = (tp + tn) / (tp + fp + fn + tn)
# Precision
PPV = tp / (tp + fp)
# recall
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
NPV = tn / (tn + fn)
# fmeasure = (2 * PPV * sensitivity) / (PPV + sensitivity)
N = tn + tp + fn + fp
S = (tp + fn) / N
P = (tp + fp) / N

MCC = ((tp / N) - S * P) / tf.sqrt(P * S* (1 - S) * (1 - P))


# Add metrics to TensorBoard.    
tf.summary.scalar('Accuracy', accuracy)
tf.summary.scalar('PPV', PPV)
tf.summary.scalar('sensitivity', sensitivity)
tf.summary.scalar('specificity', specificity)
tf.summary.scalar('NPV', NPV)
tf.summary.scalar('MCC', MCC)

actual_Test = y_test
    
# Count true positives, true negatives, false positives and false negatives.
tp_test = tf.count_nonzero(prediction * actual_Test)
tn_test = tf.count_nonzero((prediction - 1) * (actual_Test - 1))
fp_test = tf.count_nonzero(prediction * (actual_Test - 1))
fn_test = tf.count_nonzero((prediction - 1) * actual_Test)

# Calculate accuracy, precision, recall and F1 score.
accuracy_test = (tp_test + tn_test) / (tp_test + fp_test + fn_test + tn_test)
# Precision
PPV_test = tp_test / (tp_test + fp_test)
# recall
sensitivity_test = tp_test / (tp_test + fn_test)
specificity_test = tn_test / (tn_test + fp_test)
NPV_test = tn_test / (tn_test + fn_test)
# fmeasure = (2 * PPV * sensitivity) / (PPV + sensitivity)
N_test = tn_test + tp_test + fn_test + fp_test
S_test = (tp_test + fn_test) / N_test
P_test = (tp_test + fp_test) / N_test

MCC_test = ((tp_test / N_test) - S_test * P_test) / tf.sqrt(P_test * S_test* (1 - S_test) * (1 - P_test))

with tf.Session() as sess:
    # tensorboard --logdir=./tmp/tb_save/
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)
    
    # Initializer Tensorflow variables
    sess.run(tf.global_variables_initializer())
    
    print('Learning Start')
    for step in range(301):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data, keep_prob: 1})
        summary, loss, acc = sess.run([merged_summary, cost, accuracy], feed_dict={X: x_data, Y: y_data, keep_prob: 1})
        ppv, npv, sn, sp, mcc = sess.run([PPV, NPV, sensitivity, specificity, MCC], feed_dict={X: x_data, Y: y_data, keep_prob: 1})
        writer.add_summary(summary, global_step=step)
        
        if step % 50 is 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}\tPPV:{:.2%}\t NPV:{:.2%}\tSN:{:.2%}\tSP:{:.2%}\tMCC:{:.3f}".format(step, loss, acc, ppv, npv, sn, sp, mcc))
    print('Learning Finish')  
    #Pred = sess.run([prediction], feed_dict={X:x_test, Y:y_test, keep_prob:1})   
    #print(Pred)
    print("test conclusion:")
    L, ac, SN, SP, pp, np, mc = sess.run([cost, accuracy_test, sensitivity_test, specificity_test, PPV_test, NPV_test, MCC_test], feed_dict={X:x_test, Y:y_test, keep_prob:1}) 
    print("Loss: {:.3f}\tAcc: {:.2%}\tPPV:{:.2%}\t NPV:{:.2%}\tSN:{:.2%}\tSP:{:.2%}\tMCC:{:.3f}".format( L, ac, pp, np, SN, SP, mc))



