import numpy as np
import time
import collections
import sklearn
import tensorflow as tf


def predict(sess, model,
            samples, prior, labels
            ):
    loss = 0
    n_samples = samples.shape[0]
    predictions = np.empty(n_samples)

    for begin in range(0, n_samples, model.batch_size):
        end = begin + model.batch_size
        end = min([end, n_samples])

        batch_samples = np.zeros(shape=[model.batch_size] + list(samples.shape[1:]))

        tmp_data_samples = samples[begin:end]
        if type(tmp_data_samples) is not np.ndarray:
            tmp_data_samples = tmp_data_samples.toarray()  # convert sparse matrices

        batch_samples[:end - begin] = tmp_data_samples

        feed_dict = {
            model.ph_sample: batch_samples,
            model.ph_priors: prior,
            model.ph_bn: False
        }

        # Compute loss if labels are given.
        if labels is not None:
            batch_label = np.zeros(model.batch_size)
            batch_label[:end - begin] = labels[begin:end]
            feed_dict[model.ph_label] = batch_label
            batch_prediction, batch_loss = sess.run([model.op_prediction, model.ce_loss], feed_dict)
            loss += batch_loss
        else:
            batch_prediction = sess.run(model.op_prediction, feed_dict)

        predictions[begin:end] = batch_prediction[:end - begin]

    if labels is not None:
        return predictions, loss * model.batch_size / n_samples
    else:
        return predictions


def evaluate(sess, model,
             samples, matrix, labels,
             ):
    t_process, t_wall = time.process_time(), time.time()

    predictions, loss = predict(sess, model, samples, matrix, labels)
    n_corrects = np.sum(predictions == labels)
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    f1 = sklearn.metrics.f1_score(labels, predictions)
    precision = sklearn.metrics.precision_score(labels, predictions)
    recall = sklearn.metrics.recall_score(labels, predictions)
    string = 'accuracy: {:.4f} ({:d} / {:d}), recall : {:.4f}, precision : {:.4f}, f1 : {:.4f}, clf_loss: {:.4e}'. \
        format(accuracy, n_corrects, len(labels), recall, precision, f1, loss)
    if sess is None:
        string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall)
    return string, accuracy, f1, loss


def fit(sess, model,
        n_epochs,
        learning_rate, lr_decay_rate,
        train_data_samples, train_priors, train_labels,
        val_data_samples, val_priors, val_labels,
        logger):
    t_process, t_wall = time.process_time(), time.time()

    n_samples = train_data_samples.shape[0]

    # Training.
    # validation performance
    val_accuracies = []
    val_losses = []

    indices = collections.deque()
    num_steps = int(n_epochs * n_samples / model.batch_size)

    eval_frequency = int(n_samples / model.batch_size)

    for step in range(1, num_steps + 1):
        # Be sure to have used all the samples before using one a second time.
        if len(indices) < model.batch_size:
            indices.extend(np.random.permutation(n_samples))
        idx = [indices.popleft() for _ in range(model.batch_size)]

        batch_samples = train_data_samples[idx]
        batch_label = train_labels[idx]

        _, learning_rate, loss_average, clf_loss, l2_loss = sess.run(

            fetches=[model.op_train,
                     model.learning_rate,

                     model.loss_average,
                     model.ce_loss_average,
                     model.l2_loss_average,

                     ],
            feed_dict={
                model.ph_priors: train_priors,
                model.ph_sample: batch_samples,
                model.ph_label: batch_label,
                model.ph_bn: True,
                model.ph_lr: learning_rate,

            }
        )

        # Periodical evaluation of the model.
        if step % eval_frequency == 0 or step == num_steps:
            learning_rate *= lr_decay_rate

            learning_rate = max(learning_rate, 1e-6)

            epoch = step * model.batch_size / n_samples
            print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, n_epochs))

            print(
                '  learning_rate = {:.2e}, loss = {:.2e}, clf = {:.2e}, l2 = {:.2e}.'.format(
                    learning_rate, loss_average, clf_loss, l2_loss)
            )

            string, accuracy, f1, loss = evaluate(sess, model, val_data_samples, val_priors, val_labels)
            val_accuracies.append(accuracy)
            val_losses.append(loss)
            print('  validation {}'.format(string))
            print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall))

            # Save model parameters (for evaluation).
            model.op_saver.save(sess, logger + "/model.ckpt", global_step=step)

    print('validation accuracy: peak = {:.4f}, mean = {:.4f}'.format(
        max(val_accuracies), np.mean(val_accuracies[-10:])))

    t_step = (time.time() - t_wall) / num_steps
    return val_accuracies, val_losses, t_step


def run(sess, model,
        n_epochs,
        learning_rate, lr_decay_rate,
        train_data_samples, train_prior, train_labels,
        val_data_samples, val_prior, val_labels,
        test_data_samples, test_prior, test_labels,

        logger):
    fit(sess, model,
        n_epochs,
        learning_rate, lr_decay_rate,
        train_data_samples, train_prior, train_labels,
        val_data_samples, val_prior, val_labels,
        logger)

    string, train_accuracy, train_f1, train_loss = evaluate(sess, model,
                                                            train_data_samples, train_prior, train_labels,
                                                            )
    print('train {}'.format(string))

    model.op_saver.restore(sess, tf.train.latest_checkpoint(logger))

    string, val_accuracy, val_f1, val_loss = evaluate(sess, model,
                                                      val_data_samples, val_prior, val_labels
                                                      )
    print('val  {}'.format(string))

    string, test_accuracy, test_f1, test_loss = evaluate(sess, model,
                                                         test_data_samples, test_prior,
                                                         test_labels
                                                         )
    print('test  {}'.format(string))
