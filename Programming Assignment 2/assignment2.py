#################################
# Your name: May Siva
#################################
import numpy
import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        X = np.random.uniform(0, 1, m)
        X.sort()
        probability = np.array([0.8 if (x <= 0.2 or (x >= 0.4 and x <= 0.6) or x >= 0.8) else 0.1 for x in X])
        Y = np.array([np.random.choice([1, 0], p=[p, 1 - p]) for p in probability])
        return np.column_stack((X, Y))

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
            Calculates the empirical error and the true error.
            Plots the average empirical and true errors.
            Input: m_first - an integer, the smallest size of the data sample in the range.
                   m_last - an integer, the largest size of the data sample in the range.
                   step - an integer, the difference between the size of m in each loop.
                   k - an integer, the maximum number of intervals.
                   T - an integer, the number of times the experiment is performed.

            Returns: np.ndarray of shape (n_steps,2).
                A two dimensional array that contains the average empirical error
                and the average true error for each m in the range accordingly.
            """
        avg_true_error = []
        avg_empirical_error = []
        for i in range(m_first, m_last + 1, step):
            curr_true_error = []
            curr_empirical_error = []
            for j in range(T):
                sample = self.sample_from_D(i)
                x = sample[:, 0]
                y = sample[:, 1]
                list_of_intervals, err_count = intervals.find_best_interval(x, y, k)
                curr_true_error.append(self.calc_true_error(list_of_intervals))
                curr_empirical_error.append(err_count / i)
            avg_true_error.append(sum(curr_true_error) / len(curr_true_error))
            avg_empirical_error.append(sum(curr_empirical_error) / len(curr_empirical_error))

        plt.title("Experiment m range ERM")
        plt.xlabel("n")
        plt.plot(np.arange(m_first, m_last + 1, step), avg_true_error, label="True Error")
        plt.plot(np.arange(m_first, m_last + 1, step), avg_empirical_error, label="Empirical Error")
        plt.legend()
        plt.show()

        return np.column_stack((avg_empirical_error, avg_true_error))

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        true_error = []
        empirical_error = []
        sample = self.sample_from_D(m)
        x = sample[:, 0]
        y = sample[:, 1]
        for i in range(k_first, k_last + 1, step):
            list_of_intervals, err_count = intervals.find_best_interval(x, y, i)
            true_error.append(self.calc_true_error(list_of_intervals))
            empirical_error.append(err_count / m)

        plt.title("Experiment k range ERM")
        plt.xlabel("k")
        plt.plot(np.arange(k_first, k_last + 1, step), true_error, label="True error")
        plt.plot(np.arange(k_first, k_last + 1, step), empirical_error, label="Empirical error")
        plt.legend()
        plt.show()

        return np.argmin(empirical_error) * step + k_first

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """

        sample = self.sample_from_D(m)
        np.random.shuffle(sample)
        training_set = sample[:int(0.8 * m)]
        rest_of_set = sample[int(0.8 * m):]
        training_set = sorted(training_set, key=lambda val: val[0])
        training_set = np.array(training_set)
        x_train = training_set[:, 0]
        y_train = training_set[:, 1]
        empirical_error = []
        best_intervals = []

        for i in range(1, 11):
            list_of_intervals, err_count = intervals.find_best_interval(x_train, y_train, i)
            empirical_error.append(self.calc_empirical_holdout_error(list_of_intervals, rest_of_set))
            best_intervals.append(list_of_intervals)

        lst_touple = list(zip(empirical_error, best_intervals))
        lst_touple = sorted(lst_touple, key=lambda x: x[0])

        print("The best interval is:\n " + str(lst_touple[0][1]))


        return np.argmin(empirical_error) + 1

        #################################

    # Place for additional methods

    def calc_true_error(self, list_of_intervals):
        """This function gets a list of intervals and calculates the true error of the returned hypothesis.
        The first interval is [0,0.2] ∪ [0.4,0.6] ∪ [0.8,1].
        The second interval is [0.2,0.4] ∪ [0.6,0.8] """
        sum_first = 0
        sum_second = 0
        sum_first += self.calc_size_of_interval(list_of_intervals, 0, 0.2)
        sum_second += self.calc_size_of_interval(list_of_intervals, 0.2, 0.4)
        sum_first += self.calc_size_of_interval(list_of_intervals, 0.4, 0.6)
        sum_second += self.calc_size_of_interval(list_of_intervals, 0.6, 0.8)
        sum_first += self.calc_size_of_interval(list_of_intervals, 0.8, 1)

        return 0.2 * sum_first + 0.8 * (0.6 - sum_first) + 0.9 * sum_second + (0.4 - sum_second) * 0.1

    def calc_size_of_interval(self, list_of_intervals, start_interval, end_interval):
        """This function gets a list of intervals and a start and end of a given interval
        and calculates the intersection between the given interval to each interval in the given list"""
        curr_sum = 0
        for curr_interval in list_of_intervals:
            start = max(start_interval, curr_interval[0])
            end = min(end_interval, curr_interval[1])
            if start < end:
                curr_sum += (end - start)

        return curr_sum

    def is_in_interval(self, list_of_intervals, x):
        """This function gets a list of intervals and an integer x
        and checks if x is in one of the intervals of the list"""
        for interval in list_of_intervals:
            if interval[0] <= x <= interval[1]:
                return True
        return False

    def calc_empirical_holdout_error(self, list_of_intervals, sample):
        """This function gets a list of intervals and a sample and calculates the empirical error"""
        curr_sum = 0
        for x, y in sample:
            if (not (self.is_in_interval(list_of_intervals, x)) and y == 1) or (
                    self.is_in_interval(list_of_intervals,
                                        x) and y == 0):
                curr_sum += 1
        return curr_sum / len(sample)

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)
