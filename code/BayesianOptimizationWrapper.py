from ScratchGPModular import GaussianProcess
from AcquisitionFunction import AcquisitionFunction
from BayesianOptimization import BayesianOptimization
from Functions import FunctionHelper, Custom_Print
import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys

# setting up the global parameters for plotting graphs i.e, graph size and suppress warning from multiple graphs
# being plotted
plt.rcParams["figure.figsize"] = (5, 5)
plt.rcParams['figure.max_open_warning'] = 0
# np.seterr(divide='ignore', invalid='ignore')


# Class for starting Bayesian Optimization with the specified parameters
class BayesianOptimizationWrapper:

    def opt_wrapper(start_time):
        # Required for plotting graphs in the casse of 1D
        linspacexmin = 0
        linspacexmax = 10
        linspaceymin = -3
        linspaceymax = 3

        # Epsilons is the value used during the maximization of PI and EI ACQ functions
        # Greater value like Epsilon = 10 is more of exploration and Epsilon = 0.0001 (exploitation)
        # Epsilon1 used in PI : 3
        epsilon1 = 3
        # epsilon2 used in EI : 4
        epsilon2 = 4

        # Initial value to denote the type of ACQ function to be used, but ignored as all ACQs are run in the sequence
        # acq_fun_list = ['ei', 'pi', 'rs', 'ucb']
        acq_fun_list = ['ei']

        # Number of points required to be observed to evaluate the unknown function ((10:20)*no_dimensions )
        # number_of_iterations = number_of_dimensions * 10
        number_of_iterations = 40

        # List of true functions to choose from
        # 1D functions : sin, cos, custom
        # 2D functions : branin2d, sphere
        # true_func = 'sin'
        # true_func = 'cos'
        true_func = 'custom'
        # true_func = 'sphere'
        # true_func = 'branin2d'
        # true_func = 'hartmann3d'
        # true_func = 'hartmann6d'

        if true_func == 'sin' or true_func == 'cos' or true_func == 'custom':

            # Number of the dimensions being used in the optimization
            number_of_dimensions = 1
            # Different bounds used for different problems
            # bounds used for 1D Sin and custom function
            oned_bounds = [[linspacexmin, linspacexmax]]
            bounds = oned_bounds
            epsilon1 = 0.005
            epsilon2 = 0.005
            number_of_iterations = 10

        elif true_func == 'branin2d':
            number_of_dimensions = 2
            branin_bounds = [[-5, 10], [0, 15]]
            bounds = branin_bounds
            epsilon1 = 0.001
            epsilon2 = 0.001
            number_of_iterations = 40

        elif true_func == 'sphere':
            number_of_dimensions = 2
            sphere_bounds = [[0, 10], [0, 10]]
            bounds = sphere_bounds

        elif true_func == 'hartmann3d':
            number_of_dimensions = 3
            hartmann3d_bounds = [[0, 1] for nd in np.arange(number_of_dimensions)]
            bounds = hartmann3d_bounds
            epsilon1 = 0.01
            epsilon2 = 0.01
            number_of_iterations = 20

        elif (true_func == 'hartmann6d'):
            number_of_dimensions = 6
            hartmann6d_bounds = [[0, 1] for nd in np.arange(number_of_dimensions)]
            bounds = hartmann6d_bounds
            epsilon1 = 0.1
            epsilon2 = 0.1
            number_of_iterations = 20

        # Number of samples for which we have the y values to be known (no_dimensions+1)
        number_of_observed_samples = number_of_dimensions + 1

        # Number of restarts required during the calculation of the maxima during the maximization of acq functions
        number_of_restarts = 15

        # Number of runs the BO has to be run for calculating the regret and the corresponding means and the standard devs
        number_of_runs = 3

        # Type of kernel to be used in the optimization process
        # 0 - Squared Exponential; 1 - Rational Quadratic Function; 2 - Exponential; 3 - Periodic***(To be fixed)
        kernel_type = 0

        # Characteristics of the Kernel being used, currently implemented for the square exponential kernel
        kernel_char = 'ard'

        # Characteristic length scale to be used in the kernel function
        len_scale_bounds = [0.1, 1]

        # charcteristic_length_scale = np.array([1 for nd in np.arange(number_of_dimensions)])
        charcteristic_length_scale = [0.5 for nd in np.arange(number_of_dimensions)]

        # Boolean to specify estimation of length scale
        # params_estimation = False
        params_estimation = True

        # Signal variance bounds
        signal_variance_bounds = [0.1, 1]

        # Initial Signal Variance
        signal_variance = 1

        # Number of unseen data points in order to evaluate the function
        number_of_test_datapoints = 100

        # Noise to be added in the system***
        noise = 0.0

        # To fix the random number genration, currently not able, so as to retain the random selection of points
        random_seed = 500

        # Kappa value to be used during the maximization of UCB ACQ function, but this is overriden in the case if
        # Kappa is calculated at each iteration as a function of the iteration and other parameters
        # kappa=10 is more of exploration and kappa = 0.1 is more of exploitation
        kappa = 0.1

        # Data structure to hold the regrets obtained from each type of ACQ function in each run of the BO
        total_ucb_regret = []
        total_ei_regret = []
        total_pi_regret = []
        total_rs_regret = []

        # Added to generate the results for comparing regrets for spatially varying length scales.
        ei_ard_regret = []
        ei_var_l_regret = []
        ei_fixed_l_regret = []

        #############################################################################
        #############################################################################

        print("\n###################################################################\n")
        timenow = datetime.datetime.now()
        print("Generating results Start time: ", timenow.strftime("%H%M%S_%d%m%Y"))

        # Run Optimization for the specified number of runs
        for i in range(number_of_runs):

            # Create Gaussian Process  object to carry on the prediction and model fitting tasks
            gaussianObject = GaussianProcess(kernel_type, params_estimation, charcteristic_length_scale,
                                             len_scale_bounds, signal_variance, signal_variance_bounds,
                                             number_of_test_datapoints, noise,
                                             random_seed, linspacexmin, linspacexmax, linspaceymin, linspaceymax,
                                             bounds,
                                             number_of_dimensions, number_of_observed_samples, kernel_char)

            # Creating an ACQ object to hold the parameters of the ACQ functions
            acq_func_obj = AcquisitionFunction(None, number_of_restarts, kappa, epsilon1,
                                               epsilon2)

            func_helper_obj = FunctionHelper(true_func)

            # Create a Bayesian Optimization object to hold the parameters of the current Bayesian Optimization settings
            bay_opt_obj = BayesianOptimization('mybayesianobject', gaussianObject, acq_func_obj, func_helper_obj,
                                               number_of_iterations)

            # Generate the random sample points and its true function values, that acts as a prior to our GP

            # Data structure to hold the data
            random_points = []
            X = []

            # Generate specified (number of observed samples) random numbers for each dimension
            for dim in np.arange(number_of_dimensions):
                random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
                                                               number_of_observed_samples).reshape(1,
                                                                                                   number_of_observed_samples)
                random_points.append(random_data_point_each_dim)

            # Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
            random_points = np.vstack(random_points)

            # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
            for sample_num in np.arange(number_of_observed_samples):
                array = []
                for dim_count in np.arange(number_of_dimensions):
                    array.append(random_points[dim_count, sample_num])
                X.append(array)
            X = np.vstack(X)

            # Obtain the function values observed at given randomly generated samples
            y = bay_opt_obj.func_helper_obj.get_true_func_value(X)

            # Fit the gaussian model for the generated sample data
            gaussianObject.gaussian_fit(X, y)

            # Running the optimization for different acquisition values
            # Make sure to reset the Gaussian Process to the initial setting before running the optimization for
            # different acquisition functions, so that comparison can be made
            # Acquisition function types : 'ei', 'ucb', 'pi', 'rs'

            # #Algorithm running for PI acquisition function
            if ('pi' in acq_fun_list):
                acq_type = 'pi'
                bay_opt_obj.acq_func_obj.set_acq_func_type(acq_type)
                # Resetting the GP model because of the above stated reason
                gaussianObject.charcteristic_length_scale = charcteristic_length_scale
                gaussianObject.signal_variance = signal_variance
                gaussianObject.gaussian_fit(X, y)

                # Store the regret obtained in each run so that mean and variance can be calculated to plot the simple regret
                pi_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                total_pi_regret.append(pi_regret_eachrun)

            if ('rs' in acq_fun_list):
                # Algorithm running for random search
                acq_type = 'rs'
                bay_opt_obj.acq_func_obj.set_acq_func_type(acq_type)
                gaussianObject.charcteristic_length_scale = charcteristic_length_scale
                gaussianObject.signal_variance = signal_variance
                gaussianObject.gaussian_fit(X, y)
                rs_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                total_rs_regret.append(rs_regret_eachrun)

            if ('ei' in acq_fun_list):
                # Algorithm running for EI acquisition function
                acq_type = 'ei'
                bay_opt_obj.acq_func_obj.set_acq_func_type(acq_type)

                for iter_type in ['ard', 'var_l', 'fix_l']:
                    gaussianObject.charcteristic_length_scale = charcteristic_length_scale
                    gaussianObject.signal_variance = signal_variance
                    gaussianObject.gaussian_fit(X, y)
                    # total_ei_regret.append(ei_regret_eachrun)         #Commented as it is replaced by other code

                    gaussianObject.kernel_char = iter_type
                    if(iter_type == 'ard'):
                        gaussianObject.params_estimation = True
                        ei_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                        ei_ard_regret.append(ei_regret_eachrun)

                    elif(iter_type == 'var_l'):
                        ei_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                        ei_var_l_regret.append(ei_regret_eachrun)

                    elif (iter_type == 'fix_l'):
                        gaussianObject.params_estimation = False
                        ei_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                        ei_fixed_l_regret.append(ei_regret_eachrun)


            if ('ucb' in acq_fun_list):
                # Algorithm running for UCB acquisition function
                acq_type = 'ucb'
                bay_opt_obj.acq_func_obj.set_acq_func_type(acq_type)
                gaussianObject.charcteristic_length_scale = charcteristic_length_scale
                gaussianObject.signal_variance = signal_variance
                gaussianObject.gaussian_fit(X, y)
                ucb_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                total_ucb_regret.append(ucb_regret_eachrun)

            # plot regret to verify the posterior after each run
            # iterations_axes = np.arange(start = 1, stop = number_of_iterations+1, step = 1)
            # plt.figure("Regret"+str(i+1))
            # plt.clf()
            # plt.plot(iterations_axes, pi_regret_eachrun, 'b')
            # plt.axis([1, number_of_iterations, 0 , 1])
            # plt.title('Regret for run: '+ str(i))
            # plt.savefig('regret_ucb'+str(i)+'.png')

            # $$$$$$$$$$$$$$$$$$$$$$$$        End of each run       $$$$$$$$$$$$$$$$$$$$$$$$$

        # Generate the values for the Xaxis in the simple regret graph to display mean and variance at each evaluation
        # iterations_axes_values = np.arange(start = 1, stop = number_of_iterations+1, step = 1)
        iterations_axes_values = [i + 1 for i in np.arange(number_of_iterations)]
        fig_name = 'Regret_'+str(true_func)+'_iter'+str(number_of_iterations)+'_runs'+str(number_of_runs)+'_'
        plt.figure(str(fig_name))
        plt.clf()
        ax = plt.subplot(111)

        if ('pi' in acq_fun_list):
            # Accumulated values for the regret is used to calculate the mean and deviations at each evaluations/iterations
            # Vertically stack the arrays obtained in each run as a matrix rows and columns to calculate the mean and deviations
            total_pi_regret = np.vstack(total_pi_regret)

            # Calculate the mean and deviations of regrets for each evaluations in each run
            pi_regret_mean = np.mean(total_pi_regret, axis=0)
            pi_regret_std_dev = np.std(total_pi_regret, axis=0)

            # Display the values of the regret to help debugging if required
            print("\n\nTotal PI Regret\n", total_pi_regret, "\n\nPI Regret Mean",
                  pi_regret_mean, "\n\nPI Regret Deviation\n", pi_regret_std_dev)

            # Plot the deviations observed for each of the ACQ type and errorbar if required
            ax.plot(iterations_axes_values, pi_regret_mean, 'g')
            # plt.errorbar(iterations_axes_values, pi_regret_mean, yerr=pi_regret_std_dev)
            plt.gca().fill_between(iterations_axes_values, pi_regret_mean + pi_regret_std_dev,
                                   pi_regret_mean - pi_regret_std_dev, color="green", alpha=0.25, label='PI')


        if ('ei' in acq_fun_list):
            total_ei_regret = np.vstack(total_ei_regret)
            ei_regret_mean = np.mean(total_ei_regret, axis=0)
            ei_regret_std_dev = np.std(total_ei_regret, axis=0)
            print("\n\nTotal EI Regret\n", total_ei_regret, "\n\nEI Regret Mean", ei_regret_mean,
                  "\n\nEI Regret Deviation\n",
                  ei_regret_std_dev)

            ax.plot(iterations_axes_values, ei_regret_mean, 'b')
            # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
            plt.gca().fill_between(iterations_axes_values, ei_regret_mean + ei_regret_std_dev,
                                   ei_regret_mean - ei_regret_std_dev, color="blue", alpha=0.25, label="EI")

        if ('ucb' in acq_fun_list):
            total_ucb_regret = np.vstack(total_ucb_regret)
            ucb_regret_mean = np.mean(total_ucb_regret, axis=0)
            ucb_regret_std_dev = np.std(total_ucb_regret, axis=0)
            print("\n\nTotal UCB Regret\n", total_ucb_regret, "\n\nUCB Regret Mean", ucb_regret_mean,
                  "\n\nUCB Regret Deviation\n", ucb_regret_std_dev)

            ax.plot(iterations_axes_values, ucb_regret_mean, 'r')
            # plt.errorbar(iterations_axes_values, ucb_regret_mean, yerr=ucb_regret_std_dev)
            plt.gca().fill_between(iterations_axes_values, ucb_regret_mean + ucb_regret_std_dev,
                                   ucb_regret_mean - ucb_regret_std_dev, color="red", alpha=0.25, label='UCB')


        if ('rs' in acq_fun_list):
            total_rs_regret = np.vstack(total_rs_regret)
            rs_regret_mean = np.mean(total_rs_regret, axis=0)
            rs_regret_std_dev = np.std(total_rs_regret, axis=0)
            print("\n\nTotal RS Regret\n", total_rs_regret, "\n\nRS Regret Mean", rs_regret_mean,
                  "\n\nRS Regret Deviation\n", rs_regret_std_dev)

            ax.plot(iterations_axes_values, rs_regret_mean, 'y')
            # plt.errorbar(iterations_axes_values, ucb_regret_mean, yerr=ucb_regret_std_dev)
            plt.gca().fill_between(iterations_axes_values, rs_regret_mean + rs_regret_std_dev,
                                   rs_regret_mean - rs_regret_std_dev, color="yellow", alpha=0.25, label='RS')

        timenow = datetime.datetime.now()
        print("\nEnd time: ", timenow.strftime("%H%M%S_%d%m%Y"))

        # Set the parameters of the simple regret graph
        plt.axis([1, number_of_iterations, 0, 1])
        plt.title('Regret for ARD Kernel')
        plt.xlabel('Evaluations')
        plt.ylabel('Simple Regret')
        plt.savefig(fig_name+str(start_time)+'.png')
        legend = ax.legend(loc=1, fontsize='x-small')
        plt.show()



    if __name__ == "__main__":
        timenow = datetime.datetime.now()
        stamp =  timenow.strftime("%H%M%S_%d%m%Y")
        f = open('console_output_'+str(stamp)+'.txt', 'w')
        original = sys.stdout
        sys.stdout = Custom_Print(sys.stdout, f)
        opt_wrapper(stamp)




'''
#################### References ###########################

#different ways of generating data 


# Fixed samples
# X = np.array([[1], [3], [4], [8]]).reshape(4, 1)

# Randomly select samples between the upper bound and the lower bound
# X = np.array(np.random.uniform(linspacexmin, linspacexmax,
#                                number_of_observed_samples)).reshape(number_of_observed_samples, 1)


random_points = []
X = []

for dim in np.arange(number_of_dimensions):
    random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
                                                   number_of_observed_samples).reshape(1, number_of_observed_samples)
    print(random_data_point_each_dim)
    random_points.append(random_data_point_each_dim)

random_points = np.vstack(random_points)

for sample_num in np.arange(number_of_observed_samples):
    array = []
    for dim_count in np.arange(number_of_dimensions):
        array.append(random_points[dim_count, sample_num])
    X.append(array)
X = np.vstack(X)


        # Doesnt work for multi dimensions
        # X =[]
        # for sample_num in np.arange(number_of_observed_samples):
        #     array =[]
        #     for dim_count in np.arange(number_of_dimensions):
        #         value =  np.random.uniform(bounds[dim_count][0], bounds[dim_count][1],1)
        #         array.append(value)
        #     X.append(array)
        #
        # X = np.vstack(X)

### Tried different methods to append the matrix

            # ynew = self.func_helper_obj.get_true_func_value(np.matrix(xnew))
            # y = np.append(self.gp_obj.y, ynew.reshape(-1,self.gp_obj.number_of_dimensions), axis=0)
            # self.gp_obj.gaussian_fit(X, y)

            # X = self.gp_obj.X
            # if( np.array_equal(xnew,np.zeros(self.gp_obj.number_of_dimensions))):
            #     zero_value_bool = True
            #     print('\nzeroes encountered in run: ', run_count, " iteration: ", i + 1)
            #
            # xnew = np.matrix(xnew)
            # ynew = self.func_helper_obj.get_true_func_value(xnew)
            # X = np.append(X, xnew, axis=0)
            #
            # # ynew = np.array(ynew).ravel()
            # y = np.append(self.gp_obj.y, ynew.reshape(-1,self.gp_obj.number_of_dimensions), axis=0)
            # self.gp_obj.gaussian_fit(X, y)
            #
            # print("added value ", X.shape, y.shape)


            # plot_axes = [linspacexmin, linspacexmax, 0, 20]
            # self.acq_func_obj.plot_acquisition_function(i+1, Xs, acq_func_values, plot_axes)



'''